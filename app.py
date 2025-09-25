# app.py
import os
import re
import json
import time
import pathlib
import difflib
import logging
from typing import Dict, Any, List, Tuple, Optional

import httpx
from slugify import slugify
from fastapi import FastAPI, Header, HTTPException, Request, Query
from pydantic import BaseModel
from openai import OpenAI

# ========= Config / ENV =========
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # usar guion, no punto
SERVICE_TOKEN = os.getenv("SERVICE_TOKEN", "avbrief123")

ALLOW_WEB = os.getenv("ALLOW_WEB", "1") == "1"
WEB_TIMEOUT = float(os.getenv("WEB_TIMEOUT", "8.0"))
MAX_WEB_BYTES = int(os.getenv("MAX_WEB_BYTES", "180_000"))

DEBUG_ENV = os.getenv("DEBUG", "0") == "1"

if not OPENAI_API_KEY:
    raise RuntimeError("Falta OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

ROOT = pathlib.Path(__file__).resolve().parent
TPL_PATH = ROOT / "Plantilla_MD.md"

# ========= Logging =========
logger = logging.getLogger("brief-filler")
handler = logging.StreamHandler()
formatter = logging.Formatter(
    fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG if DEBUG_ENV else logging.INFO)

# ========= FastAPI =========
app = FastAPI(title="AV Brief Filler", version="1.0.8")

# ========= Regex para detectar líneas tipo "Etiqueta:" =========
LABEL_RE = re.compile(
    r"""^(\s*(?:[-*+]\s+|\d+\.\s+)?)      # bullet opcional
        (?:\*\*)?                         # ** opcional
        ([^:*]+?)                         # etiqueta
        (?:\*\*)?
        \s*:\s*$                          # termina en :
    """,
    re.X,
)

# ========= Modelos =========
class FillResponse(BaseModel):
    markdown: str

class KeysResponse(BaseModel):
    keys: List[str]

# ========= Utilidades plantilla =========
def read_template() -> Tuple[str, List[str]]:
    if not TPL_PATH.exists():
        raise RuntimeError("No se encontró Plantilla_MD.md en la raíz del proyecto.")
    text = TPL_PATH.read_text(encoding="utf-8")
    return text, text.splitlines()

def extract_fields(lines: List[str]) -> List[Dict[str, Any]]:
    fields: List[Dict[str, Any]] = []
    for i, ln in enumerate(lines):
        m = LABEL_RE.match(ln)
        if m:
            raw = m.group(2).strip()
            key = slugify(raw, separator="_")
            fields.append({"line_idx": i, "raw_label": raw, "key": key})
    return fields

def ensure_value(v: Any) -> str:
    if v is None:
        return "Sin datos"
    if isinstance(v, (dict, list)):
        try:
            v = json.dumps(v, ensure_ascii=False)
        except Exception:
            v = str(v)
    v = str(v).strip()
    return v if v else "Sin datos"

def assemble_markdown(template_lines: List[str], fields: List[Dict[str, Any]], data: Dict[str, Any]) -> str:
    out = template_lines[:]
    for f in fields:
        line = out[f["line_idx"]]
        val = ensure_value(data.get(f["key"], ""))
        idx = line.rfind(":")
        if idx == -1:
            base = line.rstrip()
            out[f["line_idx"]] = base + " " + val
        else:
            base = line[: idx + 1]
            out[f["line_idx"]] = base + " " + val
    return "\n".join(out) + "\n"

# ========= Normalización =========
def normalize_user_data_to_keys(user_data: Dict[str, Any], expected_keys: List[str]) -> Dict[str, Any]:
    norm_expected = {slugify(k, separator="_"): k for k in expected_keys}
    out = {k: "Sin datos" for k in expected_keys}

    for k_in, v in (user_data or {}).items():
        slug_in = slugify(str(k_in), separator="_")
        if slug_in in norm_expected:
            out[norm_expected[slug_in]] = v
            continue
        best = difflib.get_close_matches(slug_in, list(norm_expected.keys()), n=1, cutoff=0.8)
        if best:
            out[norm_expected[best[0]]] = v
    return out

# ========= Ingesta flexible =========
async def ingest_user_data(request: Request) -> Dict[str, Any]:
    # 1) JSON
    try:
        payload = await request.json()
        if isinstance(payload, dict):
            if "user_data" in payload and isinstance(payload["user_data"], dict):
                return payload["user_data"]
            if payload:
                return payload
    except Exception:
        pass
    # 2) Query
    try:
        qs = dict(request.query_params)
        if "user_data" in qs:
            try:
                ud = json.loads(qs["user_data"])
                if isinstance(ud, dict):
                    return ud
            except Exception:
                return {}
        elif qs:
            return qs
    except Exception:
        pass
    # 3) Form
    try:
        form = await request.form()
        if "user_data" in form:
            try:
                ud = json.loads(form["user_data"])
                if isinstance(ud, dict):
                    return ud
            except Exception:
                return {}
        elif form:
            return dict(form)
    except Exception:
        pass
    return {}

# ========= Scrape sencillo =========
def _strip_html(text: str) -> str:
    # recorta scripts/estilos rudimentariamente y tags
    text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return re.sub(r"\n{2,}", "\n", text).strip()

def fetch_site_text(url: str, timeout: float = WEB_TIMEOUT, max_bytes: int = MAX_WEB_BYTES) -> str:
    if not url:
        return ""
    if not url.startswith("http"):
        url = "https://" + url
    headers = {
        "User-Agent": "AVBriefFiller/1.0 (+https://master-brief.onrender.com)"
    }
    try:
        with httpx.Client(timeout=timeout, headers=headers, follow_redirects=True) as s:
            r = s.get(url)
            status = r.status_code
            raw = r.content[:max_bytes]
            ctype = r.headers.get("content-type", "")
            logger.info(f"WEB fetch status={status} bytes={len(raw)} type={ctype} url={url}")
            if status >= 400 or not raw:
                return ""
            if "text" in ctype or "html" in ctype or ctype == "":
                try:
                    text = raw.decode(r.encoding or "utf-8", errors="ignore")
                except Exception:
                    text = raw.decode("utf-8", errors="ignore")
                return _strip_html(text)
            # si no es texto, devolvemos vacío
            return ""
    except Exception as e:
        logger.warning(f"WEB fetch error url={url} err={repr(e)}")
        return ""

# ========= OpenAI rules =========
SYSTEM_RULES = """Sos un asistente que rellena un brief y responde SOLO en JSON (objeto).
Reglas:
- Prioridad: (1) Usuario, (2) Sitio oficial, (3) Secundarias confiables.
- Si falta info: devolver el string EXACTO "Sin datos".
- Si hay contradicción fuerte con el usuario: usar el dato del usuario y agregar " (verificar internamente)".
- No inventes datos sensibles ni números sin evidencia.
- Devolvé un JSON con EXACTAMENTE las KEYS indicadas (sin keys extra).
"""

def call_model_to_get_json(fields: List[Dict[str, Any]], payload: Dict[str, Any], web_ctx: str = "") -> Dict[str, str]:
    keys_list = [f["key"] for f in fields]

    # Armamos el mensaje de usuario
    body = {
        "keys": keys_list,
        "user_data": payload or {},
        "web_context_excerpt": (web_ctx[:6000] if web_ctx else ""),  # cap de contexto
    }
    user_text = json.dumps(body, ensure_ascii=False)

    messages = [
        {"role": "system", "content": SYSTEM_RULES},
        {"role": "user", "content": (
            "Completá EXACTAMENTE estas keys (sin agregar nuevas). "
            "Si no hay dato, devolvé 'Sin datos'. "
            "Respondé SOLO JSON válido (sin texto adicional).\n\n"
            + user_text
        )},
    ]

    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0
        )
        took = (time.time() - t0) * 1000
        raw = resp.choices[0].message.content
        data = {}
        try:
            data = json.loads(raw or "{}")
        except Exception as je:
            logger.warning(f"OpenAI JSON parse error: {repr(je)} raw_len={len(raw) if raw else 0}")
            data = {}

        # Sanitiza: solo keys esperadas
        out = {k: ensure_value(data.get(k, "Sin datos")) for k in keys_list}

        # Métrica de completitud
        filled = sum(1 for k in out if out[k] != "Sin datos")
        logger.info(f"OAI ok model={OPENAI_MODEL} took_ms={took:.0f} keys={len(keys_list)} filled={filled}")
        if DEBUG_ENV:
            sample = {k: out[k] for k in list(out.keys())[:6]}
            logger.debug(f"OAI sample_out={json.dumps(sample, ensure_ascii=False)}")

        return out
    except Exception as e:
        logger.error(f"OpenAI error: {repr(e)}")
        # Fallback: todo Sin datos
        return {k: "Sin datos" for k in keys_list}

# ========= Endpoints =========
@app.get("/health")
def health():
    return {"ok": True, "version": "1.0.8", "model": OPENAI_MODEL}

@app.get("/brief/keys", response_model=KeysResponse)
def brief_keys():
    _, lines = read_template()
    fields = extract_fields(lines)
    return KeysResponse(keys=[f["key"] for f in fields])

@app.post("/brief/fill", response_model=FillResponse)
async def fill_brief(
    request: Request,
    authorization: str = Header(None),
    debug: Optional[int] = Query(default=0),
    allow_ai: Optional[int] = Query(default=0),
):
    # Auth tipo Bearer
    expected = f"Bearer {SERVICE_TOKEN}" if SERVICE_TOKEN else None
    if expected and authorization != expected:
        logger.warning("401 Unauthorized attempt")
        raise HTTPException(status_code=401, detail="Unauthorized")
    
        # --- debug crudo de la solicitud ---
    try:
        body_bytes = await request.body()
        print("DEBUG raw headers:", dict(request.headers))
        print("DEBUG raw query:", dict(request.query_params))
        print("DEBUG raw body bytes len:", len(body_bytes))
        print("DEBUG raw body (trunc 500):", body_bytes[:500])
    except Exception as e:
        print("DEBUG error leyendo body:", repr(e))
    # -----------------------------------
    
    # 1) Ingesta
    user_data = await ingest_user_data(request)
    try:
        logger.info(f"INGEST keys_in={len(user_data)} sample={list(user_data.keys())[:5]}")
    except Exception:
        pass
    # -----------------------------------
    print("DEBUG user_data (post-ingest):", {k: user_data.get(k) for k in list(user_data.keys())[:8]})

    # -----------------------------------
    # 2) Plantilla / keys oficiales
    template_text, template_lines = read_template()
    fields = extract_fields(template_lines)
    if not fields:
        raise HTTPException(status_code=400, detail="No se detectaron campos en la plantilla")
    expected_keys = [f["key"] for f in fields]
    logger.info(f"TEMPLATE fields_detected={len(expected_keys)}")

    # 3) Normalizar
    normalized = normalize_user_data_to_keys(user_data, expected_keys)
    if debug or DEBUG_ENV:
        # Mostrar pequeño sample sólo en logs
        norm_sample = {k: normalized[k] for k in expected_keys[:5]}
        logger.debug(f"NORMALIZED sample={json.dumps(norm_sample, ensure_ascii=False)}")

    # 4) Contexto web (opcional)
    web_ctx = ""
    website = normalized.get("website", "")
    if ALLOW_WEB and allow_ai:
        web_ctx = fetch_site_text(website, timeout=WEB_TIMEOUT, max_bytes=MAX_WEB_BYTES)
        logger.info(f"WEB_CTX chars={len(web_ctx)} src={website}")

    # 5) Llamada al modelo → JSON exact keys
    data = call_model_to_get_json(fields, normalized, web_ctx=web_ctx)

    # 6) Volcar en plantilla
    md = assemble_markdown(template_lines, fields, data)

    # 7) Métrica final
    filled = sum(1 for k in data if data[k] != "Sin datos")
    logger.info(f"FILL result filled_keys={filled}/{len(data)}")

    return FillResponse(markdown=md)
