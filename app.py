# app.py
import os, re, json, pathlib, difflib, logging
from typing import Dict, Any, List, Tuple, Optional

import httpx
from slugify import slugify
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel
from openai import OpenAI

# ===== Logging =====
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("av-brief")

# ===== Config =====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # <- GUION, no punto
SERVICE_TOKEN = os.getenv("SERVICE_TOKEN", "avbrief123")

if not OPENAI_API_KEY:
    raise RuntimeError("Falta OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

ROOT = pathlib.Path(__file__).resolve().parent
TPL_PATH = ROOT / "Plantilla_MD.md"

app = FastAPI(title="AV Brief Filler", version="1.0.8")

LABEL_RE = re.compile(
    r"""^(\s*(?:[-*+]\s+|\d+\.\s+)?) (?:\*\*)? ([^:*]+?) (?:\*\*)? \s*:\s*$""",
    re.X,
)

class FillResponse(BaseModel):
    markdown: str

class KeysResponse(BaseModel):
    keys: List[str]

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
        try: v = json.dumps(v, ensure_ascii=False)
        except Exception: v = str(v)
    v = str(v).strip()
    return v if v else "Sin datos"

def assemble_markdown(template_lines: List[str], fields: List[Dict[str, Any]], data: Dict[str, Any]) -> str:
    out = template_lines[:]
    for f in fields:
        line = out[f["line_idx"]]
        val = ensure_value(data.get(f["key"], "Sin datos"))
        idx = line.rfind(":")
        out[f["line_idx"]] = (line.rstrip() + " " + val) if idx == -1 else (line[:idx+1] + " " + val)
    return "\n".join(out) + "\n"

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

async def ingest_user_data(request: Request) -> Dict[str, Any]:
    # JSON
    try:
        payload = await request.json()
        if isinstance(payload, dict):
            if "user_data" in payload and isinstance(payload["user_data"], dict):
                return payload["user_data"]
            if payload:
                return payload
    except Exception:
        pass
    # Query
    try:
        qs = dict(request.query_params)
        if "user_data" in qs:
            try:
                ud = json.loads(qs["user_data"])
                if isinstance(ud, dict): return ud
            except Exception:
                return {}
        elif qs:
            return qs
    except Exception:
        pass
    # Form
    try:
        form = await request.form()
        if "user_data" in form:
            try:
                ud = json.loads(form["user_data"])
                if isinstance(ud, dict): return ud
            except Exception:
                return {}
        elif form:
            return dict(form)
    except Exception:
        pass
    return {}

def strip_html(html: str) -> str:
    # súper básico: fuera scripts/styles y tags
    html = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
    html = re.sub(r"(?is)<[^>]+>", " ", html)
    html = re.sub(r"\s+", " ", html)
    return html.strip()

async def fetch_site_text(url: str, timeout_s: int = 8) -> Optional[str]:
    if not url: return None
    if not url.startswith("http"):
        url = "https://" + url
    try:
        async with httpx.AsyncClient(timeout=timeout_s, follow_redirects=True) as http:
            r = await http.get(url)
            if r.status_code >= 400:
                log.warning("fetch_site_text: status=%s url=%s", r.status_code, url)
                return None
            text = strip_html(r.text)
            # recortamos a 15k chars para el prompt
            return text[:15000]
    except Exception as e:
        log.warning("fetch_site_text error: %r", e)
        return None

SYSTEM_RULES = """Sos un asistente que rellena un brief y responde SOLO en JSON (objeto).
Reglas:
- Prioridad: (1) Usuario, (2) Sitio oficial provisto en 'website' (si hay), (3) Secundarias confiables.
- Si falta info: devolver el string EXACTO "Sin datos".
- Si hay contradicción fuerte con el usuario: usar el dato del usuario y agregar " (verificar internamente)".
- No inventes datos sensibles ni números sin evidencia.
- Devolvé un JSON con EXACTAMENTE las KEYS indicadas (sin keys extra).
"""

def call_model_to_get_json(keys_list: List[str], user_payload: Dict[str, Any], web_context: Optional[str]) -> Dict[str, str]:
    user_text = json.dumps(user_payload, ensure_ascii=False, indent=2)
    user_msg = (
        "Estas son las KEYS a completar (usar EXACTAMENTE estas, sin agregar otras):\n"
        + ", ".join(keys_list)
        + "\n\nDatos del usuario (pueden estar incompletos):\n"
        + user_text
        + "\n\nSi hay CONTEXTO_WEB, usalo como fuente secundaria. Devolvé SOLO JSON válido."
    )
    messages = [{"role": "system", "content": SYSTEM_RULES},
                {"role": "user", "content": user_msg}]
    if web_context:
        messages.append({"role": "user", "content": "CONTEXTO_WEB (texto plano del sitio oficial):\n" + web_context})

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0
        )
        raw = resp.choices[0].message.content
        data = json.loads(raw)
        return {k: ensure_value(data.get(k, "Sin datos")) for k in keys_list}
    except Exception as e:
        log.warning("OpenAI error: %r", e)
        # Fallback: NO pisar los valores del usuario
        return {k: ensure_value(user_payload.get(k, "Sin datos")) for k in keys_list}

@app.get("/health")
def health():
    return {"ok": True, "version": "1.0.8", "model": OPENAI_MODEL}

@app.get("/brief/keys", response_model=KeysResponse)
def brief_keys():
    _, lines = read_template()
    fields = extract_fields(lines)
    return KeysResponse(keys=[f["key"] for f in fields])

@app.post("/brief/fill", response_model=FillResponse)
async def fill_brief(request: Request, authorization: str = Header(None)):
    expected = f"Bearer {SERVICE_TOKEN}" if SERVICE_TOKEN else None
    if expected and authorization != expected:
        log.warning("401 Unauthorized")
        raise HTTPException(status_code=401, detail="Unauthorized")

    # flags de control
    qs = dict(request.query_params)
    debug = qs.get("debug") in ("1","true","True","yes")
    allow_ai_flag = False

    # ingesta
    user_data_in = await ingest_user_data(request)
    # también permitimos allow_ai en el body
    try:
        body = await request.json()
        if isinstance(body, dict):
            allow_ai_flag = bool(body.get("allow_ai", False))
    except Exception:
        pass
    if qs.get("allow_ai") in ("1","true","True","yes"):
        allow_ai_flag = True

    # plantilla
    _, template_lines = read_template()
    fields = extract_fields(template_lines)
    expected_keys = [f["key"] for f in fields]

    # normalizar y loggear
    normalized = normalize_user_data_to_keys(user_data_in, expected_keys)
    if debug:
        sample = {k: normalized.get(k) for k in ["cliente_marca","website","sector","que_venden_1_frase"] if k in normalized}
        log.info("INGEST allow_ai=%s normalized_sample=%s", allow_ai_flag, sample)

    # contexto web (opcional)
    web_ctx = None
    if allow_ai_flag:
        website = (normalized.get("website") or "").strip()
        if website and website != "Sin datos":
            web_ctx = await fetch_site_text(website)
            if debug:
                log.info("WEB_CTX chars=%s from=%s", len(web_ctx) if web_ctx else 0, website)

    # IA (con fallback amistoso a user_data)
    model_out = call_model_to_get_json(expected_keys, normalized, web_ctx)

    # Merge “usuario primero, IA después” para no perder inputs usuario si IA puso "Sin datos".
    merged: Dict[str, Any] = {}
    for k in expected_keys:
        user_val = ensure_value(normalized.get(k, "Sin datos"))
        ai_val = ensure_value(model_out.get(k, "Sin datos"))
        merged[k] = user_val if user_val != "Sin datos" else ai_val

    if debug:
        filled = sum(1 for k,v in merged.items() if v != "Sin datos")
        log.info("FILL result filled_keys=%d/%d", filled, len(expected_keys))

    md = assemble_markdown(template_lines, fields, merged)
    return FillResponse(markdown=md)
