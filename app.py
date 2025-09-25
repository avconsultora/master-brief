# app.py
import os
import re
import json
import pathlib
import difflib
import logging
from typing import Dict, Any, List, Tuple

from slugify import slugify
from fastapi import FastAPI, Header, HTTPException, Request, Query
from pydantic import BaseModel
from openai import OpenAI

# ========= Logging =========
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("av-brief")

# ========= Config =========
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
SERVICE_TOKEN = os.getenv("SERVICE_TOKEN", "avbrief123")

# autocorrección común: gpt-4o.mini -> gpt-4o-mini
if "." in OPENAI_MODEL and "-" not in OPENAI_MODEL:
    corrected = OPENAI_MODEL.replace(".", "-")
    logger.warning("OPENAI_MODEL corregido de '%s' a '%s'", OPENAI_MODEL, corrected)
    OPENAI_MODEL = corrected

if not OPENAI_API_KEY:
    raise RuntimeError("Falta OPENAI_API_KEY")

try:
    client = OpenAI(api_key=OPENAI_API_KEY)
    logger.info("OpenAI client inicializado. Modelo: %s", OPENAI_MODEL)
except Exception as e:
    logger.exception("No se pudo inicializar OpenAI client: %r", e)
    raise

ROOT = pathlib.Path(__file__).resolve().parent
TPL_PATH = ROOT / "Plantilla_MD.md"

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

# ========= Utilidades =========
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

# ========= OpenAI rules =========
SYSTEM_RULES = """Sos un asistente que rellena un brief y responde SOLO en JSON (objeto).
Reglas:
- Prioridad: (1) Usuario, (2) Sitio oficial, (3) Secundarias confiables.
- Si falta info: devolvé el string EXACTO "Sin datos".
- Si hay contradicción fuerte con el usuario: usá el dato del usuario y agregá " (verificar internamente)".
- No inventes datos sensibles ni números sin evidencia.
- Devolvé un JSON con EXACTAMENTE las KEYS indicadas (sin keys extra).
"""

def call_model_to_get_json(fields: List[Dict[str, Any]], payload: Dict[str, Any], debug: bool = False) -> Dict[str, str]:
    keys_list = [f["key"] for f in fields]
    user_text = json.dumps(payload, ensure_ascii=False, indent=2)

    prompt = (
        "Estas son las KEYS a completar (usar EXACTAMENTE estas, sin agregar otras):\n"
        + ", ".join(keys_list)
        + "\n\nDatos del usuario/hallazgos (pueden estar incompletos):\n"
        + user_text
        + "\n\nDevolvé SOLO JSON válido (sin texto adicional)."
    )

    messages = [
        {"role": "system", "content": SYSTEM_RULES},
        {"role": "user", "content": prompt},
    ]

    if debug:
        logger.info("LLM prompt keys=%d payload_non_empty=%d", len(keys_list), sum(1 for v in payload.values() if v and v != "Sin datos"))

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        raw = resp.choices[0].message.content
        if debug:
            logger.info("LLM raw (primeros 500 chars): %s", (raw or "")[:500])
        data = json.loads(raw)
        return {k: ensure_value(data.get(k, "Sin datos")) for k in keys_list}
    except Exception as e:
        logger.exception("ERROR OpenAI (model=%s): %r", OPENAI_MODEL, e)
        return {k: "Sin datos" for k in keys_list}

# ========= Endpoints =========
@app.get("/health")
def health():
    return {
        "ok": True,
        "version": "1.0.8",
        "model": OPENAI_MODEL,
        "openai_configured": bool(OPENAI_API_KEY),
    }

@app.get("/brief/keys", response_model=KeysResponse)
def brief_keys():
    _, lines = read_template()
    fields = extract_fields(lines)
    return KeysResponse(keys=[f["key"] for f in fields])

@app.post("/brief/fill", response_model=FillResponse)
async def fill_brief(
    request: Request,
    authorization: str = Header(None),
    debug: int = Query(0, description="Set 1 para loguear prompts/respuestas LLM"),
):
    # Auth
    expected = f"Bearer {SERVICE_TOKEN}" if SERVICE_TOKEN else None
    if expected and authorization != expected:
        logger.warning("401 Unauthorized: authorization header incorrecto")
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Ingesta
    user_data = await ingest_user_data(request)
    logger.info("Ingest user_data keys=%d sample=%s", len(user_data), list(user_data.keys())[:6])

    # Plantilla y fields
    _, template_lines = read_template()
    fields = extract_fields(template_lines)
    if not fields:
        raise HTTPException(status_code=400, detail="No se detectaron campos en la plantilla")

    # Normalizar y completar con IA
    expected_keys = [f["key"] for f in fields]
    normalized = normalize_user_data_to_keys(user_data, expected_keys)
    non_empty = sum(1 for v in normalized.values() if v and v != "Sin datos")
    logger.info("Normalized keys=%d non_empty=%d", len(normalized), non_empty)

    data = call_model_to_get_json(fields, normalized, debug=bool(debug))

    # Ensamblar markdown
    md = assemble_markdown(template_lines, fields, data)
    return FillResponse(markdown=md)
