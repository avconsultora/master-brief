# app.py
import os
import re
import json
import pathlib
import difflib
from typing import Dict, Any, List, Tuple

from slugify import slugify
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel

# ========= Config / Paths =========
SERVICE_TOKEN = os.getenv("SERVICE_TOKEN", "avbrief123")
USE_LLM = os.getenv("USE_LLM", "false").lower() == "true"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

ROOT = pathlib.Path(__file__).resolve().parent
TPL_PATH = ROOT / "Plantilla_MD.md"

app = FastAPI(title="AV Brief Filler", version="1.0.6")

# ========= (Opcional) Cliente OpenAI si USE_LLM =========
client = None
if USE_LLM:
    if not OPENAI_API_KEY:
        raise RuntimeError("Falta OPENAI_API_KEY y USE_LLM=true")
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

# ========= Regex para detectar líneas tipo "Etiqueta:" =========
LABEL_RE = re.compile(
    r"""^(\s*(?:[-*+]\s+|\d+\.\s+)?)      # bullet opcional
        (?:\*\*)?                         # ** opcional
        ([^:*]+?)                         # etiqueta (sin los dos puntos)
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
    """
    Devuelve dicts con posición de línea y key slug:
      [{"line_idx": int, "raw_label": "Cliente/Marca", "key": "cliente_marca"}, ...]
    """
    fields: List[Dict[str, Any]] = []
    for i, ln in enumerate(lines):
        m = LABEL_RE.match(ln)
        if m:
            raw = m.group(2).strip()
            key = slugify(raw, separator="_")
            fields.append({"line_idx": i, "raw_label": raw, "key": key})
    return fields

def ensure_value(v: Any) -> str:
    """Devuelve 'Sin datos' si viene vacío. Dict/list -> JSON string."""
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
    """
    Inserta cada valor en la MISMA línea (después de ':') manteniendo la plantilla.
    """
    out = template_lines[:]  # copia superficial
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
    """
    Acepta alias (con/sin acentos, variantes) y los mapea a las keys exactas de expected_keys.
    - slugify(k_in) -> match exacto
    - si no, best-match difflib con cutoff 0.8
    - inicia todas las keys en "Sin datos"
    """
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
    """
    Ingesta flexible:
      1) JSON: {"user_data": {...}} o body plano {...}
      2) Query: ?user_data={...} o ?key=value
      3) Form: user_data={...} o campos sueltos
    """
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

    # 2) Querystring
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

# ========= (Opcional) Reglas y llamada a LLM =========
SYSTEM_RULES = """Sos un asistente que rellena un brief y responde SOLO en JSON (objeto).
Reglas:
- Prioridad: (1) Usuario, (2) Sitio oficial, (3) Secundarias confiables.
- Si falta info: devolver el string EXACTO "Sin datos".
- Si hay contradicción fuerte con el usuario: usar el dato del usuario y agregar " (verificar internamente)" al final del valor.
- No inventes datos sensibles ni números sin evidencia.
- Devolvé un JSON con EXACTAMENTE las KEYS indicadas (sin keys extra).
"""

def call_llm_to_complete(fields: List[Dict[str, Any]], payload: Dict[str, Any]) -> Dict[str, str]:
    if not USE_LLM or client is None:
        # “Passthrough”: devuelve tal cual (faltantes ya están en “Sin datos”)
        keys_list = [f["key"] for f in fields]
        return {k: ensure_value(payload.get(k, "Sin datos")) for k in keys_list}

    keys_list = [f["key"] for f in fields]
    user_text = json.dumps(payload, ensure_ascii=False, indent=2)

    messages = [
        {"role": "system", "content": SYSTEM_RULES},
        {"role": "user", "content":
            "Estas son las KEYS a completar (usar EXACTAMENTE estas, sin agregar otras):\n"
            + ", ".join(keys_list)
            + "\n\nDatos del usuario/hallazgos (pueden estar incompletos):\n"
            + user_text
            + "\n\nDevolvé SOLO JSON válido (sin texto adicional)."}
    ]

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
        print("ERROR OpenAI:", repr(e))
        return {k: ensure_value(payload.get(k, "Sin datos")) for k in keys_list}

# ========= Endpoints =========
@app.get("/health")
def health():
    return {"ok": True, "version": "1.0.6"}

@app.get("/brief/keys", response_model=KeysResponse)
def brief_keys():
    _, lines = read_template()
    fields = extract_fields(lines)
    return KeysResponse(keys=[f["key"] for f in fields])

@app.post("/brief/fill", response_model=FillResponse)
async def fill_brief(request: Request, authorization: str = Header(None)):
    # Auth tipo Bearer
    expected = f"Bearer {SERVICE_TOKEN}" if SERVICE_TOKEN else None
    if expected and authorization != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # 1) Ingesta de user_data
    user_data = await ingest_user_data(request)
    try:
        print("DEBUG /brief/fill keys (ingest):", list(user_data.keys())[:8])
    except Exception:
        pass

    # 2) Leer plantilla y extraer keys oficiales
    _, template_lines = read_template()
    fields = extract_fields(template_lines)
    if not fields:
        raise HTTPException(
            status_code=400,
            detail="No se detectaron campos 'Etiqueta:' que terminen con ':' en la plantilla",
        )
    expected_keys = [f["key"] for f in fields]

    # 3) Normalizar alias -> keys oficiales y completar faltantes con "Sin datos"
    normalized = normalize_user_data_to_keys(user_data, expected_keys)

    # 4) Passthrough o completar con LLM (según USE_LLM)
    data = call_llm_to_complete(fields, normalized)

    # 5) Volcar en plantilla (espejo 1:1, misma línea)
    md = assemble_markdown(template_lines, fields, data)
    return FillResponse(markdown=md)