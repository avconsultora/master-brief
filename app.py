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
SERVICE_TOKEN = os.getenv("SERVICE_TOKEN", "avbrief123")  # default útil en dev
ROOT = pathlib.Path(__file__).resolve().parent
TPL_PATH = ROOT / "Plantilla_MD.md"

app = FastAPI(title="AV Brief Filler", version="1.0.5")

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
            # por compatibilidad si alguna línea terminó sin ':'
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
        # si no hay match, se ignora la key desconocida
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

# ========= Endpoints =========
@app.get("/health")
def health():
    return {"ok": True, "version": "1.0.5"}

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
    template_text, template_lines = read_template()
    fields = extract_fields(template_lines)
    if not fields:
        raise HTTPException(
            status_code=400,
            detail="No se detectaron campos 'Etiqueta:' que terminen con ':' en la plantilla",
        )
    expected_keys = [f["key"] for f in fields]

    # 3) Normalizar alias -> keys oficiales y completar faltantes con "Sin datos"
    normalized = normalize_user_data_to_keys(user_data, expected_keys)
    data = {k: ensure_value(normalized.get(k, "Sin datos")) for k in expected_keys}

    # 4) Volcar en plantilla (espejo 1:1, misma línea)
    md = assemble_markdown(template_lines, fields, data)
    return FillResponse(markdown=md)