# app.py
import os
import re
import json
import pathlib
from typing import Any, Dict

from dotenv import load_dotenv
from slugify import slugify
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from openai import OpenAI

# ==== Carga de entorno ====
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # seguro y disponible
SERVICE_TOKEN = os.getenv("SERVICE_TOKEN", "")

if not OPENAI_API_KEY:
    raise RuntimeError("Falta OPENAI_API_KEY en .env")

client = OpenAI(api_key=OPENAI_API_KEY)

# ==== Paths ====
ROOT = pathlib.Path(__file__).resolve().parent
TPL_PATH = ROOT / "Plantilla_MD.md"

# ==== App ====
app = FastAPI(title="AV Brief Filler", version="1.0.0")

# ==== Regex helpers ====
HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$")
LABEL_RE  = re.compile(
    r"""^(\s*(?:[-*+]\s+|\d+\.\s+)?)      # bullet opcional
        (?:\*\*)?                         # ** opcional
        ([^:*]+?)                         # etiqueta
        (?:\*\*)?                         
        \s*:\s*$                          # termina en :
    """, re.X)

# ==== Modelos ====
class FillRequest(BaseModel):
    user_data: Dict[str, Any] = {}

class FillResponse(BaseModel):
    markdown: str

# ==== Utilidades de plantilla ====
def read_template():
    if not TPL_PATH.exists():
        raise RuntimeError("No se encontró Plantilla_MD.md en la raíz del proyecto.")
    text = TPL_PATH.read_text(encoding="utf-8")
    return text, text.splitlines()

def extract_fields(lines):
    """
    Devuelve lista de campos con forma:
      [{"line_idx": int, "raw_label": "Cliente/Marca", "key": "cliente_marca"}, ...]
    Se detecta TODA línea que termine con ":" (sea bullet o no).
    """
    fields = []
    for i, ln in enumerate(lines):
        m = LABEL_RE.match(ln)
        if m:
            raw = m.group(2).strip()
            key = slugify(raw, separator="_")
            fields.append({"line_idx": i, "raw_label": raw, "key": key})
    return fields

def ensure_value(v: Any) -> str:
    """ Si el valor está vacío, devuelve 'Sin datos'. Si es dict/list, lo serializa. """
    if v is None:
        return "Sin datos"
    if isinstance(v, (dict, list)):
        try:
            v = json.dumps(v, ensure_ascii=False)
        except Exception:
            v = str(v)
    v = str(v).strip()
    return v if v else "Sin datos"

def assemble_markdown(template_lines, fields, data):
    """
    Inserta cada valor en la MISMA línea de la etiqueta original, después de los dos puntos.
    Conserva labels con espacios (p.ej. 'Razón Social:').
    """
    out = template_lines[:]  # copiar
    for f in fields:
        line = out[f["line_idx"]]
        val = ensure_value(data.get(f["key"], ""))

        # Buscar el último ":" de la línea y conservar hasta ahí
        idx = line.rfind(":")
        if idx == -1:
            # Si por algún motivo no hay ":", no tocamos el label y agregamos al final
            base = line.rstrip()
            out[f["line_idx"]] = base + " " + val
        else:
            base = line[:idx + 1]  # incluye los dos puntos
            out[f["line_idx"]] = base + " " + val
    return "\n".join(out) + "\n"


# ==== Reglas del sistema ====
SYSTEM_RULES = """Sos un asistente que rellena un brief empresarial y responde SOLO en JSON (modo objeto).
Reglas:
- Prioridad de fuentes: (1) Usuario, (2) Sitio oficial, (3) Secundarias confiables.
- Si falta info: devolver el string EXACTO "Sin datos".
- Si hay contradicción fuerte con el usuario: usar el dato del usuario y agregar " (verificar internamente)" al final del valor.
- No inventes datos sensibles ni números sin evidencia.
- Debés devolver un JSON con EXACTAMENTE las KEYS indicadas (sin keys extra).
"""

# ==== Llamada a OpenAI (Chat Completions + JSON mode) ====
def call_model_to_get_json(fields, payload):
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
            model=OPENAI_MODEL,               # ej: gpt-4o-mini
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0
        )
        raw = resp.choices[0].message.content
        data = json.loads(raw)

        # Sanitizar: quedarnos SOLO con las keys esperadas
        clean = {}
        for k in keys_list:
            clean[k] = ensure_value(data.get(k, "Sin datos"))
        return clean

    except Exception as e:
        # Log mínimo y fallback APB
        print("ERROR OpenAI (chat.completions):", repr(e))
        return {k: "Sin datos" for k in keys_list}

# ==== Endpoints ====
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/brief/fill", response_model=FillResponse)
def fill_brief(req: FillRequest, authorization: str = Header(None)):
    # Auth simple tipo Bearer
    expected = f"Bearer {SERVICE_TOKEN}" if SERVICE_TOKEN else None
    if expected and authorization != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")

    template_text, template_lines = read_template()
    fields = extract_fields(template_lines)
    if not fields:
        raise HTTPException(status_code=400, detail="No se detectaron campos 'Etiqueta:' que terminen con ':' en la plantilla")

    data = call_model_to_get_json(fields, req.user_data)

    # Ensamble espejo 1:1
    md = assemble_markdown(template_lines, fields, data)
    return FillResponse(markdown=md)

@app.get("/brief/keys")
def brief_keys():
    _, lines = read_template()
    fields = extract_fields(lines)
    return {"keys": [f["key"] for f in fields]}