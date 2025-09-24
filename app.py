# app.py
import os
import re
import json
import pathlib
from typing import Any, Dict, List

from dotenv import load_dotenv
from slugify import slugify
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel
from openai import OpenAI

# ====== Carga de entorno ======
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
SERVICE_TOKEN = os.getenv("SERVICE_TOKEN", "")  # <-- setear en Render/Local (.env)

if not OPENAI_API_KEY:
    raise RuntimeError("Falta OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# ====== Paths ======
ROOT = pathlib.Path(__file__).resolve().parent
TPL_PATH = ROOT / "Plantilla_MD.md"

# ====== App ======
app = FastAPI(title="AV Brief Filler", version="1.0.4")

# ====== Regex helpers ======
LABEL_RE = re.compile(
    r"""^(\s*(?:[-*+]\s+|\d+\.\s+)?)      # bullet opcional
        (?:\*\*)?                         # ** opcional
        ([^:*]+?)                         # etiqueta (sin los dos puntos)
        (?:\*\*)?
        \s*:\s*$                          # termina en :
    """,
    re.X,
)

# ====== Modelos ======
class FillResponse(BaseModel):
    markdown: str

# ====== Utilidades de plantilla ======
def read_template():
    if not TPL_PATH.exists():
        raise RuntimeError("No se encontró Plantilla_MD.md en la raíz del proyecto.")
    text = TPL_PATH.read_text(encoding="utf-8")
    return text, text.splitlines()

def extract_fields(lines: List[str]):
    """
    Devuelve lista de dicts:
    [{"line_idx": int, "raw_label": "Cliente/Marca", "key": "cliente_marca"}, ...]
    Detecta toda línea que termine con ":" (sea bullet o no).
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
    """Si el valor está vacío, devuelve 'Sin datos'. Si es dict/list, lo serializa."""
    if v is None:
        return "Sin datos"
    if isinstance(v, (dict, list)):
        try:
            v = json.dumps(v, ensure_ascii=False)
        except Exception:
            v = str(v)
    v = str(v).strip()
    return v if v else "Sin datos"

def assemble_markdown(template_lines: List[str], fields, data: Dict[str, Any]) -> str:
    """
    Inserta cada valor en la MISMA línea (después de los dos puntos).
    Conserva labels con espacios (p.ej. 'Razón social:').
    """
    out = template_lines[:]  # copia
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

# ====== Reglas del sistema ======
SYSTEM_RULES = """Sos un asistente que rellena un brief y responde SOLO en JSON (objeto).
Reglas:
- Prioridad: (1) Usuario, (2) Sitio oficial, (3) Secundarias confiables.
- Si falta info: devolver el string EXACTO "Sin datos".
- Si hay contradicción fuerte con el usuario: usar el dato del usuario y agregar " (verificar internamente)" al final del valor.
- No inventes datos sensibles ni números sin evidencia.
- Devolvé un JSON con EXACTAMENTE las KEYS indicadas (sin keys extra).
"""

# ====== Llamada a OpenAI (Chat Completions + JSON mode) ======
def call_model_to_get_json(fields, payload: Dict[str, Any]) -> Dict[str, str]:
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
        # Sanitizar: solo keys esperadas
        return {k: ensure_value(data.get(k, "Sin datos")) for k in keys_list}
    except Exception as e:
        # Log mínimo, sin secretos
        print("ERROR OpenAI (chat.completions):", repr(e))
        return {k: "Sin datos" for k in keys_list}

# ====== Endpoints ======
@app.get("/health")
def health():
    return {"ok": True, "version": "1.0.4"}

@app.get("/brief/keys")
def brief_keys():
    _, lines = read_template()
    fields = extract_fields(lines)
    return {"keys": [f["key"] for f in fields]}

@app.post("/brief/fill", response_model=FillResponse)
async def fill_brief(request: Request, authorization: str = Header(None)):
    # Auth simple tipo Bearer (no logueamos el token por seguridad)
    expected = f"Bearer {SERVICE_TOKEN}" if SERVICE_TOKEN else None
    if expected and authorization != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Aceptar tanto {"user_data": {...}} como body plano {...}
    try:
        payload = await request.json()
    except Exception:
        payload = {}

    if isinstance(payload, dict) and "user_data" in payload and isinstance(payload["user_data"], dict):
        user_data = payload["user_data"]
    elif isinstance(payload, dict):
        user_data = payload  # fallback: tomar todo el body como user_data
    else:
        user_data = {}

    # Debug acotado
    try:
        print("DEBUG /brief/fill keys:", list(user_data.keys())[:8])
    except Exception:
        pass

    template_text, template_lines = read_template()
    fields = extract_fields(template_lines)
    if not fields:
        raise HTTPException(status_code=400, detail="No se detectaron campos 'Etiqueta:' que terminen con ':' en la plantilla")

    data = call_model_to_get_json(fields, user_data)
    md = assemble_markdown(template_lines, fields, data)
    return FillResponse(markdown=md)