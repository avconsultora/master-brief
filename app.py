# app.py
import os
import re
import json
import pathlib
import difflib
from typing import Dict, Any, List

from dotenv import load_dotenv
from slugify import slugify
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel
from openai import OpenAI

# ========= Entorno / cliente OpenAI =========
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
SERVICE_TOKEN = os.getenv("SERVICE_TOKEN", "avbrief123")  # default para conveniencia

if not OPENAI_API_KEY:
    raise RuntimeError("Falta OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# ========= Paths / plantilla =========
ROOT = pathlib.Path(__file__).resolve().parent
TPL_PATH = ROOT / "Plantilla_MD.md"

# ========= FastAPI =========
app = FastAPI(title="AV Brief Filler", version="1.0.4")

# ========= Regex para detectar campos "Etiqueta:" =========
LABEL_RE = re.compile(
    r"""^(\s*(?:[-*+]\s+|\d+\.\s+)?)      # bullet opcional
        (?:\*\*)?                         # ** opcional
        ([^:*]+?)                         # etiqueta (sin los dos puntos)
        (?:\*\*)?
        \s*:\s*$                          # termina en :
    """,
    re.X,
)

# ========= Models (Pydantic) =========
class FillResponse(BaseModel):
    markdown: str

class KeysResponse(BaseModel):
    keys: List[str]

# ========= Utilidades =========
def read_template() -> tuple[str, List[str]]:
    if not TPL_PATH.exists():
        raise RuntimeError("No se encontró Plantilla_MD.md en la raíz del proyecto.")
    text = TPL_PATH.read_text(encoding="utf-8")
    return text, text.splitlines()

def extract_fields(lines: List[str]) -> List[Dict[str, Any]]:
    """
    Devuelve lista de dicts:
    [{"line_idx": int, "raw_label": "Cliente/Marca", "key": "cliente_marca"}, ...]
    Detecta toda línea que termine con ":" (sea bullet o no).
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

def assemble_markdown(template_lines: List[str], fields: List[Dict[str, Any]], data: Dict[str, Any]) -> str:
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

def normalize_user_data_to_keys(user_data: Dict[str, Any], expected_keys: List[str]) -> Dict[str, Any]:
    """
    Acepta alias con acentos/variantes y las mapea a las keys exactas (sin acentos) de expected_keys.
    Regla: slugify(k_in) -> match exacto; si no hay, best match por ratio >= 0.8; sino se ignora.
    Además, inicializa todas las keys esperadas en "Sin datos" para evitar faltantes.
    """
    norm_expected = {slugify(k, separator="_"): k for k in expected_keys}
    out = {k: "Sin datos" for k in expected_keys}

    for k_in, v in (user_data or {}).items():
        slug_in = slugify(str(k_in), separator="_")
        if slug_in in norm_expected:
            out[norm_expected[slug_in]] = v
            continue
        # best-effort por similitud
        candidates = list(norm_expected.keys())
        best = difflib.get_close_matches(slug_in, candidates, n=1, cutoff=0.8)
        if best:
            out[norm_expected[best[0]]] = v
        # si no hay match, se ignora (no agregamos keys desconocidas)
    return out

# ========= Reglas del sistema para el modelo =========
SYSTEM_RULES = """Sos un asistente que rellena un brief y responde SOLO en JSON (objeto).
Reglas:
- Prioridad: (1) Usuario, (2) Sitio oficial, (3) Secundarias confiables.
- Si falta info: devolver el string EXACTO "Sin datos".
- Si hay contradicción fuerte con el usuario: usar el dato del usuario y agregar " (verificar internamente)" al final del valor.
- No inventes datos sensibles ni números sin evidencia.
- Devolvé un JSON con EXACTAMENTE las KEYS indicadas (sin keys extra).
"""

def call_model_to_get_json(fields: List[Dict[str, Any]], payload: Dict[str, Any]) -> Dict[str, str]:
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
        # Sanitiza: solo keys esperadas, rellena faltantes
        return {k: ensure_value(data.get(k, "Sin datos")) for k in keys_list}
    except Exception as e:
        # Log mínimo, sin secretos
        print("ERROR OpenAI (chat.completions):", repr(e))
        return {k: "Sin datos" for k in keys_list}

# ========= Endpoints =========
@app.get("/health")
def health():
    return {"ok": True, "version": "1.0.4"}

@app.get("/brief/keys", response_model=KeysResponse)
def brief_keys():
    _, lines = read_template()
    fields = extract_fields(lines)
    return KeysResponse(keys=[f["key"] for f in fields])

@app.post("/brief/fill", response_model=FillResponse)
async def fill_brief(request: Request, authorization: str = Header(None)):
    # Auth simple tipo Bearer (no logueamos el token por seguridad)
    expected = f"Bearer {SERVICE_TOKEN}" if SERVICE_TOKEN else None
    if expected and authorization != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")

    user_data: Dict[str, Any] = {}

    # 1) Intento A: JSON body
    try:
        payload = await request.json()
        if isinstance(payload, dict):
            if "user_data" in payload and isinstance(payload["user_data"], dict):
                user_data = payload["user_data"]
            elif payload:  # body plano
                user_data = payload
    except Exception:
        pass

    # 2) Intento B: querystring (?user_data=... o keys sueltas)
    if not user_data:
        try:
            qs = dict(request.query_params)
            if "user_data" in qs:
                # puede venir como string JSON
                try:
                    ud = json.loads(qs["user_data"])
                    if isinstance(ud, dict):
                        user_data = ud
                except Exception:
                    # si vino como “key1=..&key2=..” plano, no JSON
                    user_data = {}
            else:
                # quizá mandaron todas las keys como params sueltos
                if qs:
                    user_data = qs
        except Exception:
            pass

    # 3) Intento C: form (application/x-www-form-urlencoded o multipart)
    if not user_data:
        try:
            form = await request.form()
            if "user_data" in form:
                try:
                    ud = json.loads(form["user_data"])
                    if isinstance(ud, dict):
                        user_data = ud
                except Exception:
                    user_data = {}
            else:
                if form:
                    user_data = dict(form)
        except Exception:
            pass

    # Debug acotado
    try:
        print("DEBUG /brief/fill keys (ingest):", list(user_data.keys())[:8])
    except Exception:
        pass

    # Plantilla y fields
    template_text, template_lines = read_template()
    fields = extract_fields(template_lines)
    if not fields:
        raise HTTPException(status_code=400, detail="No se detectaron campos 'Etiqueta:' que terminen con ':' en la plantilla")

    # Normalizar alias/acentos a keys esperadas
    expected_keys = [f["key"] for f in fields]
    user_data_norm = normalize_user_data_to_keys(user_data, expected_keys)

    # Llamada al modelo → JSON con exactamente esas keys (faltantes = "Sin datos")
    data = call_model_to_get_json(fields, user_data_norm)

    # Armar markdown espejo 1:1
    md = assemble_markdown(template_lines, fields, data)
    return FillResponse(markdown=md)