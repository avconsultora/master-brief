# app.py
import os, re, json, pathlib, difflib, logging
from typing import Dict, Any, List, Tuple, Optional

from slugify import slugify
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, ConfigDict

# ---------- Config / paths ----------
SERVICE_TOKEN = os.getenv("SERVICE_TOKEN", "avbrief123")
ROOT = pathlib.Path(__file__).resolve().parent
TPL_PATH = ROOT / "Plantilla_MD.md"

log = logging.getLogger("brief")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

app = FastAPI(title="AV Brief Filler", version="1.0.6")

# ---------- Regex: líneas "Etiqueta:" ----------
LABEL_RE = re.compile(
    r"""^(\s*(?:[-*+]\s+|\d+\.\s+)?)      # bullet opcional
        (?:\*\*)?                         # ** opcional
        ([^:*]+?)                         # etiqueta (sin los dos puntos)
        (?:\*\*)?
        \s*:\s*$                          # termina en :
    """,
    re.X,
)

# ---------- Modelos ----------
class FillResponse(BaseModel):
    markdown: str

class KeysResponse(BaseModel):
    keys: List[str]

# Modelo “formal” para Actions que envían {"user_data": {...}}
class FillRequest(BaseModel):
    model_config = ConfigDict(extra="allow")  # no romper si vienen extras
    user_data: Dict[str, Any]

# ---------- Utilidades ----------
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
        base = line if idx == -1 else line[: idx + 1]
        if not base.endswith(":"):
            base = base.rstrip()
        out[f["line_idx"]] = f"{base} {val}"
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
    # 1) JSON: {"user_data": {...}} o body plano {...}
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
    qs = dict(request.query_params)
    if "user_data" in qs:
        try:
            ud = json.loads(qs["user_data"])
            if isinstance(ud, dict):
                return ud
        except Exception:
            return {}
    if qs:
        return qs
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
        if form:
            return dict(form)
    except Exception:
        pass
    return {}

# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {"ok": True, "version": "1.0.6"}

@app.get("/brief/keys", response_model=KeysResponse)
def brief_keys():
    _, lines = read_template()
    fields = extract_fields(lines)
    return KeysResponse(keys=[f["key"] for f in fields])

@app.post("/brief/fill", response_model=FillResponse)
async def fill_brief(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    # Bearer simple
    expected = f"Bearer {SERVICE_TOKEN}" if SERVICE_TOKEN else None
    if expected and authorization != expected:
        log.warning("401 Unauthorized: header incorrecto/ausente")
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Ingesta flexible
    user_data = await ingest_user_data(request)
    log.info("Ingestadas keys: %s", list(user_data.keys())[:8])

    # Plantilla → keys oficiales
    template_text, template_lines = read_template()
    fields = extract_fields(template_lines)
    if not fields:
        raise HTTPException(status_code=400, detail="No se detectaron campos 'Etiqueta:' con ':' final en la plantilla")

    expected_keys = [f["key"] for f in fields]
    normalized = normalize_user_data_to_keys(user_data, expected_keys)
    data = {k: ensure_value(normalized.get(k, "Sin datos")) for k in expected_keys}

    md = assemble_markdown(template_lines, fields, data)
    return FillResponse(markdown=md)