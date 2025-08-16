# src/api/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, Body, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import io, os, logging

from src.service.gan_infer import GANService, GenerateParams


log = logging.getLogger(__name__)


# ---------- Lifespan handler ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    log.info("Starting GAN API...")
    yield
    # shutdown
    log.info("Shutting down GAN API.")


app = FastAPI(title="GAN API+UI", lifespan=lifespan)

# CORS（dev 時讓 React :5173 能請求）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

svc = GANService()  # one service per process
loaded_ckpt: str | None = None
# -------- API 路由（帶 /api 前綴） --------
api = APIRouter(prefix="/api")


class LoadReq(BaseModel):
    ckpt: str
    device: str | None = None


class GenReq(BaseModel):
    n: int = 64
    seed: int = 42
    nrow: int = 8
    use_ema: bool = False


@api.post("/load")
def load(req: LoadReq):
    global loaded_ckpt
    if not req.ckpt or not os.path.exists(req.ckpt):
        return JSONResponse(
            {"ok": False, "error": f"ckpt not found: {req.ckpt}"}, status_code=400
        )
    svc.__init__(device=req.device)
    svc.load_checkpoint(req.ckpt)
    loaded_ckpt = req.ckpt
    return {
        "ok": True,
        "device": str(svc.device),
        "img_size": svc.cfg["img_size"],  # type: ignore
        "ckpt": loaded_ckpt,
    }


# alias
@api.post("/gan/load")
def load_alias(req: LoadReq):
    return load(req)


def _do_generate(req: GenReq) -> Response:
    if svc.G is None:
        return JSONResponse(
            {"ok": False, "error": "no model loaded; call /api/load first"},
            status_code=400,
        )
    img = svc.generate_grid(
        GenerateParams(
            n=req.n, seed=req.seed, nrow=req.nrow, use_ema_shadow=req.use_ema
        )
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@api.post("/generate")
def generate_plain(req: GenReq):
    return _do_generate(req)


@api.post("/gan/generate")
def generate_alias(req: GenReq):
    return _do_generate(req)


@api.get("/info")
def info():
    return {
        "ok": True,
        "device": str(svc.device),
        "loaded_ckpt": loaded_ckpt,
        "has_model": svc.G is not None,
    }


@api.get("/health")
def health():
    return {"ok": True}


@api.get("/_routes")
def routes():
    return sorted([getattr(r, "path", str(r)) for r in app.router.routes])


app.include_router(api)

# ---- serve React build if present ----
DIST_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../..", "gan-ui", "dist")
)
if os.path.isdir(DIST_DIR):
    app.mount("/", StaticFiles(directory=DIST_DIR, html=True), name="static")
else:

    @app.get("/", response_class=HTMLResponse)
    def index():
        return "<h3>GAN API is running.</h3><p>Build UI to 'gan-ui/dist' to serve frontend here.</p>"
