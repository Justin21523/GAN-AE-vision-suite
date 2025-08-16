# src/api/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, Body, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import io, os, logging

from src.service.gan_infer import GANService, GenerateParams


log = logging.getLogger(__name__)


# ---------- Lifespan handler ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    log.info(
        "Registered routes (pre-mount): %s",
        [getattr(r, "path", str(r)) for r in app.router.routes],
    )
    yield
    # shutdown（若需要釋放資源可放這裡）
    # e.g., close file handles, release GPU memory, etc.


app = FastAPI(title="GAN API")

# CORS（dev 時讓 React :5173 能請求）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

svc = GANService()  # one service per process

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


@app.post("/load")
def load(req: LoadReq):
    svc.__init__(device=req.device)
    svc.load_checkpoint(req.ckpt)
    return {"ok": True, "device": str(svc.device), "img_size": svc.cfg["img_size"]}  # type: ignore


# ✅ 別名：有些前端/舊代碼可能打 /api/gan/load
@api.post("/gan/load")
def load_alias(req: LoadReq):
    return load(req)


@app.post("/gan/generate")
def generate(req: GenReq):
    img = svc.generate_grid(
        GenerateParams(
            n=req.n, seed=req.seed, nrow=req.nrow, use_ema_shadow=req.use_ema
        )
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


# ✅ 診斷端點
@api.get("/health")
def health():
    return {"ok": True}


@api.get("/_routes")
def routes():
    # 列出所有註冊路由，方便定位 404
    return sorted([getattr(r, "path", str(r)) for r in app.router.routes])


app.include_router(api)

# -------- 產線：託管 React build --------
DIST_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../..", "gan-ui", "dist")
)
if os.path.isdir(DIST_DIR):
    app.mount("/", StaticFiles(directory=DIST_DIR, html=True), name="static")
else:

    @app.get("/", response_class=HTMLResponse)
    def index():
        return "<h3>GAN API is running.</h3><p>Build UI to 'gan-ui/dist' to serve frontend here.</p>"
