"""NeuroVista web servisi: çok-modelli BraTS çıkarımını tarayıcıdan sunan yerel FastAPI uygulaması.

Katmanlar (C3):
- ``settings``: yollar, sınırlar, ortam değişkenleriyle açılan seçenekler.
- ``services/``: motor ve oturumlar (``session``), yüklemeler, kesit/rapor çizimi, 3B yüzeyler.
- ``api/``: uç noktalar (sistem, analiz, görüntüleyici, dışa aktarım, sayfalar) ve hata yanıtları.
Bilimsel protokol (ön işleme, etiket kararı, ensemble) yalnız ``inference_core``'dadır.

Başlatma: ``python run.py`` ya da ``uvicorn backend.app.main:app``.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from . import settings
from .api import access, analysis, errors, export, pages, system, viewer
from .services import session, uploads

logger = logging.getLogger("neurovista")
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Başlangıç: eski geçici yüklemeleri sil, CPU'da OpenVINO'yu arka planda hazırla."""
    uploads.purge_stale_uploads()
    logger.info("Model seti: %s · cihaz %s", session.ENGINE.model_set.note, session.ENGINE.device.type)
    session.ENGINE.warm_openvino_async()
    yield


app = FastAPI(title="NeuroVista BraTS Inference API", version="2.0.0", lifespan=lifespan)
# Sayfalar API ile aynı kökenden sunulur; başka kökene (CORS) izin verilmez.
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",        # yüklenen/indirilen içerik türü tahmin edilmez
    "X-Frame-Options": "DENY",                  # sayfalar başka sitede çerçevelenemez (tıklama kaçırma)
    "Referrer-Policy": "no-referrer",           # oturum/hasta kimliği içeren adresler dışarı sızmaz
}


@app.middleware("http")
async def security_headers(request, call_next):
    response = await call_next(request)
    for name, value in SECURITY_HEADERS.items():
        response.headers.setdefault(name, value)
    return response


for module in (system, analysis, viewer, export, pages):
    app.include_router(module.router)
errors.register(app)
access.register(app)   # NEUROVISTA_ACCESS_TOKEN ayarlı değilse etkisiz
app.mount("/", pages.RevalidatedStaticFiles(directory=settings.STATIC_DIR, html=True), name="web")

# Geriye uyumlu adlar (backend/scripts/verify_gpu.py ve eski betikler)
settings.RUNTIME_DIR.mkdir(exist_ok=True)
ENGINE = session.ENGINE
MODEL_SET, MODEL_KEYS = ENGINE.model_set, ENGINE.keys
DEVICE, DEVICE_INFO, OVERLAP = ENGINE.device, ENGINE.device_info, ENGINE.overlap
RUNTIME_DIR, RESULTS_PATH = settings.RUNTIME_DIR, settings.RESULTS_PATH
run_inference = session.run_inference
