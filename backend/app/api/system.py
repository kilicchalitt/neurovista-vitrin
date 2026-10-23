"""Sistem uç noktaları: sağlık/izlenebilirlik ve kilitli sonuçlar."""
from __future__ import annotations

import functools
import platform
from importlib import metadata

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from .. import pipeline
from .. import settings
from ..services import session

router = APIRouter(prefix="/api/v1", tags=["sistem"])


@functools.lru_cache(maxsize=1)
def environment() -> dict[str, str | None]:
    """Çalışma ortamı sürümleri (C15): yerel, Docker ve Kaggle sonuçlarının karşılaştırılabilmesi için."""
    import monai
    import numpy
    import torch

    def version(package: str) -> str | None:
        try:
            return metadata.version(package)
        except metadata.PackageNotFoundError:
            return None
    return {"python": platform.python_version(), "platform": platform.platform(terse=True),
            "torch": torch.__version__, "cuda": torch.version.cuda, "monai": monai.__version__,
            "numpy": numpy.__version__, "openvino": version("openvino"), "fastapi": version("fastapi")}


@router.get("/health")
def health():
    engine, model_set = session.ENGINE, session.ENGINE.model_set
    return {
        "status": "ready",
        "device": engine.device.type,
        "device_info": engine.device_info,
        "model_set": model_set.version,
        "model_set_note": model_set.note,
        "models": [{"key": key, "name": engine.name(key), "runtime": engine.planned_runtime(key),
                    "loaded": engine.is_loaded(key)} for key in engine.keys],
        "ensemble": None if model_set.ensemble is None else {
            **model_set.ensemble, "method_name": pipeline.ENSEMBLE_METHOD_NAMES[model_set.ensemble["ensemble_method"]],
            "web_tta": pipeline.INTERACTIVE_TTA},
        "openvino_state": engine.openvino_state,
        "overlap": engine.overlap,
        "checkpoints": session.model_fingerprints(),
        "environment": environment(),
        "intended_use": "Araştırma amaçlı karar destek prototipi; tanı için kullanılamaz.",
    }


@router.get("/results")
def results():
    """Kilitli sonuçlar (tek kaynak: tools/build_results.py). Sayfalar sayıları buradan okur."""
    if not settings.RESULTS_PATH.exists():
        raise HTTPException(404, detail="Sonuç dosyası yok (python tools/build_results.py)")
    return FileResponse(settings.RESULTS_PATH, media_type="application/json")
