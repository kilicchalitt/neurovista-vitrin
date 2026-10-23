"""Analiz uç noktaları: dosya yükleme ve arka plan iş durumu."""
from __future__ import annotations

import logging
import shutil
import uuid
import zipfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from .. import inference_core as core
from .. import settings
from ..services import session, uploads

router = APIRouter(prefix="/api/v1", tags=["analiz"])
logger = logging.getLogger("neurovista")


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = session.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, detail="Analiz işi bulunamadı")
    return job


def _resolve_case(folder: Path, saved: list[Path]) -> tuple[dict[str, Path], str]:
    """Yüklenen dosyaları (ZIP'ler açılarak) tek hastanın modalitelerine atar; yalnız yüklenenler kullanılır.

    Disk işi (ZIP açma, klasör tarama) yaptığı için olay döngüsünü bloklamamak üzere iş parçacığında çalışır.
    """
    for path in saved:
        if path.suffix.lower() == ".zip":
            try:
                uploads.extract_nifti_members(path, folder / "archive")
            except zipfile.BadZipFile as error:
                raise HTTPException(422, detail="ZIP dosyası okunamadı") from error
            path.unlink()
        elif not core.is_nifti(path):
            raise HTTPException(422, detail="Yalnızca .nii, .nii.gz veya bunları içeren .zip kabul edilir")
    paths = uploads.collect_modalities(folder)
    return paths, core.case_display_name(paths["flair"])


@router.post("/predict")
async def predict(files: Annotated[list[UploadFile], File(description="Dört MR dosyası veya bunları içeren tek ZIP")]):
    """Tek seçiciden gelen dosyaları isimlerinden eşleştirip etkin model setini çalıştırır."""
    folder = settings.RUNTIME_DIR / uuid.uuid4().hex
    folder.mkdir(parents=True, exist_ok=False)
    try:
        saved = [await uploads.save_upload(upload, folder) for upload in files]
        paths, patient_name = await run_in_threadpool(_resolve_case, folder, saved)
        return session.start_job(paths, patient_name, folder)
    except HTTPException:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    except Exception as error:
        shutil.rmtree(folder, ignore_errors=True)
        # Ayrıntı sunucu günlüğünde; istemciye iç hata metni (yol, yığın) gönderilmez.
        logger.exception("Yükleme hazırlanamadı")
        raise HTTPException(500, detail="Yüklenen dosyalar hazırlanamadı. Ayrıntı sunucu günlüğündedir.") from error
