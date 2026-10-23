"""Çok-modelli motor, hasta oturumları ve arka plan analiz işleri.

Tüm arayüzler aynı motoru kullanır (``pipeline.SegmentationEngine``). Oturum kayıtları bellekte
tutulur (en çok ``settings.MAX_SESSIONS``); hasta verisi diske yazılmaz.
"""
from __future__ import annotations

import functools
import gc
import logging
import shutil
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from fastapi import HTTPException

from .. import inference_core as core
from .. import pipeline
from .. import settings

logger = logging.getLogger("neurovista")

# Model seti v2 checkpoint'leri ve ensemble kilidi arşivdeyse üç model + kilitli ensemble, değilse v1.
ENGINE = pipeline.SegmentationEngine()

records: dict[str, dict] = {}
jobs: "OrderedDict[str, dict]" = OrderedDict()
# Aynı anda tek çıkarım: GPU/CPU belleğini ve modellerin cihaz taşımasını (fallback) korur.
inference_lock = threading.Lock()


def get_record(patient_id: str, message: str = "Hasta oturumu bulunamadı; analizi yeniden çalıştırın") -> dict:
    record = records.get(patient_id)
    if record is None:
        raise HTTPException(404, detail=message)
    return record


@functools.lru_cache(maxsize=None)
def checkpoint_fingerprint(path: Path) -> dict[str, str]:
    """Raporda ve sağlık uç noktasında gösterilen model izlenebilirlik kaydı."""
    if not path.exists():
        return {"file": path.name, "sha256": "dosya yok"}
    return {"file": path.name, "sha256": core.sha256_file(path)}


def model_fingerprints() -> dict[str, dict[str, str]]:
    return {key: checkpoint_fingerprint(ENGINE.checkpoint_path(key)) for key in ENGINE.keys}


def run_inference(paths: dict[str, Path], patient_name: str | None = None, progress=None) -> dict:
    """Vakayı ortak motorla segmente eder; oturum kaydı ve API yanıtı üretir."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        pipeline.validate_case_files(paths)
        case = core.load_case(paths)
        result = ENGINE.segment(case, progress)
    except core.CaseValidationError as error:
        raise HTTPException(422, detail=str(error)) from error
    spacing, gt = case.spacing, case.gt
    model_set = ENGINE.model_set

    # Bellek koruması: eski hasta oturumlarını serbest bırak
    while len(records) >= settings.MAX_SESSIONS:
        del records[next(iter(records))]
    gc.collect()

    masks = result.masks
    patient_id = uuid.uuid4().hex
    records[patient_id] = {
        **masks, **{f"{key}_raw": result.segmentations[key].raw_prediction for key in result.keys},
        "raw": case.raw, "gt": gt, "brain_support": result.brain_support, "spacing": spacing,
        "affine": case.affine, "patient_name": patient_name, "consensus": result.consensus,
        "n_models": len(result.keys), "activation": result.activation, "primary": result.primary,
        "mask_sources": list(masks),
        "runtime": {"device": ENGINE.device.type, "overlap": ENGINE.overlap, "runtimes": result.runtimes,
                    "model_set": model_set.version, "interactive_tta": pipeline.INTERACTIVE_TTA,
                    "orientation": result.orientation, "warnings": result.warnings,
                    "created": time.strftime("%Y-%m-%d %H:%M")},
        "postprocessing": {"type": "all_modalities_nonzero_support",
                           **{f"{key}_removed_voxels": result.segmentations[key].removed_voxels for key in result.keys}},
    }

    def summary(prediction: np.ndarray) -> dict:
        return {"dice": core.region_dice(gt, prediction), "volumes_cm3": core.volumes_cm3(prediction, spacing)}

    models = {
        key: {"name": result.names[key], **summary(result.segmentations[key].prediction),
              "inference_time_sec": round(result.seconds[key], 2), "runtime": result.runtimes[key],
              "removed_outside_brain_voxels": result.segmentations[key].removed_voxels}
        for key in result.keys
    }
    ensemble = None
    if result.ensemble is not None:
        config = result.ensemble_config
        ensemble = {
            "name": "Ensemble", **summary(result.ensemble.prediction),
            "method": config["ensemble_method"], "method_name": pipeline.ENSEMBLE_METHOD_NAMES[config["ensemble_method"]],
            "weights": config["ensemble_weights"], "et_postprocessing": config["et_postprocessing"],
            "locked_tta": config["tta"], "web_tta": pipeline.INTERACTIVE_TTA,
            "inference_time_sec": round(result.seconds["ensemble"], 2),
            "removed_outside_brain_voxels": result.ensemble.removed_voxels,
        }
    return {
        "patient_id": patient_id, "patient_name": patient_name, "slices": int(case.raw.shape[3]), "modalities": sorted(paths),
        "model_set": model_set.version, "model_order": result.keys, "primary_source": result.primary,
        "models": models, "ensemble": ensemble,
        "agreement": {
            "score": result.agreement, "n_models": len(result.keys), "consensus_voxels": result.consensus_summary(),
            "note": "Modeller arası ortalama ikili Dice (WT/TC/ET); ground truth kullanmaz. "
                    "Kilitli testte başarısız vakaları ayırt ettiği gösterildi (ön-kayıt §6); tek vakada kesin hüküm değildir.",
        },
        "total_time_sec": round(sum(result.seconds.values()), 2),
        "device": ENGINE.device.type, "overlap": ENGINE.overlap, "orientation": result.orientation,
        "protocol_warnings": result.warnings,
    }


def register_job() -> str:
    job_id = uuid.uuid4().hex
    jobs[job_id] = {"status": "queued", "message": "Analiz sırası oluşturuldu"}
    while len(jobs) > settings.MAX_JOB_HISTORY:
        jobs.popitem(last=False)
    return job_id


def run_prediction_job(job_id: str, paths: dict[str, Path], patient_name: str | None,
                       upload_folder: Path | None = None) -> None:
    """Uzun süren çıkarımı HTTP isteğinden bağımsız ve sırayla (tek seferde bir iş) çalıştırır.

    Yüklenen dosyalar belleğe alındıktan sonra geçici klasör silinir; hasta verisi diskte kalmaz.
    """
    def update(**fields) -> None:
        # İş geçmişi sınırı (MAX_JOB_HISTORY) çalışan bir işi listeden düşürmüş olabilir.
        job = jobs.get(job_id)
        if job is not None:
            job.update(fields)

    def progress(message: str) -> None:
        update(status="running", message=message)
    try:
        if inference_lock.locked():
            update(status="queued", message="Önceki analizin bitmesi bekleniyor...")
        with inference_lock:
            progress("Hasta MR dosyaları yükleniyor ve doğrulanıyor...")
            result = run_inference(paths, patient_name, progress)
        update(status="completed", message="Analiz ve segmentasyon başarıyla tamamlandı.", result=result)
    except HTTPException as error:
        logger.warning("İş %s reddedildi: %s", job_id, error.detail)
        update(status="failed", message=str(error.detail))
    except Exception:
        # Ayrıntı (yığın, yol, CUDA iletisi) yalnız sunucu günlüğünde; kullanıcıya anlaşılır ileti ve iş kimliği.
        logger.exception("İşleme hatası (iş %s)", job_id)
        update(status="failed", message=f"Çıkarım sırasında beklenmeyen bir hata oluştu (iş {job_id[:8]}). "
                                        "Ayrıntı sunucu günlüğündedir.")
    finally:
        if upload_folder is not None:
            shutil.rmtree(upload_folder, ignore_errors=True)


def start_job(paths: dict[str, Path], patient_name: str | None, upload_folder: Path | None = None) -> dict:
    job_id = register_job()
    threading.Thread(target=run_prediction_job, args=(job_id, paths, patient_name, upload_folder), daemon=True).start()
    return {"job_id": job_id, "status": "queued"}
