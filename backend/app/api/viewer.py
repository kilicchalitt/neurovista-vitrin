"""Görüntüleyici uç noktaları: kesit panelleri, FLAIR hacmi, OBJ yüzeyi ve 3B sahne."""
from __future__ import annotations

import logging

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from ..services import mesh, rendering, session

router = APIRouter(prefix="/api/v1", tags=["görüntüleyici"])
logger = logging.getLogger("neurovista")
MESH_LAYERS = ("ncr", "ed", "et", "wt")


@router.get("/slice")
def get_slice(patient_id: str, source: str, slice_idx: int, modality: int = 3):
    record = session.get_record(patient_id)
    if source not in {"original", "activation", "consensus", *record.get("mask_sources", ())}:
        raise HTTPException(422, detail="Geçersiz panel kaynağı")
    if not 0 <= modality <= 3 or not 0 <= slice_idx < record["raw"].shape[3]:
        raise HTTPException(422, detail="Geçersiz kesit veya modalite")
    return Response(rendering.make_panel(record, source, slice_idx, modality), media_type="image/png",
                    headers={"Cache-Control": "private, max-age=3600"})


@router.get("/volume")
def get_flair_volume(patient_id: str):
    """WebGL hacim görüntüsü için gerçek FLAIR verisinin kompakt (≤128 voksel/eksen) kopyası.

    Çıkarım ve metrikler tam çözünürlüklü veride kalır; bu yalnız görsel kopyadır.
    """
    record = session.get_record(patient_id)
    flair = np.asarray(record["raw"][3], dtype=np.float32)
    values = flair[flair > 0]
    if values.size == 0:
        raise HTTPException(422, detail="FLAIR hacminde görüntülenecek veri yok")
    low, high = np.percentile(values, [1, 99])
    normalized = np.clip((flair - low) / (high - low + 1e-8), 0.0, 1.0)
    normalized[flair <= 0] = 0.0
    stride = max(1, int(np.ceil(max(normalized.shape) / 128)))
    volume = np.ascontiguousarray((normalized[::stride, ::stride, ::stride] * 255).round().astype(np.uint8))
    spacing = np.asarray(record["spacing"], dtype=np.float32) * stride
    headers = {
        "X-Volume-Shape": ",".join(str(int(value)) for value in volume.shape),
        "X-Volume-Spacing": ",".join(f"{float(value):.6f}" for value in spacing),
        "Cache-Control": "no-store",
    }
    return Response(volume.tobytes(), media_type="application/octet-stream", headers=headers)


def _mask_source(record: dict, source: str | None, message: str) -> tuple[str, np.ndarray]:
    source = source or record["primary"]
    mask = record.get(source)
    if source not in record["mask_sources"] or mask is None:
        raise HTTPException(422, detail=message)
    return source, mask


@router.get("/mesh")
def get_mesh(patient_id: str, source: str | None = None, layer: str = "wt"):
    record = session.get_record(patient_id, "Hasta oturumu bulunamadı")
    if layer not in MESH_LAYERS:
        raise HTTPException(422, detail=f"layer şunlardan biri olmalı: {', '.join(MESH_LAYERS)}")
    source, mask = _mask_source(record, source, "Bu kaynak için mesh oluşturulamaz")
    binary = {"ncr": mask == 1, "ed": mask == 2, "et": mask == 3, "wt": mask > 0}[layer]
    if binary.sum() < mesh.MIN_VOXELS:
        raise HTTPException(404, detail="Seçili katmanda mesh oluşturacak tümör bölgesi yok")
    return Response(mesh.obj_payload(binary, record["spacing"]), media_type="model/obj",
                    headers={"Content-Disposition": f'attachment; filename="{source}_{layer}.obj"'})


@router.get("/scene")
def get_scene(patient_id: str, source: str | None = None):
    record = session.get_record(patient_id, "Hasta oturumu bulunamadı")
    source, _ = _mask_source(record, source, "Bu kaynak için 3B sahne oluşturulamaz")
    scene_cache = record.setdefault("scene_cache", {})
    if source not in scene_cache:
        try:
            scene_cache[source] = mesh.scene_payload(record, source)
        except Exception as error:
            logger.exception("3B sahne hazırlanamadı (%s)", source)
            raise HTTPException(500, detail="3B sahne hazırlanamadı. Ayrıntı sunucu günlüğündedir.") from error
    return scene_cache[source]
