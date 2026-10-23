"""Dışa aktarım uç noktaları: NIfTI maske (3D Slicer / ITK-SNAP) ve özet rapor (PNG/PDF)."""
from __future__ import annotations

import re

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from .. import inference_core as core
from ..services import rendering, session

router = APIRouter(prefix="/api/v1", tags=["dışa aktarım"])


def safe_stem(record: dict) -> str:
    """İndirme adı için vaka adı: yalnız güvenli karakterler (başlık enjeksiyonu ve yol karakterleri olmaz)."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", record.get("patient_name") or "neurovista_case")[:80]


@router.get("/export-mask")
def export_mask(patient_id: str, source: str | None = None, labels: str = "brats"):
    """Model maskesini orijinal MR geometrisinde NIfTI olarak indirir."""
    record = session.get_record(patient_id)
    source = source or record.get("primary", "swin")
    allowed = [name for name in record.get("mask_sources", ("unet", "swin")) if name != "gt"]
    if source not in allowed:
        raise HTTPException(422, detail=f"source şunlardan biri olmalı: {', '.join(allowed)}")
    if labels not in {"brats", "internal"}:
        raise HTTPException(422, detail="labels 'brats' (ET=4) veya 'internal' (ET=3) olmalı")
    payload = core.mask_to_nifti_bytes(record[source], record["affine"], brats_labels=labels == "brats")
    return Response(payload, media_type="application/gzip",
                    headers={"Content-Disposition": f'attachment; filename="{safe_stem(record)}_{source}_seg.nii.gz"',
                             "Cache-Control": "no-store"})


@router.get("/export-report")
def export_report(patient_id: str, slice_idx: int | None = None, modality: int = 3, format: str = "png"):
    record = session.get_record(patient_id, "Hasta oturumu bulunamadı")
    if format not in {"png", "pdf"}:
        raise HTTPException(422, detail="format png veya pdf olmalı")
    index = record["raw"].shape[3] // 2 if slice_idx is None else slice_idx
    if not 0 <= index < record["raw"].shape[3] or not 0 <= modality <= 3:
        raise HTTPException(422, detail="Geçersiz kesit veya modalite")
    media = "image/png" if format == "png" else "application/pdf"
    return Response(rendering.render_report(record, index, modality, format), media_type=media,
                    headers={"Content-Disposition": f'attachment; filename="{safe_stem(record)}_rapor.{format}"',
                             "Cache-Control": "no-store"})
