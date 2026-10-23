"""Yüklenen dosyaların güvenli kaydı, ZIP açma ve yüklenen dosyaların tek hastanın modalitelerine atanması."""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from fastapi import HTTPException, UploadFile

from .. import inference_core as core
from .. import pipeline
from .. import settings


async def save_upload(upload: UploadFile, folder: Path) -> Path:
    if not upload.filename:
        raise HTTPException(422, detail="Dosya adı eksik")
    safe_name = Path(upload.filename.replace("\\", "/")).name
    if safe_name in {"", ".", ".."}:
        raise HTTPException(422, detail="Geçersiz dosya adı")
    target = folder / safe_name
    written = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())
    with target.open("wb") as output:
        while chunk := await upload.read(1024 * 1024):
            written += len(chunk)
            if written > settings.MAX_UPLOAD_BYTES:
                raise HTTPException(413, detail=f"Yükleme sınırı aşıldı ({settings.MAX_UPLOAD_BYTES // 1024 ** 2} MB)")
            output.write(chunk)
    return target


def _is_os_metadata(member: str) -> bool:
    """macOS'un ZIP'e eklediği üst veri (``__MACOSX/``, ``._<ad>``) hasta dosyası değildir."""
    parts = member.replace("\\", "/").split("/")
    return "__MACOSX" in parts or parts[-1].startswith("._")


def extract_nifti_members(archive_path: Path, destination: Path) -> None:
    """ZIP'ten yalnız NIfTI üyelerini, boyut ve üye sayısı sınırlarıyla çıkarır."""
    with zipfile.ZipFile(archive_path) as archive:
        members = [m for m in archive.infolist()
                   if not m.is_dir() and m.filename.lower().endswith((".nii", ".nii.gz")) and not _is_os_metadata(m.filename)]
        if len(members) > settings.MAX_ZIP_MEMBERS:
            raise HTTPException(422, detail=f"ZIP içinde çok fazla NIfTI dosyası var ({len(members)})")
        if sum(m.file_size for m in members) > settings.MAX_ZIP_UNCOMPRESSED_BYTES:
            raise HTTPException(413, detail="ZIP açıldığında boyut sınırını aşıyor")
        for member in members:
            archive.extract(member, destination)   # zipfile mutlak yolları ve '..' bileşenlerini temizler


def collect_modalities(folder: Path) -> dict[str, Path]:
    """Yüklenen NIfTI dosyalarının tamamını tek hastanın modalitelerine atar (``pipeline.assign_case_files``).

    Yalnız yüklenen dosyalar kullanılır; eksik modalite sunucu diskinden tamamlanmaz. Başka hastanın,
    adı anlaşılamayan ya da yinelenen dosyası varsa seçim hangi dosyanın sorunlu olduğu söylenerek reddedilir.
    """
    files = [path for path in folder.rglob("*") if path.is_file() and core.is_nifti(path)]
    try:
        return pipeline.assign_case_files(files)
    except core.CaseValidationError as error:
        raise HTTPException(422, detail=str(error)) from error


def purge_stale_uploads() -> None:
    """Önceki oturumlardan kalmış geçici yükleme klasörlerini siler (başlangıçta)."""
    settings.RUNTIME_DIR.mkdir(exist_ok=True)
    for item in settings.RUNTIME_DIR.iterdir():
        if item.is_dir():
            shutil.rmtree(item, ignore_errors=True)
