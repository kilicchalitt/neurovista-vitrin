"""Sayfalar: tanıtım (/) ve analiz konsolu (/analysis); diğer statik dosyalar (css/, js/) main.py'de bağlanır.

Sayfalar ve statik dosyalar ``Cache-Control: no-cache`` ile sunulur: tarayıcı kopyasını kullanmadan önce ETag ile
sunucuya sorar (değişmediyse 304, gövde gönderilmez). Önbellek başlığı olmadığında tarayıcı sezgisel önbellekle
eski HTML'i yeni betikle birlikte kullanabiliyordu; güncellemeden sonra sayfa kırılıyordu.
"""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .. import settings

router = APIRouter(include_in_schema=False)
NO_CACHE = {"Cache-Control": "no-cache"}


@router.api_route("/", methods=["GET", "HEAD"])
def landing_page():
    return FileResponse(settings.STATIC_DIR / "landing.html", headers=NO_CACHE)


@router.api_route("/analysis", methods=["GET", "HEAD"])
def analysis_page():
    return FileResponse(settings.STATIC_DIR / "index.html", headers=NO_CACHE)


class RevalidatedStaticFiles(StaticFiles):
    """css/, js/ ve diğer statik dosyalar her açılışta ETag ile yeniden doğrulanır."""

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers.setdefault("Cache-Control", "no-cache")
        return response
