"""Hata yanıtları: API için açıklayıcı JSON, tarayıcı için 404 sayfası."""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .. import settings

logger = logging.getLogger("neurovista")

STATUS_GUIDES = {
    400: ("Geçersiz İstek", "Gönderilen istek formatı veya parametreleri geçersiz.", "Lütfen seçtiğiniz dosyaları ve parametreleri kontrol edip tekrar deneyiniz."),
    401: ("Erişim Belirteci Gerekli", "Bu sunucu erişim belirteciyle korunuyor.", "Sunucu yöneticisinden aldığınız bağlantıyı (…/?token=…) kullanınız."),
    404: ("Kaynak Bulunamadı", "Ulaşmaya çalıştığınız analiz oturumu veya sayfa mevcut değil.", "Lütfen analiz konsolundan yeni bir oturum başlatınız."),
    413: ("Dosya Boyutu Çok Büyük", "Yüklenen MR arşivi sunucu sınırını aştı.", "Lütfen sıkıştırılmış NIfTI (.nii.gz) dosyalarını tercih ediniz."),
    415: ("Desteklenmeyen Dosya Türü", "Yalnızca tıbbi NIfTI (.nii, .nii.gz) veya bunları içeren .zip dosyaları kabul edilir.", "JPG, PNG veya PDF yerine lütfen doğrudan MR NIfTI dosyalarını seçiniz."),
    422: ("Tıbbi Veri Uyumsuzluğu", "MR dosyaları boyut veya modalite açısından doğrulanamadı.", "Aynı hastaya ait 4 modalitenin (T1, T1ce, T2, FLAIR) eksiksiz seçildiğinden ve boyutlarının eşit olduğundan emin olunuz."),
    500: ("Sunucu Analiz Hatası", "Model çıkarımı veya 3B mesh inşası sırasında bir sistem istisnası oluştu.", "Lütfen sunucu günlüğünü kontrol ediniz veya analizi tekrar başlatınız."),
    503: ("Yapay Zeka Servisi Meşgul", "Sinir ağı modelleri henüz başlatılıyor veya bellek dolu.", "Lütfen birkaç saniye bekleyip tekrar deneyiniz."),
}


def error_body(status: int, detail: str | None, path: str) -> dict:
    title, description, guidance = STATUS_GUIDES.get(status, ("Sistem Uyarısı", detail or "", "Lütfen işlemi tekrar deneyiniz."))
    return {"status": "error", "code": status, "title": title, "detail": detail or description,
            "guidance": guidance, "path": path}


def register(app: FastAPI) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        is_html = "text/html" in request.headers.get("accept", "") and not request.url.path.startswith("/api/")
        if exc.status_code == 404 and is_html and (settings.STATIC_DIR / "404.html").exists():
            return FileResponse(settings.STATIC_DIR / "404.html", status_code=404)
        detail = exc.detail if isinstance(exc.detail, str) else None
        return JSONResponse(status_code=exc.status_code, content=error_body(exc.status_code, detail, request.url.path))

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        return JSONResponse(status_code=422, content={
            "status": "error", "code": 422, "title": "Geçersiz Veri Formatı",
            "detail": "Gönderilen form verisi veya dosya yükleme parametreleri doğrulanamadı.",
            "guidance": "Lütfen geçerli NIfTI (.nii.gz) dosyaları seçtiğinizden emin olunuz.",
            "errors": [str(e.get("msg", e)) for e in exc.errors()],
        })

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception):
        # İç hata metni (dosya yolu, yığın) istemciye gönderilmez; ayrıntı yalnız sunucu günlüğündedir.
        logger.error("Beklenmeyen sistem hatası (%s): %s", request.url.path, exc, exc_info=True)
        return JSONResponse(status_code=500, content={
            "status": "error", "code": 500, "title": "Kritik Sistem İstisnası",
            "detail": "Analiz sunucusunda beklenmeyen bir hata oluştu. Ayrıntı sunucu günlüğündedir.",
            "guidance": "Lütfen seçilen MR dosyalarının geçerli NIfTI formatında olduğunu kontrol edip tekrar deneyiniz.",
            "path": request.url.path,
        })
