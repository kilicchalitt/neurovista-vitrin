"""İsteğe bağlı erişim belirteci (C10): NEUROVISTA_ACCESS_TOKEN ayarlıysa servis yalnız belirteçle açılır.

Akış: kullanıcı bir kez ``/?token=<belirteç>`` ile gelir; belirteç HttpOnly çereze yazılır ve adres
çubuğundan silinmek üzere aynı sayfaya yönlendirilir. Sonraki istekler (sayfalar, görüntüler, API)
çerezle geçer; programlar ``Authorization: Bearer <belirteç>`` başlığını kullanabilir.
``/api/v1/health`` konteyner sağlık denetimi için açık kalır. Karşılaştırma sabit zamanlıdır.
"""
from __future__ import annotations

import hmac

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse

from .. import settings
from .errors import error_body

OPEN_PATHS = {"/api/v1/health"}


def _matches(candidate: str | None, token: str) -> bool:
    return bool(candidate) and hmac.compare_digest(candidate.encode(), token.encode())


def register(app: FastAPI) -> None:
    @app.middleware("http")
    async def access_token_guard(request: Request, call_next):
        token = settings.ACCESS_TOKEN
        if not token or request.url.path in OPEN_PATHS:
            return await call_next(request)
        query_token = request.query_params.get("token")
        if query_token is not None and _matches(query_token, token):
            response = RedirectResponse(str(request.url.remove_query_params("token")), status_code=303)
            response.set_cookie(settings.ACCESS_COOKIE, token, httponly=True, samesite="strict")
            return response
        bearer = request.headers.get("authorization", "")
        supplied = request.cookies.get(settings.ACCESS_COOKIE) or (bearer[7:] if bearer.lower().startswith("bearer ") else None)
        if _matches(supplied, token):
            return await call_next(request)
        return JSONResponse(status_code=401, content=error_body(401, None, request.url.path))
