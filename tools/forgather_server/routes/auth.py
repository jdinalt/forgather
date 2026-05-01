"""Authentication endpoints.

Three endpoints are reachable without auth: ``GET /api/auth/status`` (so
the webui can decide whether to show the login page) and ``POST
/api/auth/login`` (which is the way *to* obtain auth). ``set-password``
and ``logout`` go through the normal middleware gate like every other
``/api/`` route.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from .. import auth as auth_mod

router = APIRouter(tags=["auth"], prefix="/auth")


class LoginRequest(BaseModel):
    token: Optional[str] = None
    password: Optional[str] = None


class LoginResponse(BaseModel):
    ok: bool
    requires_password_setup: bool


class StatusResponse(BaseModel):
    authenticated: bool
    has_password: bool
    auth_disabled: bool


class SetPasswordRequest(BaseModel):
    password: str


@router.get("/status", response_model=StatusResponse)
def auth_status(request: Request):
    return StatusResponse(
        authenticated=auth_mod.authenticate(
            request.headers, request.query_params, request.cookies
        ),
        has_password=auth_mod.has_password(),
        auth_disabled=auth_mod.auth_disabled(),
    )


@router.post("/login", response_model=LoginResponse)
def auth_login(body: LoginRequest, request: Request, response: Response):
    if auth_mod.auth_disabled():
        sid = auth_mod.create_session()
        _set_session_cookie(request, response, sid)
        return LoginResponse(ok=True, requires_password_setup=False)

    used_token = False
    ok = False
    if body.token:
        used_token = True
        ok = auth_mod.verify_token(body.token)
    elif body.password:
        ok = auth_mod.verify_password(body.password)

    if not ok:
        # 401 with a generic message — don't disclose which channel was tried.
        raise HTTPException(status_code=401, detail="invalid credentials")

    sid = auth_mod.create_session()
    _set_session_cookie(request, response, sid)
    return LoginResponse(
        ok=True,
        requires_password_setup=used_token and not auth_mod.has_password(),
    )


@router.post("/set-password")
def auth_set_password(body: SetPasswordRequest):
    # Reaching this handler means the auth middleware already accepted the
    # request, so we don't re-check credentials here.
    pw = (body.password or "").strip()
    if len(pw) < 4:
        raise HTTPException(
            status_code=400, detail="password must be at least 4 characters"
        )
    auth_mod.set_password(pw)
    return {"ok": True}


@router.post("/logout")
def auth_logout(request: Request, response: Response):
    sid = request.cookies.get(auth_mod.SESSION_COOKIE_NAME)
    auth_mod.revoke_session(sid)
    response.delete_cookie(auth_mod.SESSION_COOKIE_NAME, path="/")
    return {"ok": True}


def _set_session_cookie(request: Request, response: Response, sid: str) -> None:
    secure = request.url.scheme == "https"
    response.set_cookie(
        auth_mod.SESSION_COOKIE_NAME,
        sid,
        max_age=auth_mod.SESSION_TTL_SECONDS,
        httponly=True,
        samesite="lax",
        secure=secure,
        path="/",
    )
