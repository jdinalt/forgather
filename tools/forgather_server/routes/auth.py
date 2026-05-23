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
    current_password: Optional[str] = None


@router.get("/status", response_model=StatusResponse)
def auth_status(request: Request):
    return StatusResponse(
        authenticated=bool(
            auth_mod.authenticate(
                request.headers, request.query_params, request.cookies
            )
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
        # Persistent bearer first; fall back to the short-lived
        # single-use URL token minted by ``/api/cluster/issue_url_token``
        # for cross-node SSO. ``verify_url_token`` consumes on
        # success, so a leaked URL is only valid once.
        ok = auth_mod.verify_token(body.token) or auth_mod.verify_url_token(
            body.token
        )
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
def auth_set_password(body: SetPasswordRequest, request: Request):
    # When a password already exists, require either the current password
    # or a fresh bearer/query token. A cookie-only session is not enough:
    # otherwise a hijacked session could silently rotate the password.
    if auth_mod.has_password():
        kind = auth_mod.credential_kind(
            request.headers, request.query_params, request.cookies
        )
        if kind in ("token", "query_token", "disabled"):
            pass
        elif body.current_password and auth_mod.verify_password(body.current_password):
            pass
        else:
            raise HTTPException(status_code=401, detail="current password is incorrect")
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
