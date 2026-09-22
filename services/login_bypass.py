"""Optional login bypass for local demos.

When enabled, browser requests from this computer are signed in automatically as one staff
account, so the login page is skipped. It never applies when CAREAI_ENV=production, to
requests from other devices, or to requests that arrived through a proxy. Family portal
sign-in is separate and unaffected.

  CAREAI_BYPASS_LOGIN=0             require the normal login page (default: 1)
  CAREAI_BYPASS_LOGIN_EMAIL=...     account to sign in as (default: admin@care.ai)

Signing out pauses the bypass for one request, so the login page appears and another role
(nurse, GP) can be signed in manually.
"""
import os

from flask import request, session
from flask_login import current_user, login_user

LOOPBACK_ADDRESSES = {"127.0.0.1", "::1"}
PAUSE_KEY = "login_bypass_paused"


def login_bypass_enabled() -> bool:
    if os.getenv("CAREAI_ENV", "development") == "production":
        return False
    return os.getenv("CAREAI_BYPASS_LOGIN", "1") == "1"


def _is_local_request() -> bool:
    if request.headers.get("X-Forwarded-For") or request.headers.get("Forwarded"):
        return False
    return request.remote_addr in LOOPBACK_ADDRESSES


def pause_login_bypass() -> None:
    """Call on sign-out so the next request shows the login page."""
    session[PAUSE_KEY] = True


def install_login_bypass(app, load_user_by_email, on_login=None) -> None:
    if not login_bypass_enabled():
        return
    email = os.getenv("CAREAI_BYPASS_LOGIN_EMAIL", "admin@care.ai").strip().lower()
    app.logger.warning(
        "Login bypass is ON: browser requests from this computer are signed in as %s. "
        "Set CAREAI_BYPASS_LOGIN=0 to require the login page.", email)

    @app.before_request
    def _auto_login():
        if current_user.is_authenticated or request.endpoint == "static":
            return None
        if request.endpoint == "login" and request.method == "POST":
            return None  # let a manual sign-in (e.g. as nurse or GP) go through
        if session.pop(PAUSE_KEY, False):
            return None  # just signed out: show the login page once
        if not _is_local_request():
            return None
        user = load_user_by_email(email)
        if user is None or not user.is_active:
            return None
        login_user(user)
        session.permanent = True
        if on_login:
            on_login()
        return None
