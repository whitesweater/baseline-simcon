#!/usr/bin/env python3
"""Helpers for validating and refreshing HPC4 AIStudio auth tokens."""

from __future__ import annotations

import base64
import json
import os
import random
import re
import shlex
import string
from dataclasses import dataclass
from http.cookiejar import CookieJar
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError
from urllib.request import HTTPCookieProcessor, Request, build_opener, urlopen


DEFAULT_BASE_URL = "https://hpc4login.hpc.hkust-gz.edu.cn"
USER_API_PREFIX = "/hpc-user/api"
IAM_API_PREFIX = "/iam/api/v1"
TOKEN_ENV_KEY = "HPC4_AISTUDIO_TOKEN"
TICKET_ENV_KEY = "HPC4_AISTUDIO_TICKET"
END_ORG_ENV_KEY = "HPC4_AISTUDIO_END_ORG"
LOGIN_USER_ENV_KEYS = ("hpc4user", "HPC4USER")
LOGIN_PASSWORD_ENV_KEYS = ("hpc4password", "HPC4PASSWORD")
SHELL_FALLBACK_FILES = (Path.home() / ".profile", Path.home() / ".bashrc")
LOGIN_APP_URL = f"{DEFAULT_BASE_URL}/#/app/user"
THIRD_LOGIN_CLIENT_ID = "a521840c6e829f70e3558237e97980ec"
THIRD_LOGIN_STATE = "xyz"
THIRD_LOGIN_SCOPE = "all"
THIRD_LOGIN_REFERER = f"{DEFAULT_BASE_URL}/hpc-user/#/third_login"


class ApiError(RuntimeError):
    """Wrap a non-success HTTP response."""

    def __init__(self, url: str, status: int, detail: str) -> None:
        super().__init__(f"HTTP {status} for {url}: {detail}")
        self.url = url
        self.status = status
        self.detail = detail


@dataclass
class AuthResult:
    base_url: str
    token: str
    ticket: str | None
    end_org: str | None
    source: str


def decode_jwt_payload(token: str) -> dict[str, Any]:
    parts = token.split(".")
    if len(parts) < 2:
        return {}
    payload = parts[1]
    payload += "=" * (-len(payload) % 4)
    try:
        decoded = base64.urlsafe_b64decode(payload.encode("ascii"))
        loaded = json.loads(decoded.decode("utf-8"))
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {}


def env_first(keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = os.environ.get(key)
        if value:
            return value
    for key in keys:
        value = shell_fallback_value(key)
        if value:
            return value
    return None


def shell_fallback_value(key: str) -> str | None:
    pattern = re.compile(rf"^\s*(?:export\s+)?{re.escape(key)}=(?P<value>.+?)\s*$")
    for path in SHELL_FALLBACK_FILES:
        if not path.exists():
            continue
        try:
            lines = path.read_text().splitlines()
        except OSError:
            continue
        for line in reversed(lines):
            match = pattern.match(line)
            if not match:
                continue
            raw = match.group("value").strip()
            if raw.startswith(("'", '"')) and raw.endswith(("'", '"')) and len(raw) >= 2:
                raw = raw[1:-1]
            return raw
    return None


def json_request(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    payload: dict[str, Any] | None = None,
    timeout: int = 30,
) -> tuple[int, Any]:
    request_headers = {"Accept": "application/json"}
    if headers:
        request_headers.update(headers)

    body = None
    if payload is not None:
        request_headers.setdefault("Content-Type", "application/json")
        body = json.dumps(payload).encode("utf-8")

    request = Request(url, headers=request_headers, data=body, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            status = response.status
            text = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ApiError(url, exc.code, detail) from exc
    except URLError as exc:
        raise RuntimeError(f"Request failed for {url}: {exc}") from exc

    try:
        return status, json.loads(text)
    except json.JSONDecodeError:
        return status, text


def auth_headers(token: str, end_org: str | None = None) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {token}"}
    if end_org:
        headers["end-org"] = end_org
    return headers


def validate_token(base_url: str, token: str, end_org: str | None = None) -> bool:
    url = f"{base_url.rstrip('/')}{IAM_API_PREFIX}/users/current"
    try:
        status, payload = json_request(url, headers=auth_headers(token, end_org))
    except ApiError as exc:
        if exc.status in (401, 403, 404, 500):
            return False
        raise
    if status != 200:
        return False
    if not isinstance(payload, dict):
        return False
    if payload.get("code") not in (0, "0", None):
        return False
    data = payload.get("data")
    if not isinstance(data, dict) or not data.get("username"):
        return False
    return True


def random_alnum(length: int = 4) -> str:
    alphabet = string.digits + string.ascii_uppercase + string.ascii_lowercase
    return "".join(random.choice(alphabet) for _ in range(length))


def encode_login_password(password: str) -> str:
    # Match the current HPC4 frontend implementation exactly.
    return random_alnum(4) + base64.b64encode((random_alnum(4) + password).encode("utf-8")).decode("ascii")


def login_with_credentials(base_url: str, username: str, password: str) -> tuple[str, str | None]:
    url = f"{base_url.rstrip('/')}{USER_API_PREFIX}/login"
    headers = {
        "Origin": base_url.rstrip("/"),
        "Referer": LOGIN_APP_URL,
        "X-Requested-With": "XMLHttpRequest",
    }
    payload = {
        "username": username.strip(),
        "password": encode_login_password(password),
    }
    try:
        _status, body = json_request(url, method="POST", headers=headers, payload=payload)
    except ApiError as exc:
        raise SystemExit(f"Credential login failed: {exc.detail}") from exc
    if not isinstance(body, dict):
        raise RuntimeError(f"Unexpected login response: {body!r}")
    token = body.get("token")
    if not isinstance(token, str) or not token:
        raise RuntimeError(f"Login did not return a token: {body}")
    ticket = body.get("ticket")
    if ticket is not None and not isinstance(ticket, str):
        ticket = str(ticket)
    return token, ticket


def exchange_ticket_for_aistudio_token(
    base_url: str,
    username: str,
    ticket: str | None,
) -> str:
    if not ticket:
        raise RuntimeError("Portal login did not return a ticket, so AIStudio OAuth cannot continue.")

    redirect_uri = f"{base_url.rstrip('/')}{IAM_API_PREFIX}/auth-callback/oauth"
    params = {
        "client_id": THIRD_LOGIN_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": THIRD_LOGIN_SCOPE,
        "state": THIRD_LOGIN_STATE,
        "username": username.strip(),
        "ticket": ticket,
    }
    jar = CookieJar()
    opener = build_opener(HTTPCookieProcessor(jar))
    url = f"{base_url.rstrip('/')}/hpc-user/oauth2/authorize?{urlencode(params)}"
    request = Request(url, headers={"Referer": THIRD_LOGIN_REFERER}, method="GET")

    try:
        with opener.open(request, timeout=30):
            pass
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ApiError(url, exc.code, detail) from exc
    except URLError as exc:
        raise RuntimeError(f"Request failed for {url}: {exc}") from exc

    for cookie in jar:
        if cookie.name == "token" and cookie.value:
            return cookie.value

    raise RuntimeError("AIStudio OAuth finished without setting the expected token cookie.")


def get_or_refresh_token(base_url: str) -> AuthResult:
    base_url = base_url.rstrip("/")
    existing_token = os.environ.get(TOKEN_ENV_KEY)
    existing_ticket = os.environ.get(TICKET_ENV_KEY)
    existing_end_org = os.environ.get(END_ORG_ENV_KEY)

    if existing_token:
        payload = decode_jwt_payload(existing_token)
        inferred_end_org = existing_end_org or payload.get("organization_account")
        if validate_token(base_url, existing_token, inferred_end_org):
            return AuthResult(
                base_url=base_url,
                token=existing_token,
                ticket=existing_ticket,
                end_org=existing_end_org or payload.get("organization_account"),
                source="existing_token",
            )

    username = env_first(LOGIN_USER_ENV_KEYS)
    password = env_first(LOGIN_PASSWORD_ENV_KEYS)
    if not username or not password:
        raise SystemExit(
            "No valid HPC4 token is available. Set HPC4_AISTUDIO_TOKEN or provide "
            "login credentials through hpc4user/HPC4USER and hpc4password/HPC4PASSWORD."
        )

    _portal_token, ticket = login_with_credentials(base_url, username, password)
    token = exchange_ticket_for_aistudio_token(base_url, username, ticket)
    payload = decode_jwt_payload(token)
    return AuthResult(
        base_url=base_url,
        token=token,
        ticket=ticket,
        end_org=existing_end_org or payload.get("organization_account"),
        source="credential_login_oauth",
    )


def shell_exports(result: AuthResult) -> str:
    lines = [f"export {TOKEN_ENV_KEY}={shlex.quote(result.token)}"]
    if result.ticket:
        lines.append(f"export {TICKET_ENV_KEY}={shlex.quote(result.ticket)}")
    if result.end_org:
        lines.append(f"export {END_ORG_ENV_KEY}={shlex.quote(result.end_org)}")
    return "\n".join(lines)
