#!/usr/bin/env python3
"""Helpers for reaching the HPC2 portal through the validated remote VPN jump host."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


DEFAULT_REMOTE_HOST = "ubuntu@43.134.118.168"
DEFAULT_REMOTE_PROXY = "socks5h://127.0.0.1:1080"
DEFAULT_PORTAL_HOST = "hpc2login.hpc.hkust-gz.edu.cn"


def default_fetch_script() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "hpc2-gpu-containers"
        / "scripts"
        / "fetch_hpc2_gpu_containers.py"
    )


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if value:
        return value
    shell_cmd = (
        "source ~/.profile >/dev/null 2>&1 || true; "
        "source ~/.bashrc >/dev/null 2>&1 || true; "
        f"printf '%s' \"${name}\""
    )
    proc = subprocess.run(
        ["bash", "-lc", shell_cmd],
        text=True,
        capture_output=True,
    )
    fallback = proc.stdout
    if fallback:
        return fallback
    raise SystemExit(f"Missing environment variable: {name}")


def portal_username(explicit: str | None) -> str:
    if explicit:
        return explicit
    return (
        os.environ.get("HPC2_USER")
        or os.environ.get("HPC2_USERNAME")
        or _lookup_shell_env("HPC2_USER")
        or _lookup_shell_env("HPC2_USERNAME")
        or ""
    )


def _lookup_shell_env(name: str) -> str:
    shell_cmd = (
        "source ~/.profile >/dev/null 2>&1 || true; "
        "source ~/.bashrc >/dev/null 2>&1 || true; "
        f"printf '%s' \"${name}\""
    )
    proc = subprocess.run(
        ["bash", "-lc", shell_cmd],
        text=True,
        capture_output=True,
    )
    return proc.stdout


def _remote_fetch_args(username: str, passthrough: list[str]) -> str:
    args = [
        "--username",
        username,
        "--password-env",
        "HPC2_PASS",
        *passthrough,
    ]
    return " ".join(shlex.quote(arg) for arg in args)


def _remote_port_check_command(remote_proxy: str) -> str | None:
    parsed = urlparse(remote_proxy)
    if not parsed.hostname or not parsed.port:
        return None

    inner_cmd = (
        "python3 -c "
        + shlex.quote(
            "import socket; "
            f"socket.create_connection(({parsed.hostname!r}, {parsed.port}), 5).close()"
        )
    )
    return "bash -lc " + shlex.quote(inner_cmd)


def ensure_remote_proxy_ready(*, remote_host: str, remote_proxy: str) -> None:
    check_cmd = _remote_port_check_command(remote_proxy)
    if not check_cmd:
        return

    proc = subprocess.run(
        [
            "ssh",
            "-T",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "LogLevel=ERROR",
            remote_host,
            check_cmd,
        ],
        text=True,
        capture_output=True,
    )
    if proc.returncode == 0:
        return

    parsed = urlparse(remote_proxy)
    message = proc.stderr.strip() or proc.stdout.strip() or "connection check failed"
    raise SystemExit(
        "Remote VPN proxy is not ready on the jump host: "
        f"{parsed.hostname}:{parsed.port} via {remote_host}. "
        "Restore the jump-host EasyConnect/SOCKS service before querying the HPC2 portal. "
        f"Remote check output: {message}"
    )


def run_remote_fetch(
    *,
    username: str | None = None,
    local_password_env: str = "HPC2_PASS",
    remote_host: str = DEFAULT_REMOTE_HOST,
    remote_proxy: str = DEFAULT_REMOTE_PROXY,
    fetch_script_path: str | Path | None = None,
    passthrough: list[str] | None = None,
) -> str:
    user = portal_username(username)
    if not user:
        raise SystemExit("Missing portal username. Set HPC2_USER or pass --username.")

    password = require_env(local_password_env)
    fetch_path = Path(fetch_script_path) if fetch_script_path else default_fetch_script()
    script_text = fetch_path.read_text()
    remote_args = _remote_fetch_args(user, passthrough or [])
    ensure_remote_proxy_ready(remote_host=remote_host, remote_proxy=remote_proxy)

    remote_cmd = "\n".join(
        [
            "read -r HPC2_PASS",
            "export HPC2_PASS",
            f"export ALL_PROXY={shlex.quote(remote_proxy)}",
            f"export HTTPS_PROXY={shlex.quote(remote_proxy)}",
            f"export HTTP_PROXY={shlex.quote(remote_proxy)}",
            "tmp=$(mktemp)",
            "trap 'rm -f \"$tmp\"' EXIT",
            "cat >\"$tmp\"",
            f"python3 \"$tmp\" {remote_args}",
        ]
    )

    full_remote_cmd = "bash -c " + shlex.quote(remote_cmd)

    proc = subprocess.run(
        [
            "ssh",
            "-T",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "LogLevel=ERROR",
            remote_host,
            full_remote_cmd,
        ],
        input=password + "\n" + script_text,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        message = proc.stderr.strip() or proc.stdout.strip() or "remote fetch failed"
        raise SystemExit(message)
    return proc.stdout


def fetch_payload(
    *,
    username: str | None = None,
    local_password_env: str = "HPC2_PASS",
    remote_host: str = DEFAULT_REMOTE_HOST,
    remote_proxy: str = DEFAULT_REMOTE_PROXY,
    include_sensitive: bool = False,
    status: str = "all",
) -> dict[str, Any]:
    passthrough = ["--status", status, "--format", "json"]
    if include_sensitive:
        passthrough.append("--include-sensitive")
    output = run_remote_fetch(
        username=username,
        local_password_env=local_password_env,
        remote_host=remote_host,
        remote_proxy=remote_proxy,
        passthrough=passthrough,
    )
    return json.loads(output)


def running_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in payload.get("items", [])
        if str(item.get("status") or item.get("serviceStatus") or "").strip() in {"运行", "running", "Running"}
    ]


def _find_key(obj: Any, target_keys: tuple[str, ...]) -> Any | None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in target_keys and value not in (None, ""):
                return value
        for value in obj.values():
            found = _find_key(value, target_keys)
            if found not in (None, ""):
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _find_key(value, target_keys)
            if found not in (None, ""):
                return found
    return None


def resolve_container_connection(item: dict[str, Any]) -> dict[str, Any]:
    service_name = (
        _find_key(item, ("serviceName", "name")) or ""
    )
    user = _find_key(item, ("sshUser", "userName", "sshAccount")) or service_name
    host = _find_key(item, ("sshHost", "sshIp", "serviceHost", "serviceIp", "host"))
    port = _find_key(item, ("sshPort", "port"))
    password = _find_key(item, ("sshPassword", "password"))

    for url_key in ("sshUrl", "serviceUrl", "proxyUrl", "innerProxyUrl"):
        value = _find_key(item, (url_key,))
        if not isinstance(value, str) or not value:
            continue
        match = re.search(r"(?:(?P<user>[^@/:]+)@)?(?P<host>[A-Za-z0-9_.-]+):(?P<port>\d+)", value)
        if not match:
            continue
        user = user or match.group("user")
        host = host or match.group("host")
        port = port or match.group("port")

    resolved_host = str(host or DEFAULT_PORTAL_HOST)
    resolved_user = str(user or service_name)
    resolved_port = int(str(port)) if port not in (None, "") else None
    resolved_password = str(password or "")

    if not resolved_user:
        raise SystemExit("Could not resolve the container SSH user from the portal payload.")
    if resolved_port is None:
        raise SystemExit("Could not resolve the container SSH port from the portal payload.")
    if not resolved_password:
        raise SystemExit("Could not resolve the container SSH password from the portal payload.")

    return {
        "service_name": str(service_name),
        "user": resolved_user,
        "host": resolved_host,
        "port": resolved_port,
        "password": resolved_password,
    }


def choose_container(
    payload: dict[str, Any],
    *,
    service_name: str | None = None,
    index: int | None = None,
    running_only: bool = True,
) -> dict[str, Any]:
    items = running_items(payload) if running_only else payload.get("items", [])
    if service_name:
        for item in items:
            name = item.get("serviceName") or item.get("name")
            if name == service_name:
                return item
        raise SystemExit(f"Container not found in the selected list: {service_name}")
    if index is not None:
        if index < 1 or index > len(items):
            raise SystemExit(f"Container index out of range: {index}")
        return items[index - 1]
    if len(items) == 1:
        return items[0]

    names = [str(item.get("serviceName") or item.get("name") or "<unnamed>") for item in items]
    if not names:
        raise SystemExit("No matching containers were found.")
    raise SystemExit(
        "Multiple matching containers exist. Pass --service-name or --index. "
        + "Candidates: "
        + ", ".join(names)
    )
