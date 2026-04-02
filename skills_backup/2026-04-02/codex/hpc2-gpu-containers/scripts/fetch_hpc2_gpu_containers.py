#!/usr/bin/env python3
"""Login to the HPC2 portal and read Model Develop GPU containers."""

from __future__ import annotations

import argparse
import base64
import getpass
import json
import os
import re
import sys
import time
from typing import Any

import requests
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import padding


LOGIN_PATH = "/appform/login"
LOGIN_JS_PATH = "/appform/js/login/login.js"
SSO_PATH = "/appform/sso/main"
CHECK_PATH = "/appform/j_spring_security_check"
DESKTOP_PATH = "/appform/desktop"
AIBASE_PATH = "/jhai/aiBase/"
LIST_PATH = "/jhai/dockerService/listByModule"
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)
SENSITIVE_KEYS = {
    "serviceEnvs",
    "serviceCmd",
    "serviceWork",
    "serviceMounts",
    "innerProxyUrl",
    "proxyUrl",
    "serviceUrl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Log into the HPC2 web desktop and list Model Develop GPU containers."
    )
    parser.add_argument(
        "--base-url",
        default="https://hpc2login.hpc.hkust-gz.edu.cn",
        help="Portal base URL. Default: %(default)s",
    )
    parser.add_argument(
        "--username",
        default=os.environ.get("HPC2_USER") or os.environ.get("HPC2_USERNAME"),
        help="Portal username. Defaults to $HPC2_USER or $HPC2_USERNAME.",
    )
    parser.add_argument(
        "--password",
        help="Portal password. Prefer --password-env or interactive prompt.",
    )
    parser.add_argument(
        "--password-env",
        help="Environment variable name that stores the portal password.",
    )
    parser.add_argument(
        "--module-type",
        type=int,
        default=1,
        help="JHAI module type. Model Develop uses 1.",
    )
    parser.add_argument(
        "--status",
        choices=["all", "running", "exited"],
        default="all",
        help="Filter returned containers by status.",
    )
    parser.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format.",
    )
    parser.add_argument(
        "--include-sensitive",
        action="store_true",
        help="Include sensitive API fields. Use only when explicitly requested.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Per-request timeout in seconds.",
    )
    return parser.parse_args()


def require_username(args: argparse.Namespace) -> str:
    if args.username:
        return args.username
    raise SystemExit("Missing username. Pass --username or set $HPC2_USER.")


def resolve_password(args: argparse.Namespace) -> str:
    if args.password:
        return args.password
    if args.password_env:
        value = os.environ.get(args.password_env)
        if value is None:
            raise SystemExit(f"Environment variable not set: {args.password_env}")
        return value
    if sys.stdin.isatty():
        return getpass.getpass("HPC2 password: ")
    raise SystemExit("Missing password. Pass --password, --password-env, or run interactively.")


def build_headers(referer: str) -> dict[str, str]:
    return {
        "User-Agent": USER_AGENT,
        "Referer": referer,
    }


def expect_json(response: requests.Response) -> Any:
    content_type = response.headers.get("Content-Type", "")
    if "application/json" not in content_type:
        snippet = response.text[:200].replace("\n", " ").strip()
        raise RuntimeError(
            f"Expected JSON from {response.url}, got {content_type or 'unknown'}: {snippet}"
        )
    return response.json()


def fetch_public_key(session: requests.Session, base_url: str, timeout: float) -> str:
    response = session.get(
        base_url + LOGIN_JS_PATH,
        headers=build_headers(base_url + LOGIN_PATH),
        timeout=timeout,
    )
    response.raise_for_status()
    match = re.search(r'var publicKey = "([^"]+)"', response.text)
    if not match:
        raise RuntimeError("Could not find the RSA public key in login.js.")
    return match.group(1)


def encrypt_password(public_key_b64: str, password: str) -> str:
    public_key = serialization.load_der_public_key(base64.b64decode(public_key_b64))
    encrypted = public_key.encrypt(password.encode("utf-8"), padding.PKCS1v15())
    return base64.b64encode(encrypted).decode("ascii")


def login(session: requests.Session, base_url: str, username: str, password: str, timeout: float) -> None:
    login_url = base_url + LOGIN_PATH
    session.get(login_url, headers=build_headers(login_url), timeout=timeout).raise_for_status()

    public_key = fetch_public_key(session, base_url, timeout)
    encrypted_password = encrypt_password(public_key, password)

    sso_payload = {
        "userName": username,
        "loginTime": int(time.time() * 1000),
        "password": encrypted_password,
    }
    sso_response = session.get(
        base_url + SSO_PATH,
        params={"uinfo": json.dumps(sso_payload, separators=(",", ":"))},
        headers=build_headers(login_url),
        timeout=timeout,
    )
    sso_json = expect_json(sso_response)
    if sso_json.get("code") != 200:
        raise RuntimeError(f"SSO check failed: {sso_json.get('message') or sso_json}")

    post_headers = build_headers(login_url)
    post_headers["Origin"] = base_url
    login_response = session.post(
        base_url + CHECK_PATH,
        data={"j_username": username, "j_password": encrypted_password},
        headers=post_headers,
        allow_redirects=True,
        timeout=timeout,
    )
    login_response.raise_for_status()

    desktop_response = session.get(
        base_url + DESKTOP_PATH,
        headers=build_headers(login_url),
        allow_redirects=True,
        timeout=timeout,
    )
    desktop_response.raise_for_status()
    if "/appform/login" in desktop_response.url:
        raise RuntimeError("Login did not reach the desktop page.")
    if 'action="/appform/j_spring_security_check"' in desktop_response.text:
        raise RuntimeError("Login fell back to the login form instead of the desktop page.")
    if "modelDevelop" not in desktop_response.text and "Logout" not in desktop_response.text:
        raise RuntimeError("Desktop page loaded, but the expected app metadata was not found.")

    aibase_response = session.get(
        base_url + AIBASE_PATH,
        headers=build_headers(base_url + DESKTOP_PATH),
        timeout=timeout,
    )
    aibase_response.raise_for_status()


def fetch_container_payload(
    session: requests.Session,
    base_url: str,
    module_type: int,
    timeout: float,
) -> dict[str, Any]:
    response = session.post(
        base_url + LIST_PATH,
        json={"moduleType": module_type},
        headers=build_headers(base_url + "/jhai/aiBase/#/devEnv"),
        timeout=timeout,
    )
    response.raise_for_status()
    payload = expect_json(response)
    if not isinstance(payload, dict) or "data" not in payload:
        raise RuntimeError("Unexpected container list response shape.")
    return payload


def normalize_status(status: str | None) -> str:
    value = (status or "").strip()
    if value in {"运行", "running", "Running"}:
        return "running"
    if value in {"退出", "stopped", "Stopped"}:
        return "exited"
    return "other"


def keep_item(item: dict[str, Any], status_filter: str) -> bool:
    if status_filter == "all":
        return True
    return normalize_status(item.get("serviceStatus")) == status_filter


def summarize_gpu_binds(item: dict[str, Any]) -> list[str]:
    summaries: list[str] = []
    for bind in item.get("gpuBinds") or []:
        ids = ",".join(str(value) for value in bind.get("ID") or [])
        gpu_info = bind.get("GPUInfo") or []
        gpu_type = gpu_info[0].get("GPUType", "") if gpu_info else ""
        host = bind.get("Host", "")
        label = f"{host}[{ids}]".strip()
        if gpu_type:
            label = f"{label} {gpu_type}".strip()
        summaries.append(label)
    return summaries


def redact_sensitive_fields(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            if key in SENSITIVE_KEYS:
                continue
            if key == "sshPassword":
                continue
            redacted[key] = redact_sensitive_fields(item)
        return redacted
    if isinstance(value, list):
        return [redact_sensitive_fields(item) for item in value]
    return value


def normalize_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": item.get("id"),
        "serviceName": item.get("serviceName"),
        "serviceType": item.get("serviceType"),
        "status": item.get("serviceStatus"),
        "submitTime": item.get("submitTime"),
        "updateTime": item.get("updateTime"),
        "node": item.get("serviceNode"),
        "cpu": item.get("cpuNum"),
        "gpu": item.get("gpuNum"),
        "clusterAlias": item.get("clusterAlias"),
        "resourceComboName": item.get("resourceComboName"),
        "imageName": item.get("imageName"),
        "jobId": item.get("jobId"),
        "gpuBinds": summarize_gpu_binds(item),
    }


def print_table(items: list[dict[str, Any]]) -> None:
    if not items:
        print("No matching containers.")
        return
    for index, item in enumerate(items, start=1):
        binds = "; ".join(item.get("gpuBinds") or []) or "-"
        print(
            f"[{index}] {item.get('serviceName')} | "
            f"{item.get('serviceType')} | "
            f"{item.get('status')} | "
            f"node={item.get('node')} | "
            f"cpu={item.get('cpu')} | "
            f"gpu={item.get('gpu')} | "
            f"jobId={item.get('jobId')} | "
            f"{binds}"
        )


def main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    username = require_username(args)
    password = resolve_password(args)

    session = requests.Session()
    login(session, base_url, username, password, args.timeout)
    payload = fetch_container_payload(session, base_url, args.module_type, args.timeout)
    raw_items = payload.get("data") or []
    filtered_items = [item for item in raw_items if keep_item(item, args.status)]

    if args.include_sensitive:
        output_items = filtered_items
    else:
        output_items = [normalize_item(redact_sensitive_fields(item)) for item in filtered_items]

    if args.format == "json":
        print(
            json.dumps(
                {
                    "count": len(output_items),
                    "items": output_items,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.include_sensitive:
        print(
            "Sensitive output requested. Use --format json if you need to inspect the full payload."
        )
        print(json.dumps({"count": len(output_items), "items": output_items}, ensure_ascii=False))
        return 0

    print_table(output_items)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
