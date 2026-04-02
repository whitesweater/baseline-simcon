#!/usr/bin/env python3
"""Refresh a local AIStudio NPU machine pool file."""

from __future__ import annotations

import argparse
import base64
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

from aistudio_auth import DEFAULT_BASE_URL, get_or_refresh_token

DEFAULT_OUTPUT = Path.home() / ".codex" / "tmp" / "hpc4_runtime" / "npu_pool.json"
API_PREFIX = "/ai-arts/api/v1"
RUNNING_STATUS_CODE = 7
RUNNING_STATUS_LABELS = {"running", "运行中"}


class AistudioConfig(dict[str, Any]):
    """Tiny mapping-like config wrapper."""

    @property
    def base_url(self) -> str:
        return self["base_url"]

    @property
    def project_id(self) -> int:
        return self["project_id"]

    @property
    def project_url(self) -> str | None:
        return self["project_url"]

    @property
    def token(self) -> str:
        return self["token"]

    @property
    def end_org(self) -> str | None:
        return self["end_org"]

    @property
    def page_size(self) -> int:
        return self["page_size"]

    @property
    def auth_source(self) -> str:
        return self["auth_source"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Query AIStudio code-lab containers for an HPC4 project and "
            "refresh a local NPU pool JSON file."
        )
    )
    parser.add_argument(
        "--project-url",
        help="Full AIStudio expertDevelop page URL. proId is parsed from the query string.",
    )
    parser.add_argument(
        "--project-id",
        type=int,
        help="AIStudio project id. Use this if project-url is not available.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help=f"AIStudio base URL. Defaults to the project URL origin or {DEFAULT_BASE_URL}.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Where to write the refreshed pool file. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=100,
        help="Page size for the code-lab list API. Default: 100.",
    )
    return parser.parse_args()


def infer_project(args: argparse.Namespace) -> tuple[int, str | None, str]:
    project_url = args.project_url
    project_id = args.project_id
    base_url = args.base_url

    if project_url:
        parsed = urlparse(project_url)
        if not base_url:
            base_url = f"{parsed.scheme}://{parsed.netloc}"
        query = parse_qs(parsed.query)
        if project_id is None:
            pro_ids = query.get("proId", [])
            if not pro_ids:
                raise SystemExit("project-url is missing proId in its query string")
            project_id = int(pro_ids[0])

    if project_id is None:
        raise SystemExit("Provide either --project-url or --project-id")

    return project_id, project_url, (base_url or DEFAULT_BASE_URL).rstrip("/")


def load_config(args: argparse.Namespace) -> AistudioConfig:
    project_id, project_url, base_url = infer_project(args)
    auth = get_or_refresh_token(base_url)
    return AistudioConfig(
        base_url=base_url,
        project_id=project_id,
        project_url=project_url,
        token=auth.token,
        end_org=auth.end_org,
        page_size=args.page_size,
        auth_source=auth.source,
    )


def api_get(config: AistudioConfig, path: str) -> dict[str, Any]:
    url = f"{config.base_url}{API_PREFIX}{path}"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {config.token}",
    }
    if config.end_org:
        headers["end-org"] = config.end_org

    request = Request(url, headers=headers, method="GET")
    try:
        with urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {exc.code} for {url}: {detail}") from exc
    except URLError as exc:
        raise SystemExit(f"Request failed for {url}: {exc}") from exc

    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON from {url}: {exc}") from exc

    if payload.get("code") not in (0, "0", None):
        raise SystemExit(f"API error for {url}: {payload}")
    return payload


def fetch_code_lab_list(config: AistudioConfig) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    page_num = 1
    while True:
        payload = api_get(
            config,
            f"/projects/{config.project_id}/code-lab/list?pageNum={page_num}&pageSize={config.page_size}",
        )
        data = payload.get("data") or {}
        page_items = data.get("items") or []
        items.extend(page_items)
        if len(page_items) < config.page_size:
            break
        page_num += 1
    return items


def is_npu_entry(item: dict[str, Any]) -> bool:
    if item.get("resourceType") == "NPU":
        return True
    device = ((item.get("resourceInfo") or {}).get("device") or {})
    compute_type = str(device.get("computeType") or "").lower()
    device_type = str(device.get("deviceType") or "").lower()
    series = str(device.get("series") or "").lower()
    return (
        compute_type == "huawei_npu"
        or "ascend" in device_type
        or series.startswith("910")
    )


def is_running_entry(item: dict[str, Any]) -> bool:
    status = item.get("status")
    if status in (RUNNING_STATUS_CODE, str(RUNNING_STATUS_CODE)):
        return True
    for key in ("statusName", "statusText", "state", "stateText"):
        value = item.get(key)
        if isinstance(value, str) and value.strip().lower() in RUNNING_STATUS_LABELS:
            return True
    return False


def maybe_decode_base64(value: str | None) -> str | None:
    if not value:
        return None
    padded = value + "=" * (-len(value) % 4)
    try:
        decoded = base64.b64decode(padded.encode("ascii"))
        return decoded.decode("utf-8")
    except Exception:
        return value


def parse_ssh_command(command: str | None) -> dict[str, Any]:
    if not command:
        return {}
    parts = command.split()
    result: dict[str, Any] = {"command": command}
    if len(parts) >= 4 and parts[0] == "ssh" and parts[1] == "-p":
        result["port"] = int(parts[2])
        if "@" in parts[3]:
            user, host = parts[3].split("@", 1)
            result["user"] = user
            result["host"] = host
    return result


def simplify_service(endpoint: dict[str, Any]) -> dict[str, Any]:
    data = {
        "name": endpoint.get("name"),
        "status": endpoint.get("status"),
        "url": endpoint.get("url"),
        "port": endpoint.get("port"),
        "service_name": endpoint.get("serviceName"),
    }
    if endpoint.get("nodePort") is not None:
        data["node_port"] = endpoint.get("nodePort")
    if endpoint.get("addr"):
        data["addr"] = endpoint.get("addr")
    return data


def fetch_entry_detail(config: AistudioConfig, run_id: str) -> dict[str, Any]:
    payload = api_get(config, f"/projects/{config.project_id}/code-lab?runId={run_id}")
    return payload.get("data") or {}


def fetch_endpoints(config: AistudioConfig, lab_id: int, run_id: str) -> list[dict[str, Any]]:
    payload = api_get(
        config,
        f"/projects/{config.project_id}/code-lab/{lab_id}/runs/{run_id}/endpoints",
    )
    data = payload.get("data") or []
    if not isinstance(data, list):
        return []
    return data


def build_pool_entry(
    config: AistudioConfig,
    item: dict[str, Any],
) -> dict[str, Any] | None:
    lab_id = item.get("labId")
    run_id = item.get("runId")
    if not lab_id or not run_id:
        return None

    detail = fetch_entry_detail(config, run_id)
    endpoints = fetch_endpoints(config, int(lab_id), str(run_id))

    ssh_endpoint = next((e for e in endpoints if e.get("name") == "$ssh"), None)
    if not ssh_endpoint:
        return None

    ssh = parse_ssh_command(ssh_endpoint.get("url"))
    ssh["status"] = ssh_endpoint.get("status")
    ssh["access_key"] = ssh_endpoint.get("access_key")
    ssh["password"] = maybe_decode_base64(ssh_endpoint.get("secret_key"))
    ssh["addr"] = ssh_endpoint.get("addr")
    ssh["node_port"] = ssh_endpoint.get("nodePort")

    services: dict[str, Any] = {}
    for endpoint in endpoints:
        name = str(endpoint.get("name") or "").lstrip("$")
        if not name:
            continue
        services[name] = simplify_service(endpoint)

    resource_info = item.get("resourceInfo") or {}
    device = resource_info.get("device") or {}

    return {
        "name": item.get("name"),
        "project_id": item.get("projectId") or config.project_id,
        "lab_id": lab_id,
        "run_id": run_id,
        "resource_type": item.get("resourceType"),
        "debug_type": item.get("debugType"),
        "status": item.get("status"),
        "ready": True,
        "npu": {
            "arch": resource_info.get("arch"),
            "cpu": int(resource_info.get("cpu") or 0) or None,
            "memory_bytes": int(resource_info.get("memory") or 0) or None,
            "node_count": resource_info.get("node"),
            "device_num": int(device.get("deviceNum") or 0) or None,
            "device_type": device.get("deviceType"),
            "series": device.get("series"),
            "compute_type": device.get("computeType"),
        },
        "image": {
            "name": item.get("imageName"),
            "id": item.get("imageId"),
            "source": item.get("imageSource"),
        },
        "paths": {
            "code_path": detail.get("codePath"),
            "output_path": detail.get("outputPath"),
            "user_data_path": detail.get("userDataPath"),
            "team_data_path": detail.get("teamDataPath"),
            "user_conda_path": detail.get("userCondaPath"),
            "sample_models_path": detail.get("sampleModelsPath"),
            "dataset_path": detail.get("datasetPath"),
        },
        "ssh": ssh,
        "services": services,
    }


def build_pool(config: AistudioConfig) -> dict[str, Any]:
    items = fetch_code_lab_list(config)
    npu_items = [item for item in items if is_npu_entry(item)]
    running_items = [item for item in npu_items if is_running_entry(item)]

    pool_items = []
    for item in running_items:
        try:
            pool_item = build_pool_entry(config, item)
        except SystemExit:
            raise
        except Exception as exc:
            print(
                f"warning: failed to build pool entry for run {item.get('runId')}: {exc}",
                file=sys.stderr,
            )
            continue
        if pool_item:
            pool_items.append(pool_item)

    return {
        "version": 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "base_url": config.base_url,
            "project_id": config.project_id,
            "project_url": config.project_url,
            "auth_source": config.auth_source,
            "ready_rule": "resourceType == NPU and status == 7 (Running) and ssh endpoint exists",
        },
        "stats": {
            "total_code_lab_entries": len(items),
            "npu_entries": len(npu_items),
            "running_npu_entries": len(running_items),
            "pool_entries": len(pool_items),
        },
        "items": pool_items,
    }


def main() -> int:
    args = parse_args()
    config = load_config(args)
    pool = build_pool(config)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(pool, ensure_ascii=False, indent=2) + "\n")

    print(
        f"Wrote {pool['stats']['pool_entries']} running NPU pool entries to {output_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
