#!/usr/bin/env python3
import base64
import http.cookiejar
import json
import random
import string
import sys
import urllib.parse
import urllib.request
from typing import Optional
from urllib.error import HTTPError


def rand4():
    alphabet = string.digits + string.ascii_uppercase + string.ascii_lowercase
    return ''.join(random.choice(alphabet) for _ in range(4))


def b64decode_maybe(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    value += '=' * (-len(value) % 4)
    try:
        return base64.b64decode(value.encode()).decode()
    except Exception:
        return value


def api_get(base_url: str, path: str, token: str, end_org: str):
    request = urllib.request.Request(
        base_url + path,
        headers={
            'Accept': 'application/json',
            'Authorization': 'Bearer ' + token,
            'end-org': end_org,
        },
    )
    return json.loads(urllib.request.urlopen(request, timeout=30).read().decode())


def get_token(base_url: str, user: str, password: str) -> str:
    encoded = rand4() + base64.b64encode((rand4() + password).encode()).decode()
    request = urllib.request.Request(
        base_url + '/hpc-user/api/login',
        data=json.dumps({'username': user, 'password': encoded}).encode(),
        headers={
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'Origin': base_url,
            'Referer': base_url + '/#/app/user',
            'X-Requested-With': 'XMLHttpRequest',
        },
        method='POST',
    )
    login_body = json.loads(urllib.request.urlopen(request, timeout=30).read().decode())
    ticket = login_body['ticket']

    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    params = urllib.parse.urlencode(
        {
            'client_id': 'a521840c6e829f70e3558237e97980ec',
            'redirect_uri': base_url + '/iam/api/v1/auth-callback/oauth',
            'response_type': 'code',
            'scope': 'all',
            'state': 'xyz',
            'username': user,
            'ticket': ticket,
        }
    )
    opener.open(
        urllib.request.Request(
            base_url + '/hpc-user/oauth2/authorize?' + params,
            headers={'Referer': base_url + '/hpc-user/#/third_login'},
        ),
        timeout=30,
    )
    for cookie in jar:
        if cookie.name == 'token' and cookie.value:
            return cookie.value
    raise SystemExit('missing AIStudio token cookie')


def main() -> int:
    if len(sys.argv) != 4:
        print('usage: python3 list_remote_npu_pool.py <user> <password> <project_id>', file=sys.stderr)
        return 2

    user = sys.argv[1]
    password = sys.argv[2]
    project_id = sys.argv[3]
    base_url = 'https://hpc4login.hpc.hkust-gz.edu.cn'
    end_org = 'apulis'

    token = get_token(base_url, user, password)
    items = (
        api_get(base_url, f'/ai-arts/api/v1/projects/{project_id}/code-lab/list?pageNum=1&pageSize=100', token, end_org)
        .get('data', {})
        .get('items', [])
    )

    status_counts = {}
    for item in items:
        key = str(item.get('status'))
        status_counts[key] = status_counts.get(key, 0) + 1

    result = {
        'project_id': int(project_id),
        'total_entries': len(items),
        'status_counts': status_counts,
        'running_npu_entries': [],
    }

    running = [item for item in items if item.get('resourceType') == 'NPU' and item.get('status') == 7]
    for item in running:
        run_id = str(item.get('runId'))
        lab_id = int(item.get('labId'))
        device = ((item.get('resourceInfo') or {}).get('device') or {})
        entry = {
            'name': item.get('name'),
            'run_id': run_id,
            'lab_id': lab_id,
            'npu': device.get('deviceNum'),
            'series': device.get('series'),
            'image': item.get('imageName'),
        }
        try:
            endpoints = (
                api_get(base_url, f'/ai-arts/api/v1/projects/{project_id}/code-lab/{lab_id}/runs/{run_id}/endpoints', token, end_org)
                .get('data', [])
            )
            ssh_endpoint = next((endpoint for endpoint in endpoints if endpoint.get('name') == '$ssh'), None)
            if ssh_endpoint:
                entry['ssh'] = ssh_endpoint.get('url')
                entry['ssh_password'] = b64decode_maybe(ssh_endpoint.get('secret_key'))
            else:
                entry['ssh'] = None
                entry['ssh_password'] = None
        except HTTPError as exc:
            entry['ssh_error'] = f'HTTP {exc.code}'
        except Exception as exc:
            entry['ssh_error'] = f'{type(exc).__name__}: {exc}'
        result['running_npu_entries'].append(entry)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
