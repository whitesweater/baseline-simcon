# HPC2 Remote-VPN Notes

## Validated Path

- Jump host: `ubuntu@43.134.118.168`
- Jump-host campus proxy: `127.0.0.1:1080`
- Local workstation symptom: direct portal access can fail with SSL EOF or unresolved campus DNS
- Reliable workaround: run the portal fetch on the jump host and force it through `socks5h://127.0.0.1:1080`

## Health Check

- Do not assume the jump-host SOCKS proxy is always online.
- Fast check:

```bash
ssh ubuntu@43.134.118.168 'bash -lc '"'"'python3 -c "import socket; socket.create_connection((\"127.0.0.1\", 1080), 5).close()"'"'"''
```

- If this fails, the remote EasyConnect path is down and portal access will fail until that service is restored.

## Container-State Interpretation

- `运行`: ready for SSH and GPU work
- `等待`: resource request exists but there is no ready runtime yet
- `退出`: stopped or stale, not a live target

## Connection Heuristic

In the current validated environment:

- the container SSH user often matches `serviceName`
- the SSH host can remain `hpc2login.hpc.hkust-gz.edu.cn`
- the decisive runtime field is usually the per-container SSH port plus the container password

The connector script still prefers explicit portal fields first and only falls back to these heuristics when needed.
