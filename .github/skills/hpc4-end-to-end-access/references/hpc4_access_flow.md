# HPC4 Access Flow

## Goal

Provide the shortest repeatable path from a headless Linux machine into an HPC4 NPU container.

## End-To-End Chain

1. Docker EasyConnect VPN
2. SOCKS5 proxy validation
3. HPC4 login over the SOCKS5 proxy
4. AIStudio token exchange on the HPC4 login node
5. NPU pool refresh for project 670
6. User selects one running NPU container
7. SSH into the selected NPU container
8. Verify runtime state inside the container

## Split Points

### Split 1. VPN bootstrap

Inputs:

- VPN URL
- VPN username
- VPN password

Output:

- running `easyconnect` Docker container
- local SOCKS5 proxy at `127.0.0.1:1080`

Why split it:

- The VPN container is reusable across many HPC sessions.
- VPN failures are orthogonal to SSH and AIStudio failures.

### Split 2. VPN health check

Inputs:

- running EasyConnect container

Output:

- proof that campus portal and HPC4 login SSH port are reachable through SOCKS5

Why split it:

- It is the fastest preflight before wasting time on credentials or AIStudio APIs.

### Split 3. HPC4 login node access

Inputs:

- `hpc4user`
- `hpc4password`
- SOCKS5 proxy

Output:

- working SSH session on `hpc4login.hpc.hkust-gz.edu.cn`

Why split it:

- Login-node access is useful even when no NPU container is needed.

### Split 4. AIStudio NPU pool refresh

Inputs:

- HPC4 portal credentials
- AIStudio project id

Output:

- JSON summary of running NPU containers and their SSH connection info

Why split it:

- Container discovery changes frequently.
- Refreshing the pool is independent from entering any specific container.

### Split 5. NPU container selection and connect

Inputs:

- refreshed NPU pool summary
- user-selected container index

Output:

- interactive SSH session inside one NPU container

Why split it:

- Selection is a user decision, not a deterministic automation step.

### Split 6. In-container health check

Inputs:

- active NPU container session

Output:

- hostname
- current user
- current path
- `npu-smi info`

Why split it:

- It provides a minimal smoke test before any environment setup or training work.

## Fastest Repeat Path

```bash
/home/ubuntu/proj/skills/hpc4-docker-vpn/scripts/check_easyconnect_proxy.sh
/home/ubuntu/proj/skills/hpc4-end-to-end-access/scripts/hpc4_login_via_vpn.sh 'hostname; whoami; pwd'
/home/ubuntu/proj/skills/hpc4-end-to-end-access/scripts/refresh_npu_pool_via_hpc4.sh 670
cat ~/npu_pool_summary.json
/home/ubuntu/proj/skills/hpc4-end-to-end-access/scripts/connect_npu_from_summary.sh 1
```

## Notes

- In this environment, the local host does not resolve the HPC4 login hostname directly unless the access goes through the VPN SOCKS proxy.
- Refreshing the AIStudio pool from the HPC4 login node is more reliable than trying to call the AIStudio API directly from the local host.
- The pool summary contains decoded NPU container SSH passwords, so it must be treated as sensitive.
