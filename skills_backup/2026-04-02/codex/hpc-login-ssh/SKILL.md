---
name: hpc-login-ssh
description: Use when connecting from the current Linux workstation to HPC1, HPC2, HPC3, or HPC4 login nodes with an existing SSH key or SSH config, and when you need the correct per-cluster user, SSH alias, and shared project root before doing remote work.
---

# HPC Login SSH

## Overview

This skill handles SSH access to HPC login nodes with the local SSH key and host config. It is the right path for HPC1-4 login hosts, where the hostname pattern is similar but the remote user and allowed workdir depend on the cluster.

On this workstation, the preferred short aliases are `hpc1`, `hpc2`, `hpc3`, and `hpc4`. If those aliases are not available in the global SSH config, use the bundled config file:

```bash
bash scripts/ssh_hpc.sh hpc2
```

## Cluster Map

| Cluster | SSH Alias | Login Host | User | Shared Project Root | Notes |
| --- | --- | --- | --- | --- | --- |
| HPC1 | `hpc1` | `hpc1login.hpc.hkust-gz.edu.cn` | `yhao481` | `/hpc2hdd/home/yhao481/jhupload/proj` | Same directory convention as HPC2 |
| HPC2 | `hpc2` | `hpc2login.hpc.hkust-gz.edu.cn` | `yhao481` | `/hpc2hdd/home/yhao481/jhupload/proj` | Usually the most convenient download or migration source |
| HPC3 | `hpc3` | `hpc3login.hpc.hkust-gz.edu.cn` | `yhao481` | `/data/user/yhao481/proj` | No public internet access |
| HPC4 | `hpc4` | `hpc4login.hpc.hkust-gz.edu.cn` | `user224` | `/data/user/user224/proj` | CPU login node; heavy NPU work moves to AIStudio containers |

## When To Use

- The user asks to connect to an HPC login node.
- The target is a login host such as `hpc1login`, `hpc2login`, `hpc3login`, or `hpc4login`.
- The connection should use a local SSH key or `~/.ssh/config`, not a password prompt.
- The next work step will happen under the cluster's shared project root.

## Workflow

### 0. Decide whether direct SSH is possible or a remote VPN jump is required

Default assumption:

- If the local workstation can already resolve and reach the HPC login host directly, use the normal aliases in `references/ssh_config_hpc`.
- If the workstation cannot reach campus hosts directly, but a remote jump host already has EasyConnect VPN access, use the remote-VPN path for the clusters that support local-key login through that path.

Current tested remote-VPN path:

- jump host: `ubuntu@43.134.118.168`
- remote campus SOCKS5: `127.0.0.1:1080` on that jump host
- tested local-key login alias: `hpc2-vpn`

Use:

```bash
bash scripts/ssh_hpc_via_remote_vpn.sh hpc2-vpn
```

The remote-VPN SSH config lives at:

- `references/ssh_config_hpc_via_remote_vpn`

Important boundary:

- This path is currently intended for `hpc2` access from this workstation.
- `hpc4` is reachable through the remote VPN, but the tested working login path is still the remote-side password script flow, not the local-key alias route.

### 1. Resolve the target login host

If the exact login node is not already explicit, enumerate the plausible hosts and wait for the user to choose:

- `hpc1`
- `hpc1login.hpc.hkust-gz.edu.cn`
- `hpc2`
- `hpc2login.hpc.hkust-gz.edu.cn`
- `hpc3`
- `hpc3login.hpc.hkust-gz.edu.cn`
- `hpc4`
- `hpc4login.hpc.hkust-gz.edu.cn`

Use [scripts/render_login_hosts.py](scripts/render_login_hosts.py) to print the canonical alias, hostname, user, and workdir map.

### 2. Prefer SSH config and local keys

Before connecting:

- Check `~/.ssh/config`
- Check whether the host already has an alias
- Reuse the local SSH private key path already configured for the machine

Current workstation default:

- private key: `~/.ssh/id_rsa`
- if aliases exist, prefer `ssh hpc1`, `ssh hpc2`, `ssh hpc3`, `ssh hpc4`
- if global `~/.ssh/config` is read-only or missing aliases, use `bash scripts/ssh_hpc.sh hpc1`
- otherwise use the full login hostname directly
- if direct campus reachability is unavailable but the remote-VPN route is known to work, use `bash scripts/ssh_hpc_via_remote_vpn.sh hpc2-vpn`

For login nodes, do not switch to password-based SSH unless the user explicitly says the key path is unavailable.

### 3. Verify the remote session immediately

After login, verify:

```bash
hostname
whoami
pwd
```

Then move into the cluster-specific shared project area:

```bash
cd <shared_project_root>/<project>
```

Use these roots:

- HPC1 / HPC2: `/hpc2hdd/home/yhao481/jhupload/proj`
- HPC3: `/data/user/yhao481/proj`
- HPC4: `/data/user/user224/proj`

If the requested project path is outside the chosen cluster's shared project root, stop and ask before editing anything.

### 4. Respect cluster-specific constraints

- HPC3 does not have public internet access. Do not plan package downloads there unless the user explicitly provides an internal mirror path.
- HPC4 login is for CPU-side development and light tasks. For AIStudio NPU containers and pool-driven HPC4 execution, switch to `hpc4-runtime`.

## Boundaries

- This skill is for login nodes, not AIStudio NPU containers.
- For AIStudio containers that use password SSH, use `hpc4-runtime` instead.
- For refreshing the list of runnable NPU containers, use `hpc4-runtime`.

## Resources

- Use [scripts/render_login_hosts.py](scripts/render_login_hosts.py) to print the canonical HPC1-4 login host map.
- Use [scripts/ssh_hpc.sh](scripts/ssh_hpc.sh) to connect through the bundled SSH config.
- Use [scripts/ssh_hpc_via_remote_vpn.sh](scripts/ssh_hpc_via_remote_vpn.sh) when this workstation must reach `hpc2` through the remote EasyConnect jump host.
- Use [references/ssh_config_hpc](references/ssh_config_hpc) as a dedicated SSH config snippet when the global SSH config cannot be edited.
- Use [references/ssh_config_hpc_via_remote_vpn](references/ssh_config_hpc_via_remote_vpn) for the tested `hpc2-vpn` alias that tunnels through `ubuntu@43.134.118.168` and its remote SOCKS5 proxy.
