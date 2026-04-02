# Remote Host Registry

Use this file as the structured registry for remote machines and their allowed workdirs.

## Rules

- A single machine can have multiple allowed workdirs.
- Multiple aliases can point to the same physical machine if auth or purpose differs.
- If a user request matches several host/workdir pairs, list every plausible pair and wait for the user to choose.
- Keep new entries concise and factual.

## Registered Hosts

### hpc1

- ssh_alias: `hpc1`
- host_name: `hpc1login.hpc.hkust-gz.edu.cn`
- user: `yhao481`
- port: `22`
- identity_file: `~/.ssh/id_rsa`
- keepalive: `ServerAliveInterval 60`, `ServerAliveCountMax 3`, `TCPKeepAlive yes`, `AddKeysToAgent yes`
- allowed_workdirs:
  - `/hpc2hdd/home/yhao481/jhupload/proj`
- likely_project_paths:
  - `/hpc2hdd/home/yhao481/jhupload/proj/<project>`
- notes:
  - HPC1 login node.
  - Shares the same path convention as HPC2.

### hpc2

- ssh_alias: `hpc2`
- host_name: `hpc2login.hpc.hkust-gz.edu.cn`
- user: `yhao481`
- port: `22`
- identity_file: `~/.ssh/id_rsa`
- keepalive: `ServerAliveInterval 60`, `ServerAliveCountMax 3`, `TCPKeepAlive yes`, `AddKeysToAgent yes`
- allowed_workdirs:
  - `/hpc2hdd/home/yhao481/jhupload/proj`
- likely_project_paths:
  - `/hpc2hdd/home/yhao481/jhupload/proj/<project>`
- notes:
  - HPC2 login node.
  - Often the most convenient source machine for downloads and migrations.

### hpc3

- ssh_alias: `hpc3`
- host_name: `hpc3login.hpc.hkust-gz.edu.cn`
- user: `yhao481`
- port: `22`
- identity_file: `~/.ssh/id_rsa`
- keepalive: `ServerAliveInterval 60`, `ServerAliveCountMax 3`, `TCPKeepAlive yes`, `AddKeysToAgent yes`
- allowed_workdirs:
  - `/data/user/yhao481/proj`
- likely_project_paths:
  - `/data/user/yhao481/proj/<project>`
- notes:
  - HPC3 login node.
  - Public internet access is not available; prefer offline or pre-staged dependencies.

### hpc4

- ssh_alias: `hpc4`
- host_name: `hpc4login.hpc.hkust-gz.edu.cn`
- user: `user224`
- port: `22`
- identity_file: `~/.ssh/id_rsa`
- keepalive: `ServerAliveInterval 60`, `ServerAliveCountMax 3`, `TCPKeepAlive yes`, `AddKeysToAgent yes`
- allowed_workdirs:
  - `/data/user/user224/proj`
- likely_project_paths:
  - `/data/user/user224/proj/<project>`
- notes:
  - HPC4 CPU login node.
  - Heavy or NPU runs should move to AIStudio containers after the user chooses a running container.

### H800_dev_sg

- ssh_alias: `H800_dev_sg`
- host_name: `10.2.11.117`
- user: `root`
- port: `30636`
- identity_file: `~/.ssh/id_rsa`
- keepalive: `ServerAliveInterval 60`, `ServerAliveCountMax 3`, `TCPKeepAlive yes`
- allowed_workdirs:
  - `/data/yhao`
- likely_project_paths:
  - `/data/yhao/rank/CODI`
- notes:
  - Stay inside `/data/yhao`.
  - Current experiment work is typically under the CODI project in this tree.

## Entry Template

### <host-alias>

- ssh_alias: `<alias>`
- host_name: `<ip-or-hostname>`
- user: `<user>`
- port: `<port>`
- identity_file: `<identity-file>`
- allowed_workdirs:
  - `<workdir-1>`
- likely_project_paths:
  - `<project-path>`
- notes:
  - `<short constraint or usage note>`

### rbmproj

- ssh_alias: `rbmproj`
- host_name: `159.75.130.159`
- user: `yhao`
- port: `22`
- identity_file: `~/.ssh/id_rsa`
- keepalive: `ServerAliveInterval 60`, `ServerAliveCountMax 3`, `TCPKeepAlive yes`, `AddKeysToAgent yes`
- allowed_workdirs:
  - `/home/yhao/backend/social-media-backend`
- likely_project_paths:
  - `/home/yhao/backend/social-media-backend`
- notes:
  - Backend development workspace for the Redbird project.

### tencent_cliproxyapi

- ssh_alias: `43.134.118.168`
- host_name: `43.134.118.168`
- user: `ubuntu`
- port: `22`
- identity_file: `~/.ssh/id_rsa`
- keepalive: `ServerAliveInterval 60`, `ServerAliveCountMax 3`, `TCPKeepAlive yes`, `AddKeysToAgent yes`
- allowed_workdirs:
  - `/home/ubuntu/proj`
- likely_project_paths:
  - `/home/ubuntu/proj/CLIProxyAPI`
- notes:
  - Ubuntu host accessed via SSH key.
  - Stay inside `/home/ubuntu/proj` unless the user expands the scope.
