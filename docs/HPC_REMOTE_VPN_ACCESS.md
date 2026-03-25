# HPC 通过远端 VPN 跳板访问说明

当前这台机器本身不能直接跑 EasyConnect 容器，但可以通过远端跳板机 `ubuntu@43.134.118.168` 上已经建立好的 EasyConnect 代理，访问 HPC 登录节点。

## 当前可用链路

- 跳板机：`ubuntu@43.134.118.168`
- 跳板机上的校园代理：`127.0.0.1:1080`
- 当前已验证可达：
  - `hpc4login.hpc.hkust-gz.edu.cn:22`
  - `hpc2login.hpc.hkust-gz.edu.cn:22`

## 为什么不能直接用本地 `ssh -D`

下面这种命令：

```bash
ssh -D 127.0.0.1:1080 -C -N -f ubuntu@43.134.118.168
```

只是在本地暴露“跳板机自身网络”的 SOCKS5，不会自动转发跳板机上的 EasyConnect `127.0.0.1:1080`。

当前稳定可用的方式是，让跳板机代替本机执行：

```bash
nc -x 127.0.0.1:1080 -X 5 %h %p
```

## 推荐 SSH 入口

已提供专用 SSH config：

```bash
/root/.codex/skills/hpc-login-ssh/references/ssh_config_hpc_via_remote_vpn
```

以及 wrapper：

```bash
/root/.codex/skills/hpc-login-ssh/scripts/ssh_hpc_via_remote_vpn.sh
```

## 使用方式

### 登录 HPC2

```bash
bash /root/.codex/skills/hpc-login-ssh/scripts/ssh_hpc_via_remote_vpn.sh hpc2-vpn
```

### 非交互验证

```bash
ssh -F /root/.codex/skills/hpc-login-ssh/references/ssh_config_hpc_via_remote_vpn \
  hpc2-vpn 'hostname; whoami; pwd'
```

## 关于 HPC4

当前测试结论是：

- `hpc4login` 通过远端 EasyConnect 的网络层是可达的
- 但这台本机通过本地 key 直登 `hpc4` 的 alias 方式未验证通过

因此当前推荐：

- `hpc2`：使用这里的 `hpc2-vpn` alias
- `hpc4`：继续使用远端脚本

```bash
ssh ubuntu@43.134.118.168 \
  'bash /home/ubuntu/.github/skills/hpc4-end-to-end-access/scripts/hpc4_login_via_vpn.sh "hostname; whoami; pwd"'
```

## 用于迁移脚本

当需要把 baseline 迁到 HPC2 时，可以把这个 SSH config 传给：

```bash
bash scripts/migrate_baseline_minimal.sh \
  --dst-host hpc2-vpn \
  --dst-real /hpc2hdd/home/yhao481/jhupload/proj/baseline \
  --ssh-config /root/.codex/skills/hpc-login-ssh/references/ssh_config_hpc_via_remote_vpn
```
