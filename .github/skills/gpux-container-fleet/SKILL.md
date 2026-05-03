---
name: gpux-container-fleet
description: "Use when: adding or removing HPC2 GPU dev containers in this project's dynamic fleet, registering a new container into ~/.config/gpux/hosts.tsv, verifying SSH/GPU connectivity with the gpux CLI, fixing a fresh container that lacks tmux (sudo apt install tmux), updating the CODI training daemon HOSTS array and queue.tsv after fleet changes, diagnosing 'Permission denied' from a deprecated container, or rebalancing queued training tasks across newly added or freed containers. Covers gpux subcommands (ls/ssh/run/runc/runall/runallc/par/parc/gpu/push/pull/tmux) and the daemon at CODI/qwentrain/daemon/."
---

# GPUX Container Fleet Management

The CODI training fleet on HPC2 is a **dynamic queue of GPU dev containers**: containers
appear and disappear without warning (manual top-ups by the user, expirations, OOM kills).
This skill is the playbook for keeping `gpux`, the daemon, and the queue in sync with
whatever containers are actually alive **right now**.

## When To Use

- User says "新加了一个容器 / new container" and pastes `ssh <user>@<ip> -p <port>` + password
- User says "这个容器没了 / 容器挂了 / Permission denied" → remove a dead container
- A queued task isn't launching, or you need to confirm the daemon sees the right hosts
- Fresh container reports `tmux: command not found` on first launch
- Need to rebalance `queue.tsv` after a container frees up or comes online

## Fleet Topology (current convention)

| Item | Value |
|------|-------|
| All containers reachable at | `10.120.18.240:6988` (only the user differs) |
| Shared filesystem root | `/hpc2hdd/home/yhao481/` (project at `…/jhupload/proj/baseline`) |
| Hosts file (chmod 600, gitignored) | `~/.config/gpux/hosts.tsv` |
| Format | `alias\|user\|host\|port\|password` (pipe-separated) |
| Alias convention | `hpc2-a800-<N>` where N is the last digit of the container user (`a800_4_5` → `hpc2-a800-5`) |
| gpux binary | `/hpc2hdd/home/yhao481/.local/bin/gpux` |
| Daemon | `CODI/qwentrain/daemon/daemon.sh` (HOSTS array at line ~28, INTERVAL=3600s, CONFIRM_DELAY=300s) |
| Queue | `CODI/qwentrain/daemon/queue.tsv` (`alias\ttask_tag`) |
| Daemon tmux session | `codi-daemon` |

## Workflow A — Add a new container

Input from user: `ssh <user>@10.120.18.240 -p 6988` and a password.

1. **Pick alias.** Use `hpc2-a800-<N>` matching the user's trailing digit (e.g. `a800_4_1` → `hpc2-a800-1`). If colliding, fall back to next free number.
2. **Backup, then append to hosts file:**
   ```bash
   cp ~/.config/gpux/hosts.tsv ~/.config/gpux/hosts.tsv.bak.$(date +%s)
   echo 'hpc2-a800-N|<user>|10.120.18.240|6988|<password>' >> ~/.config/gpux/hosts.tsv
   chmod 600 ~/.config/gpux/hosts.tsv
   ```
   No restart needed — `gpux` re-reads the file on every call.
3. **Smoke-test SSH + GPU visibility:**
   ```bash
   gpux hpc2-a800-N 'hostname; nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader'
   ```
   Expect 4 rows (4×A800). If `Permission denied` → password typo; if `Connection refused` → wrong port/IP.
4. **Fix tmux if missing** (very common on a fresh container):
   ```bash
   gpux hpc2-a800-N 'command -v tmux || sudo apt install -y tmux'
   ```
   The daemon uses `tmux new-session -d` to launch jobs; without it, every launch silently fails.
5. **Register with the daemon.** Edit [CODI/qwentrain/daemon/daemon.sh](../../../CODI/qwentrain/daemon/daemon.sh) line ~28:
   ```bash
   HOSTS=(hpc2-a800-1 hpc2-a800-2 hpc2-a800-3 hpc2-a800-5 hpc2-a800-N)
   ```
6. **Restart daemon** (queue snapshot is reread on restart):
   ```bash
   tmux kill-session -t codi-daemon 2>/dev/null
   cd CODI/qwentrain/daemon && bash start.sh
   tail -n 30 daemon.log
   ```
7. **(Optional) Rebalance** queued tasks onto the new host by editing `queue.tsv` (`alias\ttask_tag`, one per line) before the daemon's next sweep.

## Workflow B — Remove a dead/deprecated container

Trigger: SSH returns `Permission denied (password)` repeatedly, or user explicitly says container is gone.

1. **Backup and delete the line:**
   ```bash
   cp ~/.config/gpux/hosts.tsv ~/.config/gpux/hosts.tsv.bak.$(date +%s)
   sed -i '/^hpc2-a800-N|/d' ~/.config/gpux/hosts.tsv
   ```
2. **Remove from daemon HOSTS array** (same line ~28 in `daemon.sh`).
3. **Rescue any of its in-flight queue items.** If `queue.tsv` still has tasks pinned to the dead alias, reassign them to a live host before restarting.
4. **Restart daemon** (same commands as A.6).
5. **Verify:** `gpux ls` should no longer show the alias; `gpux gpu` should succeed across all remaining hosts.

## gpux Subcommand Cheatsheet

```text
gpux ls                              # list aliases (no GPU query, fast)
gpux gpu                             # nvidia-smi across every host (fleet-wide health)
gpux <alias> '<cmd>'                 # implicit forward (shortest form)
gpux ssh <alias>                     # interactive shell
gpux run  <alias> '<cmd>'            # explicit non-interactive
gpux runc <alias> '<cmd>'            # same + auto cd to local $PWD (shared FS)
gpux runall  '<cmd>'                 # serial fan-out to ALL hosts
gpux runallc '<cmd>'                 # serial fan-out + cd to $PWD
gpux par  '<cmd>'                    # parallel fan-out (needs GNU parallel)
gpux parc '<cmd>'                    # parallel + cd to $PWD
gpux push <local> [remote]           # rsync push (one host)
gpux pull <alias> <remote> <local>   # rsync pull
gpux tmux                            # one tmux pane per host, synchronized input
```

Useful one-liners:
```bash
# Who's actually training right now?
gpux par 'pgrep -af "torchrun.*train.py" | head -1'

# Free GPU memory across the fleet
gpux par 'nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | paste -sd, -'

# Tail a remote log without copying it
gpux hpc2-a800-5 'tail -n 50 -f CODI/qwentrain/nohup_logs/gpt2.codi.log'
```

## Daemon Sync Checklist

Whenever the fleet changes, verify all four sources of truth agree:

| Source | What it controls | How to inspect |
|--------|-----------------|----------------|
| `~/.config/gpux/hosts.tsv` | which aliases `gpux` can dial | `gpux ls` |
| `CODI/qwentrain/daemon/daemon.sh` `HOSTS=(…)` | which aliases the daemon polls | `grep -n '^  HOSTS=' …/daemon.sh` |
| `CODI/qwentrain/daemon/queue.tsv` | task→alias pinning | `cat …/queue.tsv` |
| Live container | actually reachable, has tmux | `gpux <alias> 'command -v tmux && hostname'` |

## Known Gotchas

- **First-boot tmux missing.** Containers ship without tmux. If the daemon "launches" a task but the job never appears in `pgrep -af torchrun`, this is almost always why. Fix: `gpux <alias> 'sudo apt install -y tmux'`.
- **Same IP/port for every container.** Don't try to disambiguate by host — the user field is the only unique key.
- **Daemon pops queue before validating remote launch.** If the remote `tmux` command fails (missing tmux, bad path, rc=127), the queue entry is lost. Always keep `queue.tsv.bak.*` and watch `nohup_logs/<tag>.log` for `rc=` shortly after a sweep.
- **`hosts.tsv` is gitignored and chmod 600.** Never paste its contents into commits, PRs, or chat messages without redacting the password column.
- **`gpux ssh <alias>` with no command opens an interactive TTY.** Inside scripts always pass an explicit command (or use `run`/`runc`).
- **Shared FS shortcut.** Because all containers mount the same `/hpc2hdd/home/yhao481/`, prefer `runc`/`runallc`/`parc` so the remote shell lands in the exact same `$PWD` you're standing in locally — no `cd` boilerplate required.

## Done-When

- `gpux ls` shows exactly the live containers (no ghosts).
- `gpux gpu` returns 4×A800 rows for every alias.
- `daemon.sh` `HOSTS=(…)` matches `gpux ls` aliases.
- `queue.tsv` only references aliases that exist.
- `tail -n 20 CODI/qwentrain/daemon/daemon.log` shows the most recent sweep enumerating the new fleet.
