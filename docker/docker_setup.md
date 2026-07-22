# Docker Setup

## Overview

Docker Engine runs natively inside WSL2 (Ubuntu 20.04), managed by systemd. There is no Docker Desktop involved. VS Code on Windows connects to the container via Remote-SSH.

---

## WSL2 / Docker Configuration

### Systemd enabled
`/etc/wsl.conf` has `systemd=true` so systemd runs as PID 1 inside WSL2. Docker is managed as a systemd service via socket activation — the daemon starts on first `docker` command.

### Docker group
The `youruser` user is in the `docker` group, so `docker` commands work without `sudo`.

### Auto-start in `.bashrc`
```bash
export DOCKER_HOST=unix:///var/run/docker.sock

if ! service docker status > /dev/null 2>&1; then
    sudo service docker start > /dev/null 2>&1
fi
```

### Docker credential store
`~/.docker/config.json` has `"credStore": ""` to clear a stale Docker Desktop credential store that would otherwise block the daemon.

---

## The Image

Built from `docker/Dockerfile` and tagged `tf-512-gpu`. Stack:

- Base: `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04`
- Python 3.12 (system, no venv)
- PyTorch 2.4.1 + CUDA 12.4
- flash-attn 2.6.3 (compiled with `--no-build-isolation`)
- transformers 5.13.0, datasets 5.0.0, accelerate, peft, trl, deepspeed, bitsandbytes
- GPU architectures: sm_61 (MX250), sm_80 (A100), sm_90 (H100)
- Linux tools: tmux, vim, htop, nvtop, gpustat, adb, pdsh
- InfiniBand/RDMA libraries for multi-node training
- OpenSSH server (for VS Code attachment)
- Arduino CLI 1.5.1 + AVR core 1.8.8 + Adafruit NeoPixel/LedControl libraries (compiles/verifies `arduino/led_board_controller/firmware/`)

To rebuild:
```bash
./docker/build_docker.sh
```

---

## Starting the Container

```bash
./docker/run_docker.sh
```

This script:
1. Removes any stale container named `training`
2. Starts a new detached container with:
   - GPU access (`--gpus all`)
   - `/workspace` → project directory
   - `~/.cache/huggingface` → persistent model cache
   - `~/.docker-persist/<container-name>/vscode-server` → `/root/.vscode-server`
   - `~/.docker-persist/<container-name>/claude` → `/root/.claude`
   - `~/.docker-persist/<container-name>/claude.json` → `/root/.claude.json`
   - `--shm-size=16g` for PyTorch DataLoader workers
   - Port `2222:22` for SSH
   - Ports `6006` (TensorBoard) and `8888` (Jupyter)
3. Injects SSH public keys from both WSL (`~/.ssh/`) and Windows (`/c/Users/youruser/.ssh/`) into `/root/.ssh/authorized_keys`

### Why the VS Code extension and Claude login used to reset every run

`run_docker.sh` does `docker rm -f` before every `docker run` — the container's writable
layer (and anything living only in it) is destroyed each time. VS Code Server installs
itself into `/root/.vscode-server` on first Remote-SSH connect, and the Claude Code
extension along with it; Claude's login/session state lives in `/root/.claude` and
`/root/.claude.json`. None of those were mounted, so both got wiped on every container
recreation, forcing a fresh extension install and re-login each time.

The bind mounts above put that state on the WSL host filesystem instead, so it survives
`docker rm -f` and the next container start picks up right where the last one left off.
The very first run after adding these mounts will still do a normal one-time VS Code
Server install + Claude login; every run after that should skip both.

---

## SSH Keys

Two keys are authorized inside the container:

| Key | Location | Used by |
|-----|----------|---------|
| `id_ed25519` (WSL) | `~/.ssh/id_ed25519` | WSL terminal SSH |
| `id_ed25519` (Windows) | `C:\Users\youruser\.ssh\id_ed25519` | VS Code Remote-SSH |

These are **different key pairs** — both are injected on container start by `docker/run_docker.sh`.

The Windows SSH config at `C:\Users\youruser\.ssh\config` has:
```
Host training-container
    HostName localhost
    Port 2222
    User root
    IdentityFile C:\Users\youruser\.ssh\id_ed25519
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
```

---

## Connecting from VS Code

**Prerequisite**: Install the [Remote - SSH](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh) extension.

1. Start the container: `./docker/run_docker.sh`
2. In VS Code: `Ctrl+Shift+P` → **Remote-SSH: Connect to Host** → **training-container**
3. VS Code installs its server inside the container on first connect (~1 min)
4. Open folder: `/workspace` (your mounted project directory)

### Reconnecting after a container restart

Just run `./docker/run_docker.sh` again and reconnect via Remote-SSH. The container is stateless — only `/workspace` and `~/.cache/huggingface` persist (via volume mounts).

---

## Connecting from the WSL terminal

```bash
ssh -p 2222 root@localhost
# or via alias defined in ~/.ssh/config:
ssh training-container
```