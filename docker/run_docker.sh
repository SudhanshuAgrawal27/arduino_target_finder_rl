#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $(basename "$0") [container-name] [ssh-port]"
    echo "  container-name  default: training"
    echo "  ssh-port        default: 2222"
    exit 1
}

[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && usage

CONTAINER_NAME="${1:-training}"
SSH_PORT="${2:-2222}"
IMAGE="tf-512-gpu"
WORKSPACE="$(cd "$(dirname "$0")/.." && pwd)"

# Windows username (may differ from the WSL user), used to locate the
# Windows-side SSH keys/config under /c/Users/<win_user>.
WIN_USER="$(cmd.exe /c "echo %USERNAME%" 2>/dev/null | tr -d '\r\n')"
WIN_USER="${WIN_USER:-$USER}"
WIN_HOME="/c/Users/$WIN_USER"

# Collect all SSH public keys (WSL + Windows) to inject into the container
AUTHORIZED_KEYS=""
for f in "$HOME/.ssh/id_ed25519.pub" "$HOME/.ssh/id_rsa.pub" \
          "$WIN_HOME/.ssh/id_ed25519.pub" "$WIN_HOME/.ssh/id_rsa.pub"; do
    [[ -f "$f" ]] && AUTHORIZED_KEYS+="$(cat "$f")"$'\n'
done
AUTHORIZED_KEYS="$(echo "$AUTHORIZED_KEYS" | sort -u | grep -v '^$')"

if [[ -z "$AUTHORIZED_KEYS" ]]; then
    echo "No SSH key found — generating ~/.ssh/id_ed25519 ..."
    ssh-keygen -t ed25519 -f "$HOME/.ssh/id_ed25519" -N ""
    AUTHORIZED_KEYS="$(cat "$HOME/.ssh/id_ed25519.pub")"
fi

# Persistent state that must survive `docker rm -f` + recreate: VS Code server
# (so the Claude Code extension isn't reinstalled every time) and Claude's own
# config/credentials (so it doesn't force a re-login every time).
PERSIST_DIR="$HOME/.docker-persist/$CONTAINER_NAME"
mkdir -p "$PERSIST_DIR/vscode-server" "$PERSIST_DIR/claude"
touch "$PERSIST_DIR/claude.json"

# Remove stale container if it exists
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

# Pass the Arduino's serial device through if it's attached to WSL (via
# usbipd-win) at container-start time. Absent means plain training runs
# still work without any Arduino connected.
DEVICE_ARGS=()
for dev in /dev/ttyUSB0 /dev/ttyACM0; do
    [[ -e "$dev" ]] && DEVICE_ARGS+=(--device="$dev")
done

# Wildcard cgroup rule for USB-serial (major 188, ttyUSB*) and USB CDC-ACM
# (major 166, ttyACM*) devices. This lets a device be reattached later
# (after a driver reinstall / usbipd re-attach) without recreating the
# container: just `usbipd attach` on the host, then inside the container
#   mknod /dev/ttyUSB0 c 188 0 && chmod 666 /dev/ttyUSB0
# (adjust the minor number to match `ls -l /dev/ttyUSB0` on the host).
DEVICE_CGROUP_ARGS=(
    --device-cgroup-rule='c 188:* rmw'
    --device-cgroup-rule='c 166:* rmw'
)

docker run -d \
    --name "$CONTAINER_NAME" \
    --gpus all \
    -v "$WORKSPACE":/workspace \
    -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
    -v "$PERSIST_DIR/vscode-server":/root/.vscode-server \
    -v "$PERSIST_DIR/claude":/root/.claude \
    -v "$PERSIST_DIR/claude.json":/root/.claude.json \
    --shm-size=16g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -p "$SSH_PORT":22 \
    "${DEVICE_ARGS[@]}" \
    "${DEVICE_CGROUP_ARGS[@]}" \
    "$IMAGE" \
    bash -c "/usr/sbin/sshd && sleep infinity"

# Inject all public keys into container
docker exec "$CONTAINER_NAME" bash -c "
    mkdir -p /root/.ssh
    cat >> /root/.ssh/authorized_keys << 'KEYS'
$AUTHORIZED_KEYS
KEYS
    sort -u /root/.ssh/authorized_keys -o /root/.ssh/authorized_keys
    chmod 700 /root/.ssh
    chmod 600 /root/.ssh/authorized_keys
"

# Update SSH config for a given file, upserting the Host block
upsert_ssh_config() {
    local config_file="$1"
    local id_file="$2"
    mkdir -p "$(dirname "$config_file")"
    touch "$config_file"

    # Remove existing block for this container name
    local tmp
    tmp="$(awk -v host="$CONTAINER_NAME" '
        /^Host / { skip = ($2 == host) }
        !skip { print }
    ' "$config_file")"
    printf '%s\n' "$tmp" > "$config_file"

    # Append updated block
    cat >> "$config_file" << EOF

Host $CONTAINER_NAME
    HostName localhost
    Port $SSH_PORT
    User root
    IdentityFile $id_file
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
EOF
}

upsert_ssh_config "$HOME/.ssh/config"                  "~/.ssh/id_ed25519"
upsert_ssh_config "$WIN_HOME/.ssh/config"              "C:\\Users\\$WIN_USER\\.ssh\\id_ed25519"

echo ""
echo "Container '$CONTAINER_NAME' started."
echo "  Workspace : /workspace → $WORKSPACE"
echo "  SSH port  : $SSH_PORT"
echo ""
echo "Connect: VS Code → Remote-SSH: Connect to Host → $CONTAINER_NAME"
echo ""
echo "If the Arduino's USB-serial device drops later (driver reinstall,"
echo "unplug/replug), you don't need to recreate this container — from the"
echo "host WSL terminal run:"
echo "  bash docker/reconnect_usb.sh $CONTAINER_NAME"