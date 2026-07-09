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

# Collect all SSH public keys (WSL + Windows) to inject into the container
AUTHORIZED_KEYS=""
for f in "$HOME/.ssh/id_ed25519.pub" "$HOME/.ssh/id_rsa.pub" \
          "/c/Users/youruser/.ssh/id_ed25519.pub" "/c/Users/youruser/.ssh/id_rsa.pub"; do
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
upsert_ssh_config "/c/Users/youruser/.ssh/config"         'C:\Users\youruser\.ssh\id_ed25519'

echo ""
echo "Container '$CONTAINER_NAME' started."
echo "  Workspace : /workspace → $WORKSPACE"
echo "  SSH port  : $SSH_PORT"
echo ""
echo "Connect: VS Code → Remote-SSH: Connect to Host → $CONTAINER_NAME"