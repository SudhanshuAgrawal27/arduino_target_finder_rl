#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $(basename "$0") [container-name]"
    echo "  container-name  default: training"
    echo ""
    echo "Reattaches the Arduino's USB-serial device (CH340, VID:PID 1a86:7523)"
    echo "to WSL via usbipd and re-creates its device node inside a running"
    echo "container, without needing to recreate the container itself."
    exit 1
}

[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && usage

CONTAINER_NAME="${1:-training}"
VID_PID="1a86:7523"  # CH340 USB-serial adapter
DEV="/dev/ttyUSB0"

echo "Looking up BUSID for $VID_PID via usbipd..."
BUSID="$(usbipd.exe list | awk -v vidpid="$VID_PID" '$2 == vidpid {print $1}')"

if [[ -z "$BUSID" ]]; then
    echo "Could not find a device with VID:PID $VID_PID in 'usbipd.exe list'." >&2
    echo "Is the board plugged in? Run 'usbipd.exe list' to check." >&2
    exit 1
fi

echo "Found busid $BUSID. Attaching to WSL..."
usbipd.exe attach --wsl --busid "$BUSID" || echo "(attach reported an issue — continuing, it may already be attached)"

echo "Waiting for $DEV to appear on the WSL host..."
for _ in $(seq 1 10); do
    [[ -e "$DEV" ]] && break
    sleep 0.5
done

if [[ ! -e "$DEV" ]]; then
    echo "$DEV did not appear on the WSL host after attach." >&2
    exit 1
fi

read -r MAJOR MINOR < <(ls -l "$DEV" | awk '{gsub(",", ""); print $5, $6}')
echo "Host device $DEV -> major $MAJOR, minor $MINOR"

echo "Creating device node in container '$CONTAINER_NAME'..."
docker exec "$CONTAINER_NAME" bash -c "rm -f $DEV && mknod $DEV c $MAJOR $MINOR && chmod 666 $DEV"

echo ""
echo "Done. $DEV is available inside '$CONTAINER_NAME'."
