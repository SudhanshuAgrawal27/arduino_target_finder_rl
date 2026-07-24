"""Sweeps a white LED across the 8x8 neighborhood around a target point on
the 16x16 WS2812B board, and records the photoresistor's response at each
position as a heatmap.

At each cell: LED off, wait, read a baseline; LED on (full white), wait
again, read again; record (on - baseline). The off phase isn't just a pause
-- without it, each reading would start from wherever the LDR happened to
settle after the *previous* cell rather than a known baseline, and since
the LDR's decay (getting darker) is much slower than its rise (getting
brighter), a neighboring cell's residual brightness could bleed into the
next reading and bias the sweep. Subtracting the baseline also cancels out
slow ambient-light drift over the sweep's ~1-2 minute runtime. Both phases
reuse the same wait -- see --linger-seconds.

Standalone hardware calibration test -- not tied to any RL checkpoint/env,
since it's characterizing the LDR circuit's response to LED position, not
running a policy. Assumes the same read_ldr() interface eval_ldr_test.py
uses; once the custom LDR circuit exists, point led_board_client at it
(or swap the import) as long as that function signature is kept.

Usage:
    python3 eval_ldr_sweep.py
    python3 eval_ldr_sweep.py --target 10 3 --linger-seconds 2

    # Move the LEDs through the sweep but skip reading the LDR entirely --
    # every recorded value is 0. For checking the sweep pattern itself
    # (e.g. before the LDR circuit is wired up) without touching the sensor.
    python3 eval_ldr_sweep.py --dry-run
"""

import argparse
import time

import matplotlib.pyplot as plt

from arduino.led_board_controller.led_board_client import clear, connect, read_ldr, set_dynamic_layer

GRID_SIZE = 16
WINDOW_SIZE = 8


def compute_window_origin(target, window_size, grid_size):
    """Top-left corner of the window_size x window_size window that best
    centers `target`, clamped so the window never runs off the board."""
    tx, ty = target
    ox = min(max(tx - window_size // 2, 0), grid_size - window_size)
    oy = min(max(ty - window_size // 2, 0), grid_size - window_size)
    return ox, oy


def sweep(ser, origin, window_size, linger_seconds, dry_run):
    ox, oy = origin
    readings = {}
    for dy in range(window_size):
        for dx in range(window_size):
            x, y = ox + dx, oy + dy

            clear(ser)
            time.sleep(linger_seconds)
            baseline = 0 if dry_run else read_ldr(ser)

            # The agent layer (unlike the raw "P:" pixel command) is retained
            # firmware state that ledBoardTick()'s ~33Hz autonomous render
            # redraws unconditionally every cycle -- see ws2812b_matrix_driver.cpp's
            # render(). A plain set_pixel_color() pixel isn't tracked in any
            # retained state, so the very next tick's strip.clear() erases it
            # within ~15ms instead of holding for linger_seconds.
            set_dynamic_layer(ser, (x, y), [])
            time.sleep(linger_seconds)
            on_value = 0 if dry_run else read_ldr(ser)

            delta = on_value - baseline
            readings[(x, y)] = delta
            print(f"({x},{y}) -> baseline={baseline} on={on_value} delta={delta}")
    return readings


def plot_readings(readings, origin, window_size, target):
    ox, oy = origin
    grid = [[readings[(ox + dx, oy + dy)] for dx in range(window_size)] for dy in range(window_size)]

    fig, ax = plt.subplots()
    im = ax.imshow(grid, cmap="viridis")
    ax.set_xticks(range(window_size), [ox + dx for dx in range(window_size)])
    ax.set_yticks(range(window_size), [oy + dy for dy in range(window_size)])
    ax.set_xlabel("x (board coords)")
    ax.set_ylabel("y (board coords)")
    ax.set_title(f"LDR response vs. LED position (target={target})")
    fig.colorbar(im, ax=ax, label="LDR delta: on - baseline (raw ADC)")

    tx, ty = target
    if ox <= tx < ox + window_size and oy <= ty < oy + window_size:
        ax.plot(tx - ox, ty - oy, marker="x", color="red", markersize=12, markeredgewidth=2)

    fig.savefig("ldr_sweep_plot.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, nargs=2, default=(8, 8), metavar=("X", "Y"),
                         help="board coordinates the 8x8 sweep window is centered on")
    parser.add_argument("--linger-seconds", type=float, default=1.5,
                         help="how long to wait before each read -- reused for both the LED-off "
                              "baseline settle and the LED-on settle")
    parser.add_argument("--dry-run", action="store_true",
                         help="move the LEDs but skip reading the LDR -- records 0 for both "
                              "baseline and on, so every delta is 0")
    args = parser.parse_args()

    target = tuple(args.target)
    origin = compute_window_origin(target, WINDOW_SIZE, GRID_SIZE)

    ser = connect()
    clear(ser)
    readings = sweep(ser, origin, WINDOW_SIZE, args.linger_seconds, args.dry_run)
    clear(ser)
    ser.close()

    plot_readings(readings, origin, WINDOW_SIZE, target)


if __name__ == "__main__":
    main()
