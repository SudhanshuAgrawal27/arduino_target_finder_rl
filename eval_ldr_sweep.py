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

    # Calibration mode: instead of a per-cell heatmap, characterize how the
    # LDR delta falls off with Manhattan distance from the target, and save
    # that curve to a JSON file for eval_demo_16-16-ldr-feedback.py to turn
    # live readings into the same discrete proximity levels the policy was
    # trained on (see simulator.py's proximity_score). One baseline read for
    # the whole run (not per-cell -- ambient light is assumed stable across
    # a ~1-2 minute calibration), repeated 3x and averaged per distance
    # bucket to smooth out single-run noise.
    python3 eval_ldr_sweep.py --calibrate
"""

import argparse
import json
import time
from collections import defaultdict

import matplotlib.pyplot as plt

from arduino.led_board_controller.led_board_client import clear, connect, read_ldr, set_dynamic_layer
from simulator import proximity_score

GRID_SIZE = 16
WINDOW_SIZE = 8
DEFAULT_CALIBRATION_RUNS = 3
DEFAULT_CALIBRATION_FILE = "ldr_calibration.json"


def compute_window_origin(target, window_size, grid_size):
    """Top-left corner of the window_size x window_size window that best
    centers `target`, clamped so the window never runs off the board."""
    tx, ty = target
    ox = min(max(tx - window_size // 2, 0), grid_size - window_size)
    oy = min(max(ty - window_size // 2, 0), grid_size - window_size)
    return ox, oy


def sweep(ser, origin, window_size, linger_seconds, dry_run, max_attempts=3):
    ox, oy = origin
    readings = {}
    for dy in range(window_size):
        for dx in range(window_size):
            x, y = ox + dx, oy + dy

            for attempt in range(max_attempts):
                reply = clear(ser)
                if reply != "OK":
                    print(f"({x},{y}) attempt {attempt + 1}/{max_attempts} -- LED board "
                          f"rejected clear ({reply!r}), retrying")
                    continue
                time.sleep(linger_seconds)
                baseline = 0 if dry_run else read_ldr(ser)

                # The agent layer (unlike the raw "P:" pixel command) is retained
                # firmware state that ledBoardTick()'s ~33Hz autonomous render
                # redraws unconditionally every cycle -- see ws2812b_matrix_driver.cpp's
                # render(). A plain set_pixel_color() pixel isn't tracked in any
                # retained state, so the very next tick's strip.clear() erases it
                # within ~15ms instead of holding for linger_seconds.
                reply = set_dynamic_layer(ser, (x, y), [])
                if reply != "OK":
                    # The LED never actually moved to (x, y) -- still sitting
                    # wherever clear() (or the previous cell) left it -- so
                    # this attempt's reading would be meaningless. Retry the
                    # whole off/on cycle for this cell rather than trusting it.
                    print(f"({x},{y}) attempt {attempt + 1}/{max_attempts} -- LED board "
                          f"rejected dynamic layer ({reply!r}), retrying")
                    continue
                time.sleep(linger_seconds)
                on_value = 0 if dry_run else read_ldr(ser)
                break
            else:
                print(f"({x},{y}) -- giving up after {max_attempts} attempts, recording delta=0")
                baseline = on_value = 0

            delta = on_value - baseline
            readings[(x, y)] = delta
            print(f"({x},{y}) -> baseline={baseline} on={on_value} delta={delta}")
    return readings


def manhattan_offset(cell, target):
    x, y = cell
    tx, ty = target
    return abs(x - tx) + abs(y - ty)


def bucket_for_offset(offset, score_radius):
    """Which calibration bucket a Manhattan offset falls into: its own
    distance (0..score_radius), or "background" for everything past the
    radius -- those all map to the same trained score (0.0, see
    proximity_score), so pooling them gives a much larger, more robust
    sample than any single offset's 4*d ring cells would alone."""
    return str(offset) if offset <= score_radius else "background"


def reject_outliers(values, threshold=3.5):
    """Drops values whose modified z-score -- based on median absolute
    deviation, which (unlike mean/stddev) a single wild outlier can't drag
    around -- exceeds `threshold` (3.5 is the commonly recommended cutoff,
    Iglewicz & Hoaglin). Below 4 samples MAD is too noisy to trust, so
    everything is kept; same if MAD itself is 0 (every value already
    identical, nothing to reject against). Never drops every value -- an
    all-outlier bucket means something upstream is wrong, not something to
    silently average away to nothing."""
    values = list(values)
    if len(values) < 4:
        return values

    sorted_values = sorted(values)
    n = len(sorted_values)
    median = (sorted_values[n // 2] if n % 2 else
              (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2)
    abs_devs = sorted(abs(v - median) for v in values)
    mad = (abs_devs[n // 2] if n % 2 else
           (abs_devs[n // 2 - 1] + abs_devs[n // 2]) / 2)
    if mad == 0:
        return values

    kept = [v for v in values if 0.6745 * abs(v - median) / mad <= threshold]
    return kept or values


def calibrate(ser, origin, window_size, target, linger_seconds, num_runs, score_radius, dry_run):
    """Characterizes LDR delta vs. Manhattan distance from `target`, using
    ONE baseline (LED off, no agent) for the whole run -- unlike sweep()'s
    per-cell baseline, which exists to stop neighboring cells' heatmap
    readings from bleeding into each other. Here every cell's delta is
    relative to the same fixed reference, which is what a live game's
    single per-episode baseline (see eval_demo_16-16-ldr-feedback.py) will
    also compare against, so the calibration curve has to be measured the
    same way to be a valid reference for it.

    Repeats the full window `num_runs` times and pools every reading that
    shares a bucket_for_offset() bucket (across cells AND runs) before
    averaging, rather than trusting any single run/cell -- smooths out the
    kind of single-reading noise that could otherwise produce a
    non-monotonic curve (e.g. offset 1 reading dimmer than offset 2).
    Outliers within each pooled bucket are dropped (see reject_outliers)
    before averaging, since even a genuinely-delivered "OK" reply doesn't
    guarantee the LDR reading itself wasn't a one-off glitch.

    Between runs (not within one -- that would defeat the point of a
    single fixed baseline), the serial connection is closed and reopened
    rather than just re-clearing LED state. Reopening the port re-triggers
    the Arduino's auto-reset, which flushes both the OS-level pyserial
    receive buffer and the board's own hardware serial buffer/firmware
    state -- guarding against whatever accumulates over ~200 commands that
    a same-connection clear() wouldn't touch."""
    ox, oy = origin

    reply = clear(ser)
    if reply != "OK":
        raise RuntimeError(f"LED board rejected clear before baseline: {reply!r}")
    time.sleep(linger_seconds)
    baseline = 0 if dry_run else read_ldr(ser)

    deltas_by_bucket = defaultdict(list)
    deltas_by_cell = defaultdict(list)
    for run in range(num_runs):
        if run > 0:
            print(f"reconnecting before run {run + 1}/{num_runs} to reset the board / flush serial buffers")
            ser.close()
            ser = connect()
            reply = clear(ser)
            if reply != "OK":
                raise RuntimeError(f"LED board rejected clear after reconnect: {reply!r}")

        for dy in range(window_size):
            for dx in range(window_size):
                x, y = ox + dx, oy + dy

                # set_dynamic_layer already retries internally (see
                # led_board_client._send), but if it still comes back
                # non-OK the LED never actually moved to (x, y) -- it's
                # still sitting wherever the previous cell left it. Reading
                # the LDR anyway would silently attribute a stale position's
                # brightness to this cell, so skip the sample instead of
                # trusting it.
                reply = set_dynamic_layer(ser, (x, y), [])
                if reply != "OK":
                    print(f"run {run + 1}/{num_runs} ({x},{y}) -- LED board rejected dynamic layer "
                          f"({reply!r}) after retries, skipping this reading")
                    continue

                time.sleep(linger_seconds)
                on_value = 0 if dry_run else read_ldr(ser)
                delta = on_value - baseline

                offset = manhattan_offset((x, y), target)
                bucket = bucket_for_offset(offset, score_radius)
                deltas_by_bucket[bucket].append(delta)
                deltas_by_cell[(x, y)].append(delta)
                print(f"run {run + 1}/{num_runs} ({x},{y}) offset={offset} bucket={bucket} delta={delta}")

    reply = clear(ser)
    if reply != "OK":
        print(f"warning: LED board rejected final clear ({reply!r}) -- board may still show the last probe")

    levels = {}
    for bucket, values in deltas_by_bucket.items():
        kept = reject_outliers(values)
        if len(kept) < len(values):
            print(f"bucket {bucket}: dropped {len(values) - len(kept)} outlier reading(s) "
                  f"out of {len(values)} before averaging")
        distance = score_radius + 1 if bucket == "background" else int(bucket)
        levels[bucket] = {
            "score": proximity_score(distance, score_radius),
            "avg_delta": sum(kept) / len(kept),
            "n_samples": len(kept),
        }

    calibration = {
        "target": list(target),
        "window_origin": [ox, oy],
        "window_size": window_size,
        "grid_size": GRID_SIZE,
        "score_radius": score_radius,
        "num_runs": num_runs,
        "linger_seconds": linger_seconds,
        "baseline": baseline,
        "levels": levels,
    }
    avg_readings = {}
    for cell, values in deltas_by_cell.items():
        kept = reject_outliers(values)
        avg_readings[cell] = sum(kept) / len(kept)
    return calibration, avg_readings, ser


def plot_readings(readings, origin, window_size, target, title, out_file):
    ox, oy = origin
    grid = [[readings[(ox + dx, oy + dy)] for dx in range(window_size)] for dy in range(window_size)]

    fig, ax = plt.subplots()
    im = ax.imshow(grid, cmap="viridis")
    ax.set_xticks(range(window_size), [ox + dx for dx in range(window_size)])
    ax.set_yticks(range(window_size), [oy + dy for dy in range(window_size)])
    ax.set_xlabel("x (board coords)")
    ax.set_ylabel("y (board coords)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="LDR delta: on - baseline (raw ADC)")

    tx, ty = target
    if ox <= tx < ox + window_size and oy <= ty < oy + window_size:
        ax.plot(tx - ox, ty - oy, marker="x", color="red", markersize=12, markeredgewidth=2)

    fig.savefig(out_file)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, nargs=2, default=(8, 8), metavar=("X", "Y"),
                         help="board coordinates the 8x8 sweep window is centered on")
    parser.add_argument("--linger-seconds", type=float, default=0.25,
                         help="how long to wait before each read -- reused for both the LED-off "
                              "baseline settle and the LED-on settle")
    parser.add_argument("--dry-run", action="store_true",
                         help="move the LEDs but skip reading the LDR -- records 0 for both "
                              "baseline and on, so every delta is 0")
    parser.add_argument("--calibrate", action="store_true",
                         help="instead of a per-cell heatmap sweep, characterize LDR delta vs. "
                              "Manhattan distance from --target and save it to --calibration-file "
                              "for eval_demo_16-16-ldr-feedback.py")
    parser.add_argument("--calibration-runs", type=int, default=DEFAULT_CALIBRATION_RUNS,
                         help="--calibrate only: how many full window repeats to pool per bucket")
    parser.add_argument("--score-radius", type=int, default=2,
                         help="--calibrate only: must match the trained checkpoint's score_radius "
                              "(every checkpoint in this repo uses 2) -- sets how many distinct "
                              "distance buckets (0..score_radius) get their own calibration level "
                              "before everything past the radius collapses into one background level")
    parser.add_argument("--calibration-file", default=DEFAULT_CALIBRATION_FILE,
                         help="--calibrate only: where to save the resulting JSON curve")
    args = parser.parse_args()

    target = tuple(args.target)
    origin = compute_window_origin(target, WINDOW_SIZE, GRID_SIZE)

    ser = connect()
    clear(ser)

    if args.calibrate:
        calibration, avg_readings, ser = calibrate(
            ser, origin, WINDOW_SIZE, target, args.linger_seconds,
            args.calibration_runs, args.score_radius, args.dry_run,
        )
        ser.close()

        with open(args.calibration_file, "w") as f:
            json.dump(calibration, f, indent=2)
        print(f"\nWrote calibration to {args.calibration_file}:")
        for bucket, level in sorted(calibration["levels"].items()):
            print(f"  {bucket:>10} -> score={level['score']:.4f} "
                  f"avg_delta={level['avg_delta']:.1f} (n={level['n_samples']})")

        plot_readings(avg_readings, origin, WINDOW_SIZE, target,
                       title=f"LDR calibration: delta vs. LED position (target={target}, "
                             f"{args.calibration_runs} runs averaged)",
                       out_file="ldr_calibration_plot.png")
    else:
        readings = sweep(ser, origin, WINDOW_SIZE, args.linger_seconds, args.dry_run)
        clear(ser)
        ser.close()

        plot_readings(readings, origin, WINDOW_SIZE, target,
                       title=f"LDR response vs. LED position (target={target})",
                       out_file="ldr_sweep_plot.png")


if __name__ == "__main__":
    main()
