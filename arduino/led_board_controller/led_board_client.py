import time

import serial

PORT = "/dev/ttyUSB0"
BAUD = 115200

# The WS2812B driver's ~33Hz blink runs on its own free-running timer (see
# ledBoardTick in ws2812b_matrix_driver.cpp), independent of serial
# traffic. strip.show() disables interrupts for the several ms it takes to
# bit-bang the whole panel, so a command whose bytes arrive during that
# window can be partially dropped by the UART's tiny hardware buffer and
# come back corrupted -- the board detects this (parsing fails) and
# correctly replies "ERR" rather than silently doing the wrong thing (the
# firmware also skips a blink tick if a byte is already arriving, to
# shrink the window further, but can't close it entirely).
#
# At the current ~33Hz (kBlinkHalfPeriodMs=15 in ws2812b_matrix_driver.cpp),
# each ~7.7ms render leaves only ~7.3ms of quiet time before the next one,
# so a single short command has roughly a 35-50% chance of landing clean on
# any given attempt -- meaningfully worse than at the original ~16Hz, where
# the quiet window was much wider. A handful of retries isn't enough
# headroom to keep the failure probability negligible across a whole
# episode's worth of commands, so this is deliberately generous: even at a
# ~50% chance of failure per attempt, this many attempts pushes the
# all-attempts-fail probability for one command well under 0.1%.
_MAX_ATTEMPTS = 20
_RETRY_DELAY_SECONDS = 0.05


def connect(port=PORT, baud=BAUD):
    ser = serial.Serial(port, baud, timeout=1)
    time.sleep(2)  # wait for the board to reset after the port opens
    ser.reset_input_buffer()  # discard any boot banner / stale bytes before the first command
    return ser


def _send(ser, line):
    """Write `line` and return the board's stripped reply, retrying a few
    times on anything other than "OK" -- see _MAX_ATTEMPTS above for why a
    non-OK reply is usually a transient blink-tick race rather than a real
    failure."""
    reply = ""
    for attempt in range(_MAX_ATTEMPTS):
        ser.write(line.encode())
        reply = ser.readline().decode().strip()
        if reply == "OK":
            return reply
        if attempt < _MAX_ATTEMPTS - 1:
            time.sleep(_RETRY_DELAY_SECONDS)
    return reply


def clear(ser):
    """Turn off every pixel and, on boards with retained multi-layer state
    (WS2812B), reset it too -- so a subsequent autonomous blink tick can't
    redraw stale boundary/target/trail/agent state left over from a
    previous run. Call this once per script invocation, right after
    connect(), so a fresh run never inherits whatever the board was last
    showing (the hardware's auto-reset on port-open is not fully reliable
    over a passed-through USB connection -- see arduino/README.md)."""
    return _send(ser, "C\n")


def set_led(ser, x, y):
    return _send(ser, f"{x},{y}\n")


def set_frame(ser, points):
    """Light every (x, y) in `points` at once, replacing whatever was
    previously lit. Matches led_serial_listener.ino's "F:r0,...,r7" format --
    one byte per row, bit x of row y set for each point (x, y)."""
    rows = [0] * 8
    for x, y in points:
        if 0 <= x < 8 and 0 <= y < 8:
            rows[y] |= 1 << x
    return _send(ser, "F:" + ",".join(str(r) for r in rows) + "\n")


def set_row(ser, y, r, g, b):
    """Light row `y` the solid color (r, g, b), leaving every other row as
    it was (doesn't clear the board) -- so calling this once per row builds
    up a multi-row pattern across several calls."""
    return _send(ser, f"R:{y},{r},{g},{b}\n")


def set_pixel_color(ser, x, y, r, g, b):
    """Light a single pixel (x, y) the color (r, g, b), clearing everything
    else first -- the colored equivalent of set_led."""
    return _send(ser, f"P:{x},{y},{r},{g},{b}\n")


def set_episode_layer(ser, boundary_points, target_point):
    """WS2812B only. Set the per-episode layer: a boundary outline (dim
    green) and a target (dim blue), both retained and blinked at ~33Hz by
    the board's own timer until the next call -- so this only needs to be
    sent once per episode, not once per blink frame. Points are (x, y) in
    whatever coordinate space the caller is drawing in (eval_demo_16-16.py
    uses global board coordinates).

    The point list is prefixed with its own length ("N:x,y;x,y;...") so the
    firmware can detect a line that got truncated/corrupted in transit
    (see _send's docstring on the blink-tick race) and reply "ERR" -- since
    ledBoardSetEpisodeLayer's underlying command always "succeeds" even
    with zero boundary points, a length mismatch is the only way to catch
    a partially-dropped boundary list rather than silently rendering it as
    empty."""
    boundary_points = list(boundary_points)
    boundary = ";".join(f"{x},{y}" for x, y in boundary_points)
    tx, ty = target_point
    return _send(ser, f"E:{len(boundary_points)}:{boundary}|{tx},{ty}\n")


def set_dynamic_layer(ser, agent_point, trail_points):
    """WS2812B only. Set the per-step layer: the agent's current position
    (full-bright red, always lit) and its trail of prior positions (dim
    red, blinking in sync with the episode layer's ~33Hz cycle to save
    power). Call once per env step.

    Like set_episode_layer, the trail list is prefixed with its own length
    so a truncated line is detected and retried rather than silently
    rendering a partial trail."""
    ax, ay = agent_point
    trail_points = list(trail_points)
    trail = ";".join(f"{x},{y}" for x, y in trail_points)
    return _send(ser, f"D:{ax},{ay}|{len(trail_points)}:{trail}\n")
