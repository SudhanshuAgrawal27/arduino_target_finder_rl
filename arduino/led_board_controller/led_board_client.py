import time

import serial

PORT = "/dev/ttyUSB0"
BAUD = 115200


def connect(port=PORT, baud=BAUD):
    ser = serial.Serial(port, baud, timeout=1)
    time.sleep(2)  # wait for the board to reset after the port opens
    ser.reset_input_buffer()  # discard any boot banner / stale bytes before the first command
    return ser


def set_led(ser, x, y):
    ser.write(f"{x},{y}\n".encode())
    return ser.readline().decode().strip()


def set_frame(ser, points):
    """Light every (x, y) in `points` at once, replacing whatever was
    previously lit. Matches led_serial_listener.ino's "F:r0,...,r7" format --
    one byte per row, bit x of row y set for each point (x, y)."""
    rows = [0] * 8
    for x, y in points:
        if 0 <= x < 8 and 0 <= y < 8:
            rows[y] |= 1 << x
    ser.write(("F:" + ",".join(str(r) for r in rows) + "\n").encode())
    return ser.readline().decode().strip()


def set_row(ser, y, r, g, b):
    """Light row `y` the solid color (r, g, b), leaving every other row as
    it was (doesn't clear the board) -- so calling this once per row builds
    up a multi-row pattern across several calls."""
    ser.write(f"R:{y},{r},{g},{b}\n".encode())
    return ser.readline().decode().strip()


def set_pixel_color(ser, x, y, r, g, b):
    """Light a single pixel (x, y) the color (r, g, b), clearing everything
    else first -- the colored equivalent of set_led."""
    ser.write(f"P:{x},{y},{r},{g},{b}\n".encode())
    return ser.readline().decode().strip()
