"""Estimates LED count / current draw for the eval_demo_16-16 display.

Mirrors constants that actually live in firmware (see
firmware/led_serial_listener/ws2812b_matrix_driver.cpp) -- Python and the
Arduino sketch can't share source, so these are kept in sync by hand. If
the firmware's colors, kBrightness, or blink timing ever change, update the
matching constants below.
"""

# Per-role RGB colors and the global strip brightness scale -- must match
# ws2812b_matrix_driver.cpp's kAgentR/kTrailR/kBoundaryR/kTargetR (colors)
# and kBrightness (global scale, applied on top of these by strip.setBrightness()).
AGENT_COLOR = (255, 0, 0)
TRAIL_COLOR = (15, 0, 0)
BOUNDARY_COLOR = (0, 20, 0)
TARGET_COLOR = (0, 0, 60)
GLOBAL_BRIGHTNESS = 40  # out of 255

# Standard WS2812B current-draw approximation: ~20mA per color channel at
# full value (255). This is the same figure arduino/README.md's "256 LEDs
# at full brightness/white can draw several amps" note is based on
# (256 * 3 * 20mA =~ 15.4A).
MA_PER_CHANNEL_AT_FULL = 20.0
SUPPLY_VOLTAGE = 5.0

# Boundary, target, and trail all blink at ~33Hz with a 50% duty cycle (see
# ledBoardTick/kBlinkHalfPeriodMs in ws2812b_matrix_driver.cpp); only the
# agent's current position is always lit.
BLINK_DUTY_CYCLE = 0.5


def led_current_ma(r, g, b, global_brightness=GLOBAL_BRIGHTNESS):
    """Estimated current draw (mA) of one WS2812B LED set to (r, g, b),
    after the global strip brightness scale that NeoPixel applies to every
    channel before actually driving the LEDs -- so this is the real
    estimated draw, not just a function of the raw color passed to
    strip.setPixelColor."""
    scale = global_brightness / 255.0
    return sum(c * scale / 255.0 * MA_PER_CHANNEL_AT_FULL for c in (r, g, b))


AGENT_MA = led_current_ma(*AGENT_COLOR)
TRAIL_MA = led_current_ma(*TRAIL_COLOR)
BOUNDARY_MA = led_current_ma(*BOUNDARY_COLOR)
TARGET_MA = led_current_ma(*TARGET_COLOR)


def frame_power_stats(boundary_count, trail_count, has_target=True, has_agent=True):
    """LED count / current draw for one composited frame: `boundary_count`
    boundary pixels + `trail_count` trail pixels + (optionally) the target
    all blink together at BLINK_DUTY_CYCLE; the agent (if present) stays
    lit continuously.

    Returns a dict with the blink-on-phase ("lit_phase") and blink-off-
    phase ("dark_phase") LED count/current, plus the duty-cycle-weighted
    "avg" of both -- the number of LEDs "effectively on" and the current a
    supply actually has to sustain on average, not just at the brightest
    instant.
    """
    blinking_count = boundary_count + trail_count + (1 if has_target else 0)
    static_count = 1 if has_agent else 0

    blinking_ma = (
        boundary_count * BOUNDARY_MA
        + trail_count * TRAIL_MA
        + (TARGET_MA if has_target else 0.0)
    )
    static_ma = AGENT_MA if has_agent else 0.0

    leds_lit_phase = blinking_count + static_count
    leds_dark_phase = static_count
    ma_lit_phase = blinking_ma + static_ma
    ma_dark_phase = static_ma

    return {
        "leds_lit_phase": leds_lit_phase,
        "leds_dark_phase": leds_dark_phase,
        "leds_avg": BLINK_DUTY_CYCLE * leds_lit_phase + (1 - BLINK_DUTY_CYCLE) * leds_dark_phase,
        "ma_lit_phase": ma_lit_phase,
        "ma_dark_phase": ma_dark_phase,
        "ma_avg": BLINK_DUTY_CYCLE * ma_lit_phase + (1 - BLINK_DUTY_CYCLE) * ma_dark_phase,
    }
