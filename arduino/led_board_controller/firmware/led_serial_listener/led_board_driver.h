#pragma once

#include <stdint.h>

// Interface every LED board driver must implement. led_serial_listener.ino
// only calls through these functions, so swapping to a different LED board
// just means writing a new driver file against this same interface --
// the listener sketch itself never changes.

// A single board coordinate, used by the multi-point episode/dynamic layer
// commands below. int8_t is plenty for any board size we target.
struct LedPoint {
  int8_t x;
  int8_t y;
};

// Upper bounds on how many points a single "E:"/"D:" command can carry,
// used both by led_serial_listener.ino (to size its parse buffers) and by
// driver implementations (to size their retained-state arrays). 64 covers
// the boundary outline of an 8x8 subgrid +1 margin on a 16x16 board (36
// points) with headroom; 16 comfortably covers any realistic trail length.
constexpr int kMaxBoundaryPoints = 64;
constexpr int kMaxTrailPoints = 16;

void ledBoardInit();
bool ledBoardSetPixel(int x, int y);

// Turns off every pixel and, on boards with retained multi-layer state
// (WS2812B), resets it too -- so a subsequent autonomous blink tick can't
// redraw stale boundary/target/trail/agent state left over from before
// this call. Meant to be sent once per script run, right after connecting,
// so a fresh run never inherits whatever the board was last showing (e.g.
// if the board's auto-reset-on-port-open didn't actually happen, which can
// be flaky over a passed-through USB connection).
bool ledBoardClear();

// Lights every pixel in `rows` at once (rows[y] bit x = column x of row y),
// replacing whatever was previously lit. Used to show several fixed points
// (target, agent, trail) simultaneously -- ledBoardSetPixel only ever lights
// one point at a time.
bool ledBoardSetFrame(const uint8_t rows[8]);

// Lights every pixel in row `y` the solid color (r, g, b), leaving every
// other row as-is (does NOT clear the rest of the board) -- so calling this
// once per row builds up a multi-row pattern across several calls. Boards
// without per-pixel color (e.g. monochrome MAX7219) treat any non-black
// color as "on" for that row.
bool ledBoardSetRow(int y, uint8_t r, uint8_t g, uint8_t b);

// Lights a single pixel (x, y) the color (r, g, b), clearing everything
// else first -- the colored equivalent of ledBoardSetPixel, for a sweep
// test where each pixel's color depends on its row. Boards without
// per-pixel color (e.g. monochrome MAX7219) treat any non-black color as
// "on".
bool ledBoardSetPixelColor(int x, int y, uint8_t r, uint8_t g, uint8_t b);

// Sets the per-EPISODE layer: a boundary outline (drawn dim green) and a
// single target point (drawn dim blue). Both are retained and blinked at a
// fixed ~33Hz by the driver's own free-running timer (see ledBoardTick)
// until the next ledBoardSetEpisodeLayer call replaces them -- the caller
// only needs to send this once per episode, not once per blink frame.
// Boards without per-pixel color/retained state (e.g. monochrome MAX7219)
// return false.
bool ledBoardSetEpisodeLayer(const LedPoint* boundary, int boundaryCount, LedPoint target);

// Sets the per-STEP layer: the agent's current position (drawn full-bright
// red, always lit) and its trail of prior positions (drawn dim red,
// blinking in sync with the episode layer's ~33Hz cycle to save power).
// Composited with the episode layer on every render. Called once per env
// step. Boards without per-pixel color/retained state (e.g. monochrome
// MAX7219) return false.
bool ledBoardSetDynamicLayer(LedPoint agent, const LedPoint* trail, int trailCount);

// Called on every pass through the sketch's main loop (not just when a
// serial command arrives), so a driver that needs to animate on its own
// clock -- e.g. toggling the episode layer's blink state at ~33Hz between
// commands -- can do so without the caller having to drive it. Boards with
// nothing to animate (e.g. monochrome MAX7219) implement this as a no-op.
void ledBoardTick();
