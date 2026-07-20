#pragma once

#include <stdint.h>

// Interface every LED board driver must implement. led_serial_listener.ino
// only calls through these functions, so swapping to a different LED board
// just means writing a new driver file against this same interface --
// the listener sketch itself never changes.

void ledBoardInit();
bool ledBoardSetPixel(int x, int y);

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
