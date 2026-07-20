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
