#pragma once

// Change this line to select which LED board driver gets compiled in.
// Every driver .cpp guards its whole implementation behind a check against
// ACTIVE_BOARD, so exactly one of them actually compiles -- the rest
// compile to nothing, avoiding duplicate-symbol link errors even though
// all driver files live in this same sketch folder.

#define BOARD_MAX7219_MATRIX 1
#define BOARD_WS2812B_MATRIX 2

#define ACTIVE_BOARD BOARD_WS2812B_MATRIX
