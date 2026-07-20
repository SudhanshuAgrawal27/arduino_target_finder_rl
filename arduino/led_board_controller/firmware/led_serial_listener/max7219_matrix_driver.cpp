#include "board_config.h"

#if ACTIVE_BOARD == BOARD_MAX7219_MATRIX

#include <LedControl.h>

#include "led_board_driver.h"

namespace {
LedControl lc = LedControl(12, 11, 10, 1); // DIN, CLK, CS, #devices
}

void ledBoardInit() {
  lc.shutdown(0, false);   // wake up the display
  lc.setIntensity(0, 8);   // brightness 0-15
  lc.clearDisplay(0);
}

bool ledBoardSetPixel(int x, int y) {
  if (x < 0 || x > 7 || y < 0 || y > 7) {
    return false;
  }
  lc.clearDisplay(0);
  lc.setLed(0, y, x, true); // row = y, col = x
  return true;
}

bool ledBoardSetFrame(const uint8_t rows[8]) {
  for (int row = 0; row < 8; row++) {
    lc.setRow(0, row, rows[row]);
  }
  return true;
}

#endif // ACTIVE_BOARD == BOARD_MAX7219_MATRIX
