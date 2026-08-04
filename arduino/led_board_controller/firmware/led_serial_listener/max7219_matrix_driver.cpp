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

bool ledBoardClear() {
  lc.clearDisplay(0);
  return true;
}

bool ledBoardSetFrame(const uint8_t rows[8]) {
  for (int row = 0; row < 8; row++) {
    lc.setRow(0, row, rows[row]);
  }
  return true;
}

bool ledBoardSetRow(int y, uint8_t r, uint8_t g, uint8_t b) {
  if (y < 0 || y > 7) {
    return false;
  }
  // Monochrome -- no color, so any non-black value just means "on" for the
  // whole row.
  lc.setRow(0, y, (r || g || b) ? 0xFF : 0x00);
  return true;
}

bool ledBoardSetPixelColor(int x, int y, uint8_t r, uint8_t g, uint8_t b) {
  if (x < 0 || x > 7 || y < 0 || y > 7) {
    return false;
  }
  // Monochrome -- no color, so any non-black value just means "on".
  lc.clearDisplay(0);
  lc.setLed(0, y, x, (r || g || b));
  return true;
}

bool ledBoardSetEpisodeLayer(const LedPoint* boundary, int boundaryCount, LedPoint target) {
  // No color and no retained multi-point state on this board -- the
  // 16x16-only blinking boundary/target display isn't supported here.
  (void)boundary;
  (void)boundaryCount;
  (void)target;
  return false;
}

bool ledBoardSetDynamicLayer(LedPoint agent, const LedPoint* trail, int trailCount) {
  (void)agent;
  (void)trail;
  (void)trailCount;
  return false;
}

bool ledBoardSetThinkingLayer(const LedPoint* points, const uint8_t* brightness, int count) {
  // No color and no retained multi-point state on this board.
  (void)points;
  (void)brightness;
  (void)count;
  return false;
}

void ledBoardTick() {
  // Nothing animates on its own clock on this board.
}

#endif // ACTIVE_BOARD == BOARD_MAX7219_MATRIX
