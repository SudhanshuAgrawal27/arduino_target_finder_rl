#include "board_config.h"

#if ACTIVE_BOARD == BOARD_WS2812B_MATRIX

#include <Adafruit_NeoPixel.h>

#include "led_board_driver.h"

namespace {
constexpr int kWidth = 16;
constexpr int kHeight = 16;
constexpr int kDataPin = 6;    // verify against actual wiring
constexpr bool kSerpentine = true;  // most 16x16 WS2812B panels wire alternate rows reversed

// Kept low on purpose: 256 WS2812B LEDs at full brightness/white can draw
// several amps, far more than a USB port or the board's onboard regulator
// can supply. Raise only with an adequate external 5V supply.
constexpr uint8_t kBrightness = 40;

Adafruit_NeoPixel strip(kWidth * kHeight, kDataPin, NEO_GRB + NEO_KHZ800);

int pixelIndex(int x, int y) {
  if (kSerpentine && (y % 2 == 1)) {
    x = kWidth - 1 - x;
  }
  return y * kWidth + x;
}
}  // namespace

void ledBoardInit() {
  strip.begin();
  strip.setBrightness(kBrightness);
  strip.clear();
  strip.show();
}

bool ledBoardSetPixel(int x, int y) {
  if (x < 0 || x >= kWidth || y < 0 || y >= kHeight) {
    return false;
  }
  strip.clear();
  strip.setPixelColor(pixelIndex(x, y), strip.Color(255, 255, 255));
  strip.show();
  return true;
}

bool ledBoardSetFrame(const uint8_t rows[8]) {
  // rows[8] (one byte per row) is sized for an 8x8 board and can't address
  // a 16x16 panel -- this driver only supports the single-pixel command for
  // now (see ledBoardSetPixel). Extend led_board_driver.h's interface if/when
  // eval_demo needs multi-pixel frames on this board.
  (void)rows;
  return false;
}

bool ledBoardSetRow(int y, uint8_t r, uint8_t g, uint8_t b) {
  if (y < 0 || y >= kHeight) {
    return false;
  }
  uint32_t color = strip.Color(r, g, b);
  for (int x = 0; x < kWidth; x++) {
    strip.setPixelColor(pixelIndex(x, y), color);
  }
  strip.show();
  return true;
}

bool ledBoardSetPixelColor(int x, int y, uint8_t r, uint8_t g, uint8_t b) {
  if (x < 0 || x >= kWidth || y < 0 || y >= kHeight) {
    return false;
  }
  strip.clear();
  strip.setPixelColor(pixelIndex(x, y), strip.Color(r, g, b));
  strip.show();
  return true;
}

#endif  // ACTIVE_BOARD == BOARD_WS2812B_MATRIX
