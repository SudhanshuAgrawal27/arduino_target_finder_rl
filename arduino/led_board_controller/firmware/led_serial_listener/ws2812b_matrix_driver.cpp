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

// Fixed per-role colors for the eval_demo_16-16 grid-search display (see
// ledBoardSetEpisodeLayer/ledBoardSetDynamicLayer). Chosen dim on purpose:
// with kBrightness=40 already capping the whole strip, these keep the
// *relative* mix between simultaneously-lit roles readable (agent clearly
// brighter than its trail; boundary/target dim enough to read as
// "background" next to the agent) without depending on per-pixel
// brightness, which NeoPixel doesn't support -- only strip.setBrightness()
// as a single global scalar.
constexpr uint8_t kAgentR = 255, kAgentG = 255, kAgentB = 255;  // full white
constexpr uint8_t kTrailR = 15, kTrailG = 0, kTrailB = 0;       // dim red
constexpr uint8_t kBoundaryR = 0, kBoundaryG = 20, kBoundaryB = 0;  // dim green
constexpr uint8_t kTargetR = 0, kTargetG = 0, kTargetB = 60;    // dim blue

// ~33Hz full on/off cycle (50% duty, same as before -- see
// power_model.py's BLINK_DUTY_CYCLE, which doesn't depend on this
// constant, so this change doesn't move the average-current estimate at
// all) for the episode layer (boundary + target) and trail. The original
// ~33Hz was well below typical flicker-fusion thresholds and read as
// distinct flashing rather than a steady dim glow.
//
// Deliberately NOT pushed higher than this: render() itself takes a fixed
// ~7.7ms to bit-bang all 256 pixels, with interrupts disabled the whole
// time (see ledBoardTick's comment). At this half-period, that's ~51% of
// each cycle -- already a real cut into the "quiet" window the E:/D:
// corruption fixes rely on. Going much faster (e.g. a half-period near
// 7.7ms) would push that toward ~100%, making every serial command
// collide with a render almost by default and lean entirely on retries
// rather than mostly avoiding the problem.
constexpr unsigned long kBlinkHalfPeriodMs = 15;

Adafruit_NeoPixel strip(kWidth * kHeight, kDataPin, NEO_GRB + NEO_KHZ800);

// Retained state for the two layers set by ledBoardSetEpisodeLayer/
// ledBoardSetDynamicLayer, composited together on every render() -- see
// led_board_driver.h for why these are separate calls (episode layer is
// set once per episode, dynamic layer once per step; both feed the same
// ~33Hz blink cycle except for the agent, which always stays lit).
LedPoint boundaryPoints[kMaxBoundaryPoints];
int boundaryCount = 0;
LedPoint targetPoint = {-1, -1};
bool hasTarget = false;

LedPoint trailPoints[kMaxTrailPoints];
int trailCount = 0;
LedPoint agentPoint = {-1, -1};
bool hasAgent = false;

// See ledBoardSetThinkingLayer -- always lit while active (not gated by
// blinkOn), cleared as a side effect of the next ledBoardSetDynamicLayer.
LedPoint thinkingPoints[kMaxThinkingPoints];
uint8_t thinkingBrightness[kMaxThinkingPoints];
int thinkingCount = 0;

bool blinkOn = true;
unsigned long lastBlinkToggle = 0;

int pixelIndex(int x, int y) {
  if (kSerpentine && (y % 2 == 1)) {
    x = kWidth - 1 - x;
  }
  return y * kWidth + x;
}

// Recomposites both layers and pushes the result to the strip. Boundary,
// target, AND trail all blink together at ~33Hz to save power -- only the
// agent's current position stays lit continuously, since that's the one
// thing that always needs to be visible. Draw order within the blink
// group doesn't matter (none of them overlap in practice), but the agent
// is always drawn last so it reads clearly even if it currently sits on
// the boundary outline or (at episode end) the target.
void render() {
  strip.clear();

  if (blinkOn) {
    for (int i = 0; i < boundaryCount; i++) {
      strip.setPixelColor(pixelIndex(boundaryPoints[i].x, boundaryPoints[i].y),
                           strip.Color(kBoundaryR, kBoundaryG, kBoundaryB));
    }
    if (hasTarget) {
      strip.setPixelColor(pixelIndex(targetPoint.x, targetPoint.y),
                           strip.Color(kTargetR, kTargetG, kTargetB));
    }
    for (int i = 0; i < trailCount; i++) {
      strip.setPixelColor(pixelIndex(trailPoints[i].x, trailPoints[i].y),
                           strip.Color(kTrailR, kTrailG, kTrailB));
    }
  }

  // Drawn after the blink group but before the agent, so the agent still
  // reads clearly if it happens to overlap a thinking candidate (e.g. an
  // illegal move's candidate cell coincides with the agent's own current
  // position).
  for (int i = 0; i < thinkingCount; i++) {
    uint8_t v = thinkingBrightness[i];
    strip.setPixelColor(pixelIndex(thinkingPoints[i].x, thinkingPoints[i].y),
                         strip.Color(v, v, 0));
  }

  if (hasAgent) {
    strip.setPixelColor(pixelIndex(agentPoint.x, agentPoint.y),
                         strip.Color(kAgentR, kAgentG, kAgentB));
  }

  strip.show();
}

bool inBounds(LedPoint p) {
  return p.x >= 0 && p.x < kWidth && p.y >= 0 && p.y < kHeight;
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

bool ledBoardSetEpisodeLayer(const LedPoint* boundary, int boundaryN, LedPoint target) {
  boundaryCount = 0;
  for (int i = 0; i < boundaryN && boundaryCount < kMaxBoundaryPoints; i++) {
    if (inBounds(boundary[i])) {
      boundaryPoints[boundaryCount++] = boundary[i];
    }
  }

  hasTarget = inBounds(target);
  targetPoint = target;

  // Restart the blink phase on-lit, so a fresh episode's boundary/target
  // are immediately visible rather than possibly appearing mid-off-phase.
  blinkOn = true;
  lastBlinkToggle = millis();

  render();
  return true;
}

bool ledBoardSetDynamicLayer(LedPoint agent, const LedPoint* trail, int trailN) {
  hasAgent = inBounds(agent);
  agentPoint = agent;

  trailCount = 0;
  for (int i = 0; i < trailN && trailCount < kMaxTrailPoints; i++) {
    if (inBounds(trail[i])) {
      trailPoints[trailCount++] = trail[i];
    }
  }

  // The step this dynamic-layer update represents has actually happened by
  // now, so any thinking-layer preview from just before it is stale --
  // clear it as a side effect rather than requiring a separate command.
  thinkingCount = 0;

  render();
  return true;
}

bool ledBoardSetThinkingLayer(const LedPoint* points, const uint8_t* brightness, int count) {
  thinkingCount = 0;
  for (int i = 0; i < count && thinkingCount < kMaxThinkingPoints; i++) {
    if (inBounds(points[i])) {
      thinkingPoints[thinkingCount] = points[i];
      thinkingBrightness[thinkingCount] = brightness[i];
      thinkingCount++;
    }
  }

  render();
  return true;
}

bool ledBoardClear() {
  // Resets the retained layer state too, not just the visible pixels --
  // otherwise a stray autonomous blink tick right after this call would
  // immediately redraw the previous episode's boundary/target/trail from
  // whatever was last set, defeating the point of clearing.
  boundaryCount = 0;
  hasTarget = false;
  trailCount = 0;
  hasAgent = false;
  thinkingCount = 0;
  blinkOn = true;
  lastBlinkToggle = millis();

  strip.clear();
  strip.show();
  return true;
}

void ledBoardTick() {
  // strip.show() (inside render()) disables interrupts for the several ms
  // it takes to bit-bang the whole panel -- long enough that UART bytes
  // arriving mid-call get dropped by the hardware's tiny receive buffer,
  // corrupting whatever command line was in flight. If a byte has already
  // started arriving, skip this blink tick and let loop() finish reading
  // the command first; we'll pick the blink back up next pass. This
  // doesn't fully close the race (a command can still start arriving
  // during an already-in-progress show()), but it closes the far more
  // common case of starting a new show() on top of a partially-received
  // line -- see led_board_client.py's retry-on-ERR for the rest.
  if (Serial.available()) {
    return;
  }

  unsigned long now = millis();
  if (now - lastBlinkToggle >= kBlinkHalfPeriodMs) {
    lastBlinkToggle = now;
    blinkOn = !blinkOn;
    render();
  }
}

#endif  // ACTIVE_BOARD == BOARD_WS2812B_MATRIX
