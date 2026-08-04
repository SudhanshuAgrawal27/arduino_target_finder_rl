#include <stdlib.h>
#include <string.h>

#include "led_board_driver.h"

namespace {

// LM358 photoresistor module's analog output. Independent of ACTIVE_BOARD --
// A0 is unused by both drivers (MAX7219: DIN/CLK/CS on 12/11/10; WS2812B:
// data on 6) -- so it's read here in the shared listener rather than behind
// led_board_driver.h, which is only about pixel output.
constexpr int kLdrPin = A0;

// Sized with headroom over the longest line we ever send: a full,
// unclipped 36-point boundary + its length prefix + a target comes to
// ~155 bytes (see arduino/README.md's protocol table).
constexpr int kLineBufferSize = 300;
char lineBuf[kLineBufferSize];

// Parses a ';'-separated list of "x,y" points out of the NUL-terminated,
// mutable C string `s` into `out` (capped at `maxOut`), returning how many
// were parsed. Writes '\0' into `s` at each delimiter as it goes (a
// standard in-place tokenizing technique) rather than allocating a String
// per token -- on a long (30+ point) boundary list, String::substring()
// once per token meant 30+ heap allocations per command on a chip with
// only ~1.6KB of free RAM, which could exhaust/fragment the heap and fail
// deterministically rather than just occasionally. Used by "E:"'s boundary
// list and "D:"'s trail list -- both variable-length, unlike every other
// command here which has a fixed field count.
int parsePoints(char* s, LedPoint* out, int maxOut) {
  int count = 0;
  char* cursor = s;
  while (*cursor != '\0' && count < maxOut) {
    char* semicolon = strchr(cursor, ';');
    if (semicolon != nullptr) {
      *semicolon = '\0';
    }

    char* comma = strchr(cursor, ',');
    if (comma != nullptr) {
      *comma = '\0';
      out[count].x = (int8_t)atoi(cursor);
      out[count].y = (int8_t)atoi(comma + 1);
      count++;
    }

    if (semicolon == nullptr) {
      break;
    }
    cursor = semicolon + 1;
  }
  return count;
}

// Like parsePoints, but each token is "x,y,v" (v = brightness 0-255) rather
// than just "x,y" -- used only by "T:"'s thinking-layer point list.
int parseThinkingPoints(char* s, LedPoint* outPoints, uint8_t* outBrightness, int maxOut) {
  int count = 0;
  char* cursor = s;
  while (*cursor != '\0' && count < maxOut) {
    char* semicolon = strchr(cursor, ';');
    if (semicolon != nullptr) {
      *semicolon = '\0';
    }

    char* comma1 = strchr(cursor, ',');
    char* comma2 = comma1 != nullptr ? strchr(comma1 + 1, ',') : nullptr;
    if (comma1 != nullptr && comma2 != nullptr) {
      *comma1 = '\0';
      *comma2 = '\0';
      outPoints[count].x = (int8_t)atoi(cursor);
      outPoints[count].y = (int8_t)atoi(comma1 + 1);
      outBrightness[count] = (uint8_t)atoi(comma2 + 1);
      count++;
    }

    if (semicolon == nullptr) {
      break;
    }
    cursor = semicolon + 1;
  }
  return count;
}

}  // namespace

void setup() {
  Serial.begin(115200);
  ledBoardInit();
}

// Nine line formats share this listener:
//   "C"                            -- turn off every pixel and reset all
//                                     retained layer state
//   "L"                            -- read the LM358 photoresistor (A0) and
//                                     reply with the raw integer directly --
//                                     the only command that isn't "OK"/"ERR",
//                                     since its whole point is returning data
//   "x,y"                          -- light a single pixel (clears the rest)
//   "F:r0,r1,r2,r3,r4,r5,r6,r7"    -- light a whole frame at once, one byte
//                                     per row (bit x of row y = pixel x,y)
//   "R:y,r,g,b"                    -- light row y a solid color, leaving
//                                     every other row as-is
//   "P:x,y,r,g,b"                  -- light a single pixel that color
//                                     (clears the rest)
//   "E:N:x,y;x,y;...|tx,ty"        -- set the per-episode layer: a boundary
//                                     point list (prefixed with its own
//                                     length N, so a line truncated in
//                                     transit is detected rather than
//                                     silently accepted as a shorter list)
//                                     plus a target point, both retained
//                                     and blinked at ~33Hz by the driver
//                                     until the next "E:" command
//   "D:ax,ay|N:x,y;x,y;..."        -- set the per-step layer: the agent
//                                     point (always lit) plus a trail point
//                                     list, also length-prefixed, that
//                                     blinks with the episode layer
//   "T:N:x,y,v;x,y,v;..."          -- set the transient "thinking" layer:
//                                     up to 4 points, each a shade of
//                                     yellow scaled by its own brightness
//                                     v (0-255), always lit until the next
//                                     "D:" implicitly clears it
//
// Reads directly into a fixed char buffer (no String involved anywhere in
// this file) so a long line never triggers a heap allocation -- see
// parsePoints's comment.
//
// ledBoardTick() runs every loop pass (not just on incoming commands) so a
// driver can animate on its own clock -- e.g. the WS2812B driver's ~33Hz
// episode-layer blink -- between commands.
void loop() {
  ledBoardTick();

  if (Serial.available()) {
    int len = Serial.readBytesUntil('\n', lineBuf, sizeof(lineBuf) - 1);
    lineBuf[len] = '\0';

    if (lineBuf[0] == 'L' && lineBuf[1] == '\0') {
      Serial.println(analogRead(kLdrPin));
      return;
    }

    bool ok = false;

    if (lineBuf[0] == 'C' && lineBuf[1] == '\0') {
      ok = ledBoardClear();
    } else if (lineBuf[0] == 'E' && lineBuf[1] == ':') {
      char* body = lineBuf + 2;
      char* bar = strchr(body, '|');
      if (bar != nullptr) {
        *bar = '\0';
        char* targetStr = bar + 1;

        char* headColon = strchr(body, ':');
        char* comma = strchr(targetStr, ',');
        if (headColon != nullptr && comma != nullptr) {
          *headColon = '\0';
          int expectedCount = atoi(body);
          char* boundaryStr = headColon + 1;

          LedPoint boundary[kMaxBoundaryPoints];
          int boundaryCount = parsePoints(boundaryStr, boundary, kMaxBoundaryPoints);

          // A count mismatch means the line was truncated/corrupted in
          // transit -- reject it rather than silently rendering a
          // partial/empty boundary, so the caller's retry-on-ERR kicks in.
          if (boundaryCount == expectedCount) {
            *comma = '\0';
            LedPoint target;
            target.x = (int8_t)atoi(targetStr);
            target.y = (int8_t)atoi(comma + 1);
            ok = ledBoardSetEpisodeLayer(boundary, boundaryCount, target);
          }
        }
      }
    } else if (lineBuf[0] == 'D' && lineBuf[1] == ':') {
      char* body = lineBuf + 2;
      char* bar = strchr(body, '|');
      if (bar != nullptr) {
        *bar = '\0';
        char* agentStr = body;
        char* tail = bar + 1;

        char* comma = strchr(agentStr, ',');
        char* tailColon = strchr(tail, ':');
        if (comma != nullptr && tailColon != nullptr) {
          *tailColon = '\0';
          int expectedCount = atoi(tail);
          char* trailStr = tailColon + 1;

          LedPoint trail[kMaxTrailPoints];
          int trailCount = parsePoints(trailStr, trail, kMaxTrailPoints);

          if (trailCount == expectedCount) {
            *comma = '\0';
            LedPoint agent;
            agent.x = (int8_t)atoi(agentStr);
            agent.y = (int8_t)atoi(comma + 1);
            ok = ledBoardSetDynamicLayer(agent, trail, trailCount);
          }
        }
      }
    } else if (lineBuf[0] == 'T' && lineBuf[1] == ':') {
      char* body = lineBuf + 2;
      char* headColon = strchr(body, ':');
      if (headColon != nullptr) {
        *headColon = '\0';
        int expectedCount = atoi(body);
        char* pointsStr = headColon + 1;

        LedPoint points[kMaxThinkingPoints];
        uint8_t brightness[kMaxThinkingPoints];
        int count = parseThinkingPoints(pointsStr, points, brightness, kMaxThinkingPoints);

        if (count == expectedCount) {
          ok = ledBoardSetThinkingLayer(points, brightness, count);
        }
      }
    } else if (lineBuf[0] == 'F' && lineBuf[1] == ':') {
      int fields[8];
      int n = sscanf(lineBuf + 2, "%d,%d,%d,%d,%d,%d,%d,%d",
                      &fields[0], &fields[1], &fields[2], &fields[3],
                      &fields[4], &fields[5], &fields[6], &fields[7]);
      if (n == 8) {
        uint8_t rows[8];
        for (int i = 0; i < 8; i++) {
          rows[i] = (uint8_t)fields[i];
        }
        ok = ledBoardSetFrame(rows);
      }
    } else if (lineBuf[0] == 'R' && lineBuf[1] == ':') {
      int y, r, g, b;
      if (sscanf(lineBuf + 2, "%d,%d,%d,%d", &y, &r, &g, &b) == 4) {
        ok = ledBoardSetRow(y, (uint8_t)r, (uint8_t)g, (uint8_t)b);
      }
    } else if (lineBuf[0] == 'P' && lineBuf[1] == ':') {
      int x, y, r, g, b;
      if (sscanf(lineBuf + 2, "%d,%d,%d,%d,%d", &x, &y, &r, &g, &b) == 5) {
        ok = ledBoardSetPixelColor(x, y, (uint8_t)r, (uint8_t)g, (uint8_t)b);
      }
    } else {
      int x, y;
      if (sscanf(lineBuf, "%d,%d", &x, &y) == 2) {
        ok = ledBoardSetPixel(x, y);
      }
    }

    Serial.println(ok ? "OK" : "ERR");
  }
}
