#include "led_board_driver.h"

void setup() {
  Serial.begin(115200);
  ledBoardInit();
}

// Four line formats share this listener:
//   "x,y"                          -- light a single pixel (clears the rest)
//   "F:r0,r1,r2,r3,r4,r5,r6,r7"    -- light a whole frame at once, one byte
//                                     per row (bit x of row y = pixel x,y)
//   "R:y,r,g,b"                    -- light row y a solid color, leaving
//                                     every other row as-is
//   "P:x,y,r,g,b"                  -- light a single pixel that color
//                                     (clears the rest)
void loop() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    bool ok = false;

    if (line.startsWith("F:")) {
      int fields[8];
      int n = sscanf(line.c_str() + 2, "%d,%d,%d,%d,%d,%d,%d,%d",
                      &fields[0], &fields[1], &fields[2], &fields[3],
                      &fields[4], &fields[5], &fields[6], &fields[7]);
      if (n == 8) {
        uint8_t rows[8];
        for (int i = 0; i < 8; i++) {
          rows[i] = (uint8_t)fields[i];
        }
        ok = ledBoardSetFrame(rows);
      }
    } else if (line.startsWith("R:")) {
      int y, r, g, b;
      if (sscanf(line.c_str() + 2, "%d,%d,%d,%d", &y, &r, &g, &b) == 4) {
        ok = ledBoardSetRow(y, (uint8_t)r, (uint8_t)g, (uint8_t)b);
      }
    } else if (line.startsWith("P:")) {
      int x, y, r, g, b;
      if (sscanf(line.c_str() + 2, "%d,%d,%d,%d,%d", &x, &y, &r, &g, &b) == 5) {
        ok = ledBoardSetPixelColor(x, y, (uint8_t)r, (uint8_t)g, (uint8_t)b);
      }
    } else {
      int x, y;
      if (sscanf(line.c_str(), "%d,%d", &x, &y) == 2) {
        ok = ledBoardSetPixel(x, y);
      }
    }

    Serial.println(ok ? "OK" : "ERR");
  }
}
