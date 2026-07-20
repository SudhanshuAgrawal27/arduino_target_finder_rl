import time

from led_board_client import connect, set_led

if __name__ == "__main__":
    ser = connect()
    for x in range(8):
        for y in range(8):
            reply = set_led(ser, x, y)
            print(f"({x},{y}) -> {reply}")
            time.sleep(0.1)
    ser.close()
