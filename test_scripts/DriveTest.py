from game_control import PressKey, ReleaseKey, W, A, S, D
import time

# print countdown 5 seconds
for i in list(range(5))[::-1]:
    print(i + 1)
    time.sleep(1)

# Press 'W' key for 3 seconds
PressKey(W)
time.sleep(3)
ReleaseKey(W)
time.sleep(1)