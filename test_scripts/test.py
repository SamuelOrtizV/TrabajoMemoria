# escribe un codigo que imprima un numero cada segundo de forma incremental

import time

num = 0
while True:
    print(num, end="\r")
    time.sleep(1)
    num += 1