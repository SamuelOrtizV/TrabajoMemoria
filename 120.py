import time

# this script shows a character in the console that moves back and forth with a certain frecuency

def move_character(freq = 3, length = 10):
    character = "O"
    position = 0
    direction = 1
    fraction_mov = 1 / (freq * length * 2)  # time to move one step

    while True:
        print("\r" + " " * position + character, end="")
        time.sleep(fraction_mov)
        position += direction

        if position == (length-1) or position == 0:
            direction *= -1

if __name__ == "__main__":
    move_character()
