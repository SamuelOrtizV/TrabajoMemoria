from game_control import PressKey, ReleaseKey, W, A, S, D
import time

# Tiempo de espera entre acciones de virar. Mayores tiempos provocan giros más pronunciados y viceversa.
SLEEP_TIME = 0.1 # Debe ser menor que 1/FPS

# Definición de las decisiones posibles

# 0 = NONE
def none():
    """
    No se presiona ninguna tecla.
    """
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(D)
# 1 = A
def move_left():
    """
    Se presiona la tecla 'A'.
    """
    PressKey(A)
    ReleaseKey(W)
    ReleaseKey(S)
    ReleaseKey(D)
    time.sleep(SLEEP_TIME)
    ReleaseKey(A)
# 2 = D
def move_right():
    """
    Se presiona la tecla 'D'.
    """
    PressKey(D)
    ReleaseKey(W)
    ReleaseKey(S)
    ReleaseKey(A)
    time.sleep(SLEEP_TIME)
    ReleaseKey(D)
# 3 = W
def move_forward():
    """
    Se presiona la tecla 'W'.
    """
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(D)
    #ReleaseKey(W)
# 4 = S
def move_back():
    """
    Se presiona la tecla 'S'.
    """
    PressKey(S)
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    #ReleaseKey(S)
# 5 = AW
def move_left_forward():
    """
    Se presionan las teclas 'A' y 'W'.
    """
    PressKey(A)
    PressKey(W)
    ReleaseKey(S)
    ReleaseKey(D)
    time.sleep(SLEEP_TIME)
    ReleaseKey(A)
    #ReleaseKey(W)
# 6 = AS
def move_left_back():
    """
    Se presionan las teclas 'A' y 'S'.
    """
    PressKey(A)
    PressKey(S)
    ReleaseKey(W)
    ReleaseKey(D)
    time.sleep(SLEEP_TIME)
    ReleaseKey(A)
    #ReleaseKey(S)
# 7 = DW
def move_right_forward():
    """
    Se presionan las teclas 'D' y 'W'.
    """
    PressKey(D)
    PressKey(W)
    ReleaseKey(S)
    ReleaseKey(A)
    time.sleep(SLEEP_TIME)
    ReleaseKey(D)
    #ReleaseKey(W)
# 8 = DS
def move_right_back():
    """
    Se presionan las teclas 'D' y 'S'.
    """
    PressKey(D)
    PressKey(S)
    ReleaseKey(W)
    ReleaseKey(A)
    time.sleep(SLEEP_TIME)
    ReleaseKey(D)
    #ReleaseKey(S)