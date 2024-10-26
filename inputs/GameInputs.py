from .game_control import PressKey, ReleaseKey, W, A, S, D, CTRL, N, Y
import time

# Tiempo de espera entre acciones de virar. Mayores tiempos provocan giros más pronunciados y viceversa.
SLEEP_TIME = 0.09 # Debe ser menor que 1/FPS

def reset_environment():
    """Reinicia la simulación (puedes personalizar este método según el juego)"""
    # Lógica para reiniciar el entorno

    # PRESIONAR CTRL + N PARA REINICIAR EL ENTORNO
    PressKey(CTRL)
    PressKey(N)
    time.sleep(0.1)  # Esperar un poco para asegurar que se presionen las teclas
    ReleaseKey(N)
    ReleaseKey(CTRL)

    time.sleep(2)  # Esperar a que el entorno se reinicie

    # PRESIONAR CTRL + Y PARA INICIAR LA CARRERA
    PressKey(CTRL)
    PressKey(Y)
    time.sleep(0.1)  # Esperar un poco para asegurar que se presionen las teclas
    ReleaseKey(Y)
    ReleaseKey(CTRL)

    pass

# Definición de las decisiones posibles

# 0 = NONE
def none():
    """
    No se presiona ninguna tecla.
    """
    #ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(D)
# 1 = A
def move_left():
    """
    Se presiona la tecla 'A'.
    """
    PressKey(A)
    #ReleaseKey(W)
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
    #ReleaseKey(W)
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
    time.sleep(SLEEP_TIME)
    ReleaseKey(S)
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
    ReleaseKey(S)
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
    ReleaseKey(S)

def move(direction: int):
    """
    Mueve el vehículo en la dirección especificada.

    :param int direction: La dirección en la que se moverá el vehículo.
    """
    if direction == 0:
        none()
    elif direction == 1:
        move_left()
    elif direction == 2:
        move_right()
    elif direction == 3:
        move_forward()
    elif direction == 4:
        move_back()
    elif direction == 5:
        move_left_forward()
    elif direction == 6:
        move_left_back()        
    elif direction == 7:
        move_right_forward()        
    elif direction == 8:
        move_right_back()
    else:
        raise ValueError("La dirección debe estar entre 0 y 8")