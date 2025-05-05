from UDP_listener import udp_listener
from xbox_controller_inputs import XboxControllerReader
from collections import deque
import time

frame_time = 0.05  # Tiempo de espera entre cada iteración (en segundos)
direction_threshold = 0.2
acc_x_threshold = 0.1  # Umbral para la aceleración lateral
buffer_duration = 0.5  # Duración del buffer en segundos (500ms)

# Calcula la cantidad máxima de elementos en el buffer
buffer_size = int(buffer_duration / frame_time)

# Inicializa el lector del controlador de Xbox
controller = XboxControllerReader(total_wait_secs=5)

# Buffers para guardar los datos de dirección y aceleración
steering_buffer = deque(maxlen=buffer_size)
acc_x_buffer = deque(maxlen=buffer_size)
block_buffer = deque(maxlen=buffer_size)

while True:
    try:
        # Lee el estado del controlador
        steering, throttle_brake = controller.read()

        # Lee los datos del UDP
        data = udp_listener()
        acc_x = data["acc_x"]

        # Agrega los datos al buffer
        steering_buffer.append(float(steering))
        acc_x_buffer.append(acc_x)

        # Calcula el promedio de los datos en el buffer
        avg_steering = sum(steering_buffer) / len(steering_buffer)
        avg_acc_x = sum(acc_x_buffer) / len(acc_x_buffer)

        # Evalúa el bloqueo lateral usando los promedios
        # Si se gira a la derecha y no hay aceleración lateral, o si se gira a la izquierda y no hay aceleración lateral, 
        # o si hay acelareción lateral y no hay giro en ese sentido, se considera que hay bloqueo lateral
        bloqueo_lateral = (avg_steering > direction_threshold and not (acc_x < -acc_x_threshold)) or \
                          (avg_steering < -direction_threshold and not (acc_x > acc_x_threshold)) or \
                          (acc_x > acc_x_threshold and avg_steering > 0.05) or \
                          (acc_x < -acc_x_threshold and avg_steering < -0.05)
        
        block_buffer.append(bloqueo_lateral)
        # Calcula el promedio de bloqueo lateral
        avg_block = sum(block_buffer) / len(block_buffer)
        # Si el promedio de bloqueo lateral es mayor a 0.5, se considera que hay bloqueo
        if avg_block > 0.5:
            bloqueo_lateral = True
        else:
            bloqueo_lateral = False

        print(f"Dirección Promedio: {avg_steering:.2f}, Acc X Promedio: {avg_acc_x:.2f}, Bloqueo Lateral: {bloqueo_lateral}   Daño: {data["car_damage"]}                        ", end="\r")

        # Espera un tiempo antes de la siguiente iteración
        time.sleep(frame_time)

    except KeyboardInterrupt:
        print("\nSaliendo...")
        break