import socket
import time

def udp_listener(udp_ip="127.0.0.1", udp_port=5005):
    """
    Escucha mensajes UDP y devuelve las variables obtenidas.

    Args:
        udp_ip (str): Dirección IP para escuchar.
        udp_port (int): Puerto UDP para escuchar.

        timeout (float): Tiempo de espera en segundos para recibir datos.

    Returns:
        variables (dict): Diccionario con las variables obtenidas.
            
                - speed (float): Velocidad del vehículo.
                - rpms (float): Revoluciones por minuto del motor.
                - gear (int): Marcha actual del vehículo.
                - laps (int): Número de vueltas completadas.
                - track_position (float): Posición en la pista.
                - tyres_out (int): Número de ruedas fuera de la pista.
                - car_damage (float): Daño del vehículo.
                - acc_x (float): Aceleración en el eje X. Negativo giro a la derecha, positivo giro a la izquierda.
                - transmitting (bool): Indica si se están transmitiendo datos.
        
    """
    timeout=0.5

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((udp_ip, udp_port))
    sock.settimeout(timeout)  # Configurar el tiempo de espera

    try:
        #print("1", end='\r')
        data, addr = sock.recvfrom(1024)  # Tamaño del buffer
        message = data.decode()

        # Separar los datos en variables individuales
        parts = message.split(", ")
        speed = float(parts[0].split(": ")[1])
        rpms = float(parts[1].split(": ")[1])
        gear = int(parts[2].split(": ")[1])
        laps = int(parts[3].split(": ")[1])
        track_position = float(parts[4].split(": ")[1])
        tyres_out = int(parts[5].split(": ")[1])
        car_damage = float(parts[6].split(": ")[1])
        acc_x = float(parts[7].split(": ")[1])
        
        # Redondear los valores a 2 decimales
        speed = round(speed, 2)
        rpms = round(rpms, 2)
        track_position = round(track_position, 5)
        car_damage = round(car_damage, 4)
        acc_x = round(acc_x, 2)
        
        # Almacenar los valores en un diccionario
        variables = {
            "speed": speed,
            "rpms": rpms,
            "gear": gear,
            "laps": laps,
            "track_position": track_position,
            "tyres_out": tyres_out,
            "car_damage": car_damage,
            "acc_x": acc_x,
            "transmitting": True
        }
    except socket.timeout:
        time.sleep(0.1)
        # Valores por defecto si no se reciben datos
        variables = {
            "speed": 0.0,
            "rpms": 0,
            "gear": 0,
            "laps": 0,
            "track_position": 0.0,
            "tyres_out": 0,
            "car_damage": 0.0,
            "acc_x": 0.0,
            "transmitting": False
        }
    except KeyboardInterrupt:
        #print("3")
        raise KeyboardInterrupt
    except Exception as e:
        print(f"{e}")

        # Valores por defecto en caso de error
        variables = {
            "speed": 0.0,
            "rpms": 0,
            "gear": 0,
            "laps": 0,
            "track_position": 0.0,
            "tyres_out": 0,
            "car_damage": 0.0,
            "acc_x": 0.0,
            "transmitting": False
        }
        
    # Devolver el diccionario con las variables
    return variables

# Ejemplo de uso
if __name__ == "__main__":
    print("Escuchando mensajes UDP...\n")
    try:
        while True:
            variables = udp_listener()
            print(variables, "               ", end='\r')
    except KeyboardInterrupt:
        print("\n\nSaliendo...")