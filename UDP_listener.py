import socket

def udp_listener(udp_ip="127.0.0.1", udp_port=5005):
    """
    Escucha mensajes UDP y devuelve las variables obtenidas.

    Args:
        udp_ip (str): Dirección IP para escuchar.
        udp_port (int): Puerto UDP para escuchar.

    Returns:
        dict: Diccionario con las variables obtenidas.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((udp_ip, udp_port))

    while True:
        data, addr = sock.recvfrom(1024)  # Tamaño del buffer
        message = data.decode()
        
        # Separar los datos en variables individuales
        parts = message.split(", ")
        speed = float(parts[0].split(": ")[1])
        laps = int(parts[1].split(": ")[1])
        track_position = float(parts[2].split(": ")[1])
        tyres_out = int(parts[3].split(": ")[1])
        car_damage = float(parts[4].split(": ")[1])
        
        # Redondear los valores a 2 decimales
        speed = round(speed, 2)
        track_position = round(track_position, 2)
        car_damage = round(car_damage, 2)
        
        # Almacenar los valores en un diccionario
        variables = {
            "speed": speed,
            "laps": laps,
            "track_position": track_position,
            "tyres_out": tyres_out,
            "car_damage": car_damage
        }
        
        # Devolver el diccionario con las variables
        return variables

# Ejemplo de uso
if __name__ == "__main__":
    variables = udp_listener()
    print(variables)

""" import socket

UDP_IP = "127.0.0.1"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

while True:
    data, addr = sock.recvfrom(1024)  # Tamaño del buffer
    message = data.decode()
    
    # Separar los datos en variables individuales
    parts = message.split(", ")
    speed = float(parts[0].split(": ")[1])
    laps = int(parts[1].split(": ")[1])
    track_position = float(parts[2].split(": ")[1])
    tyres_out = int(parts[3].split(": ")[1])
    car_damage = float(parts[4].split(": ")[1])
    
    # Redondear los valores a 2 decimales
    speed = round(speed, 2)
    track_position = round(track_position, 2)
    car_damage = round(car_damage, 2)
    
    # Imprimir los valores en la consola
    print(f"Speed: {speed}, Laps: {laps}, Track Position: {track_position}, Tyres Out: {tyres_out}, Car Damage: {car_damage}", end='\r') """