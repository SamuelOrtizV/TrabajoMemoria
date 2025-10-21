"""Simple UDP listener for receiving Assetto Corsa telemetry.

Parses a text payload of comma-separated key:value pairs and returns a
dictionary with typed values used by the environment and reward code.
"""

import socket
import time


def udp_listener(udp_ip="127.0.0.1", udp_port=5005):
    """Listen for a single UDP message and parse telemetry values.

    Args:
        udp_ip: IP address to bind
        udp_port: UDP port to bind

    Returns:
        dict with keys: speed, rpms, gear, laps, track_position, tyres_out,
        car_damage (list of 4 floats), acc_x, transmitting (bool)
    """
    timeout = 0.5

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((udp_ip, udp_port))
    sock.settimeout(timeout)

    try:
        data, addr = sock.recvfrom(1024)
        message = data.decode()

        # Split key:value pairs
        parts = message.split(", ")
        speed = float(parts[0].split(": ")[1])
        rpms = float(parts[1].split(": ")[1])
        gear = int(parts[2].split(": ")[1])
        laps = int(parts[3].split(": ")[1])
        track_position = float(parts[4].split(": ")[1])
        tyres_out = int(parts[5].split(": ")[1])
        car_damage_str = parts[6].split(": ")[1]
        car_damage = [float(d) for d in car_damage_str.split("_")]
        acc_x = float(parts[7].split(": ")[1])

        # Rounding for stability/readability
        speed = round(speed, 2)
        rpms = round(rpms, 2)
        track_position = round(track_position, 5)
        car_damage = [round(d, 4) for d in car_damage]
        acc_x = round(acc_x, 2)

        variables = {
            "speed": speed,
            "rpms": rpms,
            "gear": gear,
            "laps": laps,
            "track_position": track_position,
            "tyres_out": tyres_out,
            "car_damage": car_damage,
            "acc_x": acc_x,
            "transmitting": True,
        }
    except socket.timeout:
        time.sleep(0.1)
        # Defaults if no data
        variables = {
            "speed": 0.0,
            "rpms": 0,
            "gear": 0,
            "laps": 0,
            "track_position": 0.0,
            "tyres_out": 0,
            "car_damage": [0.0, 0.0, 0.0, 0.0],
            "acc_x": 0.0,
            "transmitting": False,
        }
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    except Exception as e:
        print(f"{e}")

        # Defaults on error
        variables = {
            "speed": 0.0,
            "rpms": 0,
            "gear": 0,
            "laps": 0,
            "track_position": 0.0,
            "tyres_out": 0,
            "car_damage": [0.0, 0.0, 0.0, 0.0],
            "acc_x": 0.0,
            "transmitting": False
        }

    return variables


if __name__ == "__main__":
    print("Listening for UDP messages...\n")
    try:
        while True:
            variables = udp_listener()
            print(variables, "               ", end="\r")
    except KeyboardInterrupt:
        print("\n\nExiting...")