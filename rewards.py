def calculate_reward(variables, rewards, previous_location):
    """Calcula la recompensa basándose en las variables del entorno
    Args:
        variables (dict): Diccionario con las variables del entorno
        rewards (dict): Diccionario con los pesos de las recompensas y penalizaciones
        previous_location (dict): Diccionario con la ubicación anterior del auto
    Returns:
        float: Recompensa calculada"""

    reward = 0.0

    # Recompensas
    if variables["laps"] > previous_location["previous_lap"]:  # Si se ha completado una vuelta
        reward += rewards["reward_laps_weight"]
        previous_location.update({"previous_checkpoint": 0.0, "previous_position": 0.0, "previous_lap": variables["laps"]})

    track_position = 0.0 if variables["track_position"] == 1.0 else variables["track_position"]

    position_difference = track_position - previous_location["previous_position"]
    checkpoint_difference = track_position - previous_location["previous_checkpoint"]

    reward += rewards["reward_speed_weight"] if variables["speed"] >= rewards["threshold_speed"] else -rewards["reward_speed_weight"]

    if checkpoint_difference > rewards["threshold_checkpoint"]:  # Si se ha avanzado en la pista
        print(f"Se ha alcanzado un checkpoint: {track_position}")
        reward += rewards["reward_track_position_weight"]
        previous_location["previous_checkpoint"] = track_position

    # Penalizaciones
    reward += rewards["penalty_tyres_out"] * variables["tyres_out"] if variables["tyres_out"] > 0 else 0  # Penalizar por ruedas salidas de la pista
    reward += rewards["penalty_car_damage"] if variables["car_damage"] > 0 else 0                         # Penalizar por daño al auto
    reward += rewards["penalty_low_rpms"] if variables["rpms"] < rewards["threshold_rpms"] else 0         # Penalizar por quedarse quieto
    reward += rewards["penalty_backwards"] if position_difference < 0 else 0                              # Penalizar por ir hacia atrás

    done = variables["tyres_out"] == 4 or variables["car_damage"] > 0  # El episodio termina si el auto está fuera de la pista o dañado

    return reward, done