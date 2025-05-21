import subprocess
import random
import time
import json
import os
import copy

config_path = "c:/Users/PC/TmrlData/config/config.json"

# Define tu grid de hiperparámetros aquí
param_grid = {
    ("ALG", "LR_ACTOR"): [1e-5, 5e-5, 1e-4],
    ("ENV", "IMG_WIDTH"): [64, 128, 256],
    ("ENV", "RTGYM_CONFIG", "time_step_duration"): [0.05, 0.1, 0.2]
}

# Lanza el server en una ventana separada (solo una vez)
server_cmd = ['start', 'cmd', '/k', 'python', 'server.py']
subprocess.Popen(server_cmd, shell=True)
time.sleep(5)  # Espera a que el server inicie

# Número de experimentos a correr
num_experiments = 5

for i in range(num_experiments):
    
    # Carga la configuración base
    with open(config_path, "r") as f:
        config = json.load(f)

    # Modifica los parámetros seleccionados aleatoriamente
    run_name_parts = []
    for keys, values in param_grid.items():
        value = random.choice(values)
        # Navega por los niveles anidados
        d = config
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = value
        run_name_parts.append(f"{'_'.join(keys)}-{value}")

    # Cambia el RUN_NAME para reflejar la configuración
    config["RUN_NAME"] = f"exp_{i}_" + "_".join(run_name_parts)

    # Sobrescribe el archivo config.json
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Lanza trainer y worker en ventanas separadas
    trainer_cmd = ['start', 'cmd', '/k', f'python trainer.py --config {config_path}']
    worker_cmd = ['start', 'cmd', '/k', f'python worker.py --config {config_path}']

    subprocess.Popen(trainer_cmd, shell=True)
    time.sleep(2)  # Espera un poco antes de lanzar el worker
    subprocess.Popen(worker_cmd, shell=True)

    # Aquí puedes implementar tu lógica de parada (por ejemplo, esperar a que termine el experimento)
    # Por ahora, espera un tiempo fijo antes de lanzar el siguiente experimento
    time.sleep(60 * 60)  # Espera 1 hora (ajusta según tu criterio)

print("Todos los experimentos han sido lanzados.")