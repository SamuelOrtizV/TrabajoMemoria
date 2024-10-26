import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import *
import os
import torchvision.transforms.functional as F
from inputs.xbox_controller_emulator import XboxControllerEmulator
from inputs.GameInputs import reset_environment
from UDP_listener import udp_listener
from ScreenRecorder import *
from torchvision import transforms, models
import time
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import random
from collections import deque
import torch.nn.functional as F

# Parámetros de captura de pantalla
screen_size = (1920, 1080)
full_screen = True
fps = 5 # HAY QUE MEDIR LA CAPACIDAD Y AJUSTAR ESTE VALOR

# Hiperparametros del modelo
name = "CNNminitest"
architecture = "CNN_RNN"    # "CNN", "CNN_RNN", "CNN_LSTM_STATE"
output_size = 2             # Giro y aceleración

# Opción para cargar un modelo entrenado
load_model = False                          # Cambia esto a True si deseas cargar un modelo entrenado
model_path = "./trained_models/model.pth"   # Ruta del modelo entrenado

# Hiperparametros de la CNN
cnn_name = "efficientnet_b0"#"efficientnet_v2_s", "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3"
input_size = (240, 135)     # 16:9 ratio
dropout = 0 # 0.5
bias = True                 # Si se desea usar bias en las capas convolucionales
cnn_train = True            # Si se desea entrenar la parte CNN

# Hiperparametros de la RNN/LSTM
hidden_size = 256           # Número de neuronas en la capa oculta de la RNN o LSTM 512 usado por Iker
num_layers = 1              # Número de capas en la RNN o LSTM
seq_len = 1                 # Número de imágenes a considerar en la secuencia

# Hiperparametros de SAC
learning_rate = 3e-4   # Tasa de aprendizaje para el optimizador
discount_factor = 0.99 # Factor de descuento para las recompensas futuras
alpha = 0.2            # Parámetro de entropía para SAC (controla la exploración)
tau = 0.005            # Parámetro de actualización suave de las redes objetivo
batch_size = 64        # Tamaño de batch para actualizar el agente

# Parámetros de recompensas
reward_speed_weight = 0.1           # Peso de la velocidad en la recompensa
reward_track_position_weight = 10.0  # Peso de la posición en la pista
reward_laps_weight = 100.0          # Peso de las vueltas completadas DEBE SER ALTO
penalty_tyres_out = -0.5            # Penalización por salirse de la pista
penalty_car_damage = -2.0           # Penalización por dañar el auto

# Otras configuraciones
max_steps_per_episode = 1000    # Máximo número de pasos por episodio
num_episodes = 500              # Número de episodios de entrenamiento
save_interval = 50              # Guardar el modelo cada 50 episodios

# Definir el directorio de guardado
save_dir = f"./trained_models/{name}"

# Definir el nombre del modelo a guardar
model_name = f"{name}-{architecture}-{cnn_name}-{seq_len}-{input_size[0]}-{input_size[1]}-{output_size}-{hidden_size}-ep"
print(f"{model_name}")

os.makedirs(save_dir, exist_ok=True) # Crear directorio de guardado si no existe

# Definir el escritor de TensorBoard para visualización
writer = SummaryWriter(log_dir="./runs/" + model_name) 

# Función para guardar el modelo
def save_model(episode, model_name):
    model_save_path = os.path.join(save_dir, model_name+f"{episode}.pth")
    torch.save({
        'actor': actor.state_dict(),
        'critic1': critic1.state_dict(),
        'critic2': critic2.state_dict(),
        'target_critic1': target_critic1.state_dict(),
        'target_critic2': target_critic2.state_dict(),
        'actor_optimizer': actor_optimizer.state_dict(),
        'critic1_optimizer': critic1_optimizer.state_dict(),
        'critic2_optimizer': critic2_optimizer.state_dict(),
    }, model_save_path)
    print(f"Modelo guardado en el episodio {episode}", end="\r")

# Función para calcular la recompensa en cada paso
def calculate_reward(variables):
    """Calcula la recompensa basándose en las variables del entorno"""

    speed = variables["speed"]
    track_position = variables["track_position"]
    laps = variables["laps"]
    tyres_out = variables["tyres_out"]
    car_damage = variables["car_damage"]

    speed_threshold = 10.0  # Velocidad mínima en km/h para recibir recompensa por velocidad no nula
    
    if speed < speed_threshold:
        reward_speed_weight = -reward_speed_weight  # No recompensar por velocidad si es menor al umbral

    # Recompensa por velocidad y posición en la pista
    reward = reward_speed_weight + reward_track_position_weight * track_position + reward_laps_weight * laps

    # Penalización por salirse de la pista y por daño
    if tyres_out > 0:
        reward += penalty_tyres_out * tyres_out
    if car_damage > 0:
        reward += penalty_car_damage

    return reward

# Definir las transformaciones de las imagenes
transform = transforms.Compose([
    transforms.Resize((input_size)),  # Cambia el tamaño de las imágenes a (height x width)
    transforms.ToTensor(),         # Convierte las imágenes a tensores
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizar la imagen
])

# Inicializar el controlador del simulador
if output_size == 2:
    controller = XboxControllerEmulator()
    print("Modelo de controlador cargado.")
else:
    raise ValueError("El tamaño de salida del modelo debe ser 2 (control)")

# Definir la región de captura de pantalla
region = get_region(screen_size, full_screen)

# Activar dispositivo CUDA si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando CUDA" if torch.cuda.is_available() else "USANDO CPU")

# Inicializar las redes del actor y crítico
actor =             Actor (cnn_name, output_size, input_size, dropout, bias, cnn_train).to(device)
critic1 =           Critic(cnn_name, output_size, input_size, dropout, bias, cnn_train).to(device)
critic2 =           Critic(cnn_name, output_size, input_size, dropout, bias, cnn_train).to(device)
target_critic1 =    Critic(cnn_name, output_size, input_size, dropout, bias, cnn_train).to(device)
target_critic2 =    Critic(cnn_name, output_size, input_size, dropout, bias, cnn_train).to(device)

# Copiar los pesos de los críticos a los críticos objetivo
target_critic1.load_state_dict(critic1.state_dict())
target_critic2.load_state_dict(critic2.state_dict())

# Optimización
actor_optimizer =   optim.Adam(actor.parameters()  , lr=learning_rate)
critic1_optimizer = optim.Adam(critic1.parameters(), lr=learning_rate)
critic2_optimizer = optim.Adam(critic2.parameters(), lr=learning_rate)

criterion = nn.MSELoss()

if load_model and os.path.exists(model_path):
    checkpoint = torch.load(model_path)
    actor.load_state_dict(checkpoint['actor'])
    critic1.load_state_dict(checkpoint['critic1'])
    critic2.load_state_dict(checkpoint['critic2'])
    target_critic1.load_state_dict(checkpoint['target_critic1'])
    target_critic2.load_state_dict(checkpoint['target_critic2'])
    actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
    critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer'])
    critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])
    print("Modelo cargado exitosamente.")
else:
    print("No se cargó ningún modelo. Entrenamiento desde cero.")

print("Iniciando entrenamiento...")

finished = False
episodio = 0
sequence_buffer = []  # Buffer para almacenar las imágenes de la secuencia en caso de usar RNN o LSTM

start_time = time.time()    # Tiempo de inicio del entrenamiento

# Iniciar el entrenamiento

try:
    while not finished: #Ciclos de episodios
        print(f"Episodio {episodio}", end="\r")
        episodio += 1

        # Reiniciar el entorno
        reset_environment()
        episode_start_time = time.time()
        total_reward = 0

        while True: #Ciclos de pasos
            step_start_time = time.time()
            # Capturar la pantalla   
            img = capture_screen(region)           
            preprocessed_img = Image.fromarray(img.astype(np.uint8)).convert('RGB')# Convertir la imagen preprocesada a un objeto PIL y aplicar las transformaciones
            preprocessed_img = transform(preprocessed_img)

            # Telemetria del juego
            variables = udp_listener()

            # Añadir la imagen preprocesada al buffer de secuencia
            sequence_buffer.append(preprocessed_img)
        
            # Mantener solo las últimas imágenes en el buffer
            if len(sequence_buffer) > seq_len:
                sequence_buffer.pop(0)

            # Verificar si tenemos suficientes imágenes para una secuencia completa
            if len(sequence_buffer) == seq_len:
                # Convertir la secuencia de imágenes en un tensor
                sequence_tensor = torch.stack(sequence_buffer).unsqueeze(0).to(device)

                # Habilitar precisión mixta y para usar la GPU para entrenar
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    if architecture == "CNN":
                        state = sequence_tensor
                        """ elif architecture == "CNN_RNN":
                        outputs = model(sequence_tensor)
                        state = sequence_tensor.view(-1) # Convertir la secuencia en un tensor 1D """
                    else:
                        print("Arquitectura no soportada")
                        #parar la ejecucion de todo el programa
                        raise SystemExit

                    # Elegir acción basada en las características extraídas por la CNN
                    action = actor(state)  # El actor toma el estado como entrada y produce una acción

                    # Aquí deberías obtener el siguiente estado del entorno
                    next_state = state  # En este caso, se asume que el siguiente estado es el mismo que el actual

                    # Calcular la recompensa
                    reward = calculate_reward(variables)  # Se calcula la recompensa basada en las variables del entorno

                    # Verificar si el episodio ha terminado
                    done = variables["tyres_out"] == 4 or variables["car_damage"] > 0  # El episodio termina si el auto está fuera de la pista o dañado

                    # Enviar la acción al simulador
                    prediction = torch.clamp(action, min=-1.0, max=1.0).tolist()[0]  # Limitar los valores de la acción entre -1.0 y 1.0 y convertir a lista
                    controller.steering(prediction[0])  # Enviar la acción de dirección al simulador
                    controller.throttle_break(prediction[1])  # Enviar la acción de aceleración/freno al simulador

                    # Calcular la pérdida para SAC
                    with torch.no_grad():  # Deshabilitar el cálculo de gradientes porque no necesitamos actualizar las redes objetivo
                        next_action = actor(next_state)  # El actor toma el siguiente estado y produce la siguiente acción
                        target_q1 = target_critic1(torch.cat([next_state, next_action], dim=-1))  # El crítico objetivo 1 calcula el valor Q para el siguiente estado y acción
                        target_q2 = target_critic2(torch.cat([next_state, next_action], dim=-1))  # El crítico objetivo 2 calcula el valor Q para el siguiente estado y acción
                        target_q = reward + discount_factor * torch.min(target_q1, target_q2) * (1 - done)  # Calcular el valor Q objetivo usando la recompensa y el valor Q mínimo de los críticos objetivos

                    # Calcular los valores Q actuales
                    current_q1 = critic1(torch.cat([state, action], dim=-1))  # El crítico 1 calcula el valor Q para el estado y acción actuales
                    current_q2 = critic2(torch.cat([state, action], dim=-1))  # El crítico 2 calcula el valor Q para el estado y acción actuales

                    # Calcular la pérdida de los críticos
                    critic1_loss = F.mse_loss(current_q1, target_q)  # Calcular la pérdida del crítico 1 como el error cuadrático medio entre el valor Q actual y el objetivo
                    critic2_loss = F.mse_loss(current_q2, target_q)  # Calcular la pérdida del crítico 2 como el error cuadrático medio entre el valor Q actual y el objetivo

                    # Optimización del crítico 1
                    critic1_optimizer.zero_grad()  # Reiniciar los gradientes del optimizador del crítico 1
                    critic1_loss.backward()  # Calcular los gradientes de la pérdida del crítico 1
                    critic1_optimizer.step()  # Actualizar los parámetros del crítico 1

                    # Optimización del crítico 2
                    critic2_optimizer.zero_grad()  # Reiniciar los gradientes del optimizador del crítico 2
                    critic2_loss.backward()  # Calcular los gradientes de la pérdida del crítico 2
                    critic2_optimizer.step()  # Actualizar los parámetros del crítico 2

                    # Calcular la pérdida del actor
                    actor_loss = -critic1(state, actor(state)).mean()  # Calcular la pérdida del actor como el negativo del valor Q promedio estimado por el crítico 1

                    # Optimización del actor
                    actor_optimizer.zero_grad()  # Reiniciar los gradientes del optimizador del actor
                    actor_loss.backward()  # Calcular los gradientes de la pérdida del actor
                    actor_optimizer.step()  # Actualizar los parámetros del actor

                    # Actualizar las redes objetivo
                    for target_param, param in zip(target_critic1.parameters(), critic1.parameters()):  # Para cada par de parámetros de los críticos objetivo y crítico 1
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)  # Actualización suave de los parámetros del crítico objetivo 1

                    for target_param, param in zip(target_critic2.parameters(), critic2.parameters()):  # Para cada par de parámetros de los críticos objetivo y crítico 2
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)  # Actualización suave de los parámetros del crítico objetivo 2

            total_reward += reward

            # Condición para reiniciar el episodio si el auto está fuera de la pista o dañado
            if variables["tyres_out"] == 4 or variables["car_damage"] > 0:
                print("El auto se ha salido de la pista o ha sufrido daño, reiniciando episodio...")
                break  # Termina el episodio si el auto está fuera de la pista o dañado

            # EN ESTA PARTE PONER UN SLEEP PARA HACER QUE LOS STEPS ESTEN ESPACIADOS DE FORMA CONSTANTE
            time.sleep(max(0, 1/fps - (time.time() - step_start_time)))

        # Guardar el modelo cada cierto número de episodios
        if episodio % save_interval == 0:
            save_model(episodio, model_name)

except KeyboardInterrupt:
    print("\nEntrenamiento interrumpido.")

print("Entrenamiento terminado.")

writer.close()  # Cerrar el escritor de TensorBoard

torch.cuda.empty_cache()
torch.cuda.ipc_collect()