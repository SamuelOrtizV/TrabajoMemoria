import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import *
import os
import torchvision.transforms.functional as F
from inputs.xbox_controller_emulator import XboxControllerEmulator
from inputs.GameInputs import reset_environment
from inputs.getkeys import key_check
from UDP_listener import udp_listener
from ScreenRecorder import *
from torchvision import transforms
import time
import torch.nn.functional as F
import threading
import sys

# Parámetros de captura de pantalla
screen_size = (1920, 1080)
full_screen = True
fps = 100 # HAY QUE MEDIR LA CAPACIDAD Y AJUSTAR ESTE VALOR

# Hiperparametros del modelo
name = "SACtest"
architecture = "CNN"    # "CNN", "CNN_RNN", "CNN_LSTM_STATE"
output_size = 2             # Giro y aceleración

# Opción para cargar un modelo entrenado
load_model = False                          # Cambia esto a True si deseas cargar un modelo entrenado
model_path = "./trained_models/model.pth"   # Ruta del modelo entrenado

# Hiperparametros de la CNN
cnn_name = "efficientnet_b0"#"efficientnet_v2_s", "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3"
input_size = (240, 135)     # 16:9 ratio
dropout = 0.5
bias = True                 # Si se desea usar bias en las capas convolucionales
cnn_train = True            # Si se desea entrenar la parte CNN

# Hiperparametros de SAC
learning_rate = 3e-4   # Tasa de aprendizaje para el optimizador
discount_factor = 0.99 # Factor de descuento para las recompensas futurasp
alpha = 0.2            # Parámetro de entropía para SAC (controla la exploración)
tau = 0.005            # Parámetro de actualización suave de las redes objetivo
batch_size = 120     # Tamaño de batch para actualizar el agente

# Parámetros de recompensas
rewards = {
    "reward_speed_weight": 0.1,             # Peso de la velocidad en la recompensa
    "reward_track_position_weight": 5,      # Peso por alcanzar un checkpoint 
    "reward_laps_weight": 500.0,            # Peso de las vueltas completadas DEBE SER ALTO
    "penalty_low_rpms": -0.2,               # Penalización por quedarse quieto
    "penalty_backwards": -0.5,              # Penalización por ir hacia atrás
    "penalty_tyres_out": -0.5,              # Penalización por salirse de la pista
    "penalty_car_damage": -2.0,
    "threshold_speed": 10.0,                # Velocidad mínima para recibir recompensa por velocidad
    "threshold_rpms": 2000.0,               # RPMs mínimas para no recibir recompensa negativa por quedarse quieto
    "threshold_checkpoint": 0.01            # Umbral de posición en la pista para recibir recompensa por posición
    }             # Penalización por dañar el auto

# Otras configuraciones
max_steps_per_episode = 20   # Máximo número de pasos por episodio
num_episodes = 500              # Número de episodios de entrenamiento
save_interval = 50              # Guardar el modelo cada 50 episodios
dots = ["   ", ".  ", ".. ", "...", " ..", "  ."]

# Inicializar el buffer de experiencia
buffer_capacity = 10000
replay_buffer = ReplayBuffer(buffer_capacity)

# Definir el directorio de guardado
save_dir = f"./trained_models/{name}"

# Definir el nombre del modelo a guardar
#model_name = f"{name}-{architecture}-{cnn_name}-{seq_len}-{input_size[0]}-{input_size[1]}-{output_size}-{hidden_size}-ep"
model_name ="test"
print(f"\n\n\nNombre del modelo:\n{model_name}")

os.makedirs(save_dir, exist_ok=True) # Crear directorio de guardado si no existe

# Variable global para controlar la interrupción del teclado
stop_event = threading.Event()
pause_event = threading.Event()

def key_detection():
    global stop_event
    while not stop_event.is_set():
        keys = key_check()
        if keys == "Q":
            stop_event.set()
        elif keys == "P":
            pause_event.clear() if pause_event.is_set() else pause_event.set()
            print(f"{'Reanudando' if not pause_event.is_set() else 'Pausando'} el modelo...                                                          ", end="\r")
            time.sleep(1)  # Evitar múltiples detecciones rápidas

# Función para actualizar los modelos
def update_models(batch):
    print("\nActualizando modelos...                                                                 ", end="\r")
    states, actions, rewards, next_states, dones = zip(*batch)

    # Crear nuevos tensores en lugar de modificar los existentes
    states = torch.cat(states).detach()
    actions = torch.cat(actions).detach()
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
    next_states = torch.cat(next_states).detach()
    dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

    # 1. Actualizar críticos
    with torch.no_grad():
        next_actions = actor(next_states)
        target_q1 = target_critic1(next_states, next_actions)
        target_q2 = target_critic2(next_states, next_actions)
        target_q = rewards + (discount_factor * torch.min(target_q1, target_q2) * (1 - dones))
        target_q = target_q.detach()  # Asegurarnos que target_q está desconectado del grafo

    # Actualizar primer crítico
    current_q1 = critic1(states, actions)
    critic1_loss = F.mse_loss(current_q1, target_q.detach())
    critic1_optimizer.zero_grad(set_to_none=True)  # Usar set_to_none=True es más eficiente
    critic1_loss.backward(retain_graph=True)
    critic1_optimizer.step()

    # Actualizar segundo crítico
    current_q2 = critic2(states, actions)
    critic2_loss = F.mse_loss(current_q2, target_q.detach())
    critic2_optimizer.zero_grad(set_to_none=True)
    critic2_loss.backward(retain_graph=True)
    critic2_optimizer.step()

    # Actualizar actor
    current_actions = actor(states)
    actor_loss = -critic1(states, current_actions).mean()
    
    actor_optimizer.zero_grad(set_to_none=True)
    actor_loss.backward()
    actor_optimizer.step()

    # Actualizar redes objetivo de forma segura
    with torch.no_grad():  # Evitar tracking de gradientes durante la actualización
        for target_param, param in zip(target_critic1.parameters(), critic1.parameters()):
            new_target_param = (1 - tau) * target_param.data + tau * param.data
            target_param.data.copy_(new_target_param)
            
        for target_param, param in zip(target_critic2.parameters(), critic2.parameters()):
            new_target_param = (1 - tau) * target_param.data + tau * param.data
            target_param.data.copy_(new_target_param)

    # Limpiar la memoria si es necesario
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
# Definir las transformaciones de las imagenes
transform = transforms.Compose([
    transforms.Resize((input_size)),  # Cambia el tamaño de las imágenes a (height x width)
    transforms.ToTensor(),         # Convierte las imágenes a tensores
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizar la imagen
])

# Inicializar el controlador del simulador
controller = XboxControllerEmulator()

# Definir la región de captura de pantalla
region = get_region(screen_size, full_screen)

# Activar dispositivo CUDA si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inicializar las redes del actor y crítico
actor =             Actor (cnn_name, output_size, (3, *input_size), dropout, bias, cnn_train).to(device)
critic1 =           Critic(cnn_name, output_size, (3, *input_size), dropout, bias, cnn_train).to(device)
critic2 =           Critic(cnn_name, output_size, (3, *input_size), dropout, bias, cnn_train).to(device)
target_critic1 =    Critic(cnn_name, output_size, (3, *input_size), dropout, bias, cnn_train).to(device)
target_critic2 =    Critic(cnn_name, output_size, (3, *input_size), dropout, bias, cnn_train).to(device)

# Copiar los pesos de los críticos a los críticos objetivo
target_critic1.load_state_dict(critic1.state_dict())
target_critic2.load_state_dict(critic2.state_dict())

# Optimización
actor_optimizer =   optim.Adam(actor.parameters()  , lr=learning_rate)
critic1_optimizer = optim.Adam(critic1.parameters(), lr=learning_rate)
critic2_optimizer = optim.Adam(critic2.parameters(), lr=learning_rate)

criterion = nn.MSELoss()

load_models(load_model, model_path, actor, critic1, critic2, target_critic1, target_critic2,
            actor_optimizer, critic1_optimizer, critic2_optimizer)

# Iniciar el hilo de detección de teclas
key_thread = threading.Thread(target=key_detection)
key_thread.start()
lock = threading.Lock()

# pause_event.set() # Pausar el modelo al inicio

latest_image = None

# Crear una cola para pasar las imágenes transformadas
def capture_and_transform(region, transform, device, stop_event, lock):
    global latest_image
    while not stop_event.is_set():
        # Capturar la pantalla
        start_time = time.time()
        preprocessed_img = capture_and_process(region, transform, device)

        # Actualizar la última imagen capturada
        # Usar el lock para evitar condiciones de carrera al actualizar latest_image
        with lock:
            latest_image = preprocessed_img        
            time.sleep(max(0, 1/fps - (time.time() - start_time))) # Esperar para mantener los FPS

# Iniciar el hilo de captura y transformación de imágenes
capture_thread = threading.Thread(target=capture_and_transform, args=(region, transform, device, stop_event, lock))
capture_thread.start()

print("Iniciando entrenamiento...")

episodio = 0

repetidas = {
    "Repetidas": 0,
    "No repetidas": 0
}

start_time = time.time()    # Tiempo de inicio del entrenamiento

# Iniciar el entrenamiento
try:
    while not stop_event.is_set(): #Ciclos de episodios

        if pause_event.is_set():
            controller.reset()
            print("Modelo pausado. Presione 'P' para reanudar.                                                                  ", end="\r")
            time.sleep(0.1)
            continue

        episodio += 1

        tiempos = []
        demoras = []

        # Reiniciar el entorno
        # reset_environment()
        previous_location = {
            "previous_checkpoint": 0.0,
            "previous_position": 0.0,
            "previous_lap": 0
        }        

        # time.sleep(1)  # Esperar un segundo para que el entorno se reinicie completamente
        episode_start_time = time.time()
        total_reward = 0
        steps = 0

        while (not stop_event.is_set()) and (steps < max_steps_per_episode): #Ciclos de pasos
            
            step_start_time = time.time()

            if pause_event.is_set():
                controller.reset()
                print("Modelo pausado. Presione 'P' para reanudar.                                                                  ", end="\r")
                time.sleep(0.1)
                continue

            # Capturar la pantalla  (Esta parte es muy CPU demandante, sobre todo aplicar las transformaciones) 

            """ img = capture_screen(region)          
            
            preprocessed_img = Image.fromarray(img.astype(np.uint8)).convert('RGB')# Convertir la imagen preprocesada a un objeto PIL y aplicar las transformaciones
            state = transform(preprocessed_img).unsqueeze(0).to(device)
            """      

            moving_dots = dots[steps % len(dots)]

            # Obtener el estado actual del entorno
            with lock:
                if latest_image is not None:
                    # Procesar la última imagen capturada
                    state = latest_image
                else:
                    time.sleep(0.01)  # Esperar un poco si la cola está vacía
                    continue

            # Habilitar precisión mixta y para usar la GPU para la inferencia
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                                     

                # Elegir acción basada en las características extraídas por la CNN
                action = actor(state)  # El actor toma el estado como entrada y produce una acción

                # Enviar la acción al entorno
                prediction = torch.clamp(action, min=-1.0, max=1.0).tolist()[0]  # Limitar los valores de la acción entre -1.0 y 1.0 y convertir a lista
                controller.steering(prediction[0])  # Enviar la acción de dirección al simulador
                controller.throttle_break(prediction[1])  # Enviar la acción de aceleración/freno al simulador

                # Obtener el siguiente estado del entorno
                with lock:
                    next_state = latest_image  # Obtener el siguiente estado del entorno

                # Telemetria del juego

                # variables = udp_listener() #Causa cuello de botella solo cuando no está recibiendo datos

                variables = { # Placeholder para las variables del entorno
                "speed": 0.0,
                "rpms": 0,
                "laps": 0,
                "track_position": 0.0,
                "tyres_out": 0,
                "car_damage": 0.0,
                "transmitting": False
                }

                """ if variables["transmitting"] == False: # Si no se reciben datos de telemetría, continuar con el siguiente paso
                    print("No se están recibiendo datos de telemetría, esperando...                                                   ", end="\r")
                    continue 
                                """
                # Calcular la recompensa
                reward, done = calculate_reward(variables, rewards, previous_location)  # Se calcula la recompensa basada en las variables del entorno

                # Almacenar la transición en el buffer de experiencia
                replay_buffer.push(state, action, reward, next_state, done)

                # Muestrear un batch del buffer y actualizar los modelos
                if len(replay_buffer) > batch_size:
                    #with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        batch = replay_buffer.sample(batch_size)
                        update_models(batch) 

            total_reward += reward
            steps += 1

            next_state = latest_image


            if torch.equal(state, next_state):                
                repetidas["Repetidas"] += 1
                
                #print("Estado actual igual al siguiente estado. Posible error en la captura de pantalla.                ", end="\r")
                #sys.exit()
            else:
                repetidas["No repetidas"] += 1

            # Condición para reiniciar el episodio si el auto está fuera de la pista o dañado
            """ if done:
                break """  # Termina el episodio si el auto está fuera de la pista o dañado           

            #print(f"{moving_dots} Episodio: {episodio} Recomensa acumulada: {total_reward:.2f} Predicción del modelo: Steering {prediction[0]:.2f} Throttle {prediction[1]:.2f} Duración: {step_time:.2f}", end="\r")
            # Esperar para mantener los FPS
            time.sleep(max(0, 1/fps - (time.time() - step_start_time)))

            step_time = time.time() - step_start_time
            acumulated_time = time.time() - episode_start_time
            print(f"Episodio: {episodio} FPS promedio: {int(steps/acumulated_time)} Duración: {step_time:.4f} FPS: {int(1/step_time)} Rep: {repetidas['Repetidas']}               ", end="\r")

        # Guardar el modelo cada cierto número de episodios
        """ if episodio % save_interval == 0:
            save_model(save_dir, model_name, actor, critic1, critic2, actor_optimizer, critic1_optimizer, critic2_optimizer)   """               


except KeyboardInterrupt:
    print("\nEntrenamiento interrumpido.                                                   ")
    """ except Exception as e:
    print(f"\nError: {e}") """
finally:
    print("\nEntrenamiento terminado.                                                            ")
    stop_event.set()
    # Limpiar y cerrar
    controller.reset()
    key_thread.join()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    print(repetidas)