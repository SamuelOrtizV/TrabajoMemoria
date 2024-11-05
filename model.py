import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
from collections import deque
from ScreenRecorder import *
import random
    
class CNN(nn.Module):
    """
    Modelo de red neuronal convolucional simple que utiliza una CNN preentrenada para extraer características de las imágenes
    y una capa completamente conectada para la clasificación final.

    Args:
        cnn_name (str): Nombre de la CNN a utilizar. Opciones: 'resnet18', 'efficientnet-b0', 'vgg11', etc.
        output_size (int): Tamaño de la salida del modelo.
        input_size (tuple, optional): Tamaño de la entrada de las imágenes (canales, altura, ancho). Máximo tamaño es 384x384.
        dropout (float, optional): Probabilidad de dropout.
        bias (bool, optional): Si se incluye sesgo en las capas lineales.
    """
    def __init__(self, cnn_name, input_size=(3, 224, 224), dropout=0, bias=True, cnn_train=True):
        super(CNN, self).__init__()
        self.cnn_name = cnn_name
        self.input_size = input_size
        self.dropout = dropout
        self.bias = bias
        
        # Obtener la CNN preentrenada
        self.cnn = self.get_cnn()

        # Usar el modelo preentrenado
        self.cnn.classifier = nn.Identity()  # Quitar la última capa de clasificación
        
        if not cnn_train:
            for param in self.cnn.parameters():  # Desactivar el gradiente para las capas convolucionales preentrenadas
                param.requires_grad = False
        
        # Obtener el tamaño de la salida de la CNN
        self.conv_output_size = self._get_conv_output_size()

        if dropout > 0:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = None
    
    def get_cnn(self):
        # Obtener la clase de la CNN
        cnn_call_method = getattr(models, self.cnn_name)
        # Instanciar el modelo
        cnn_model = cnn_call_method(weights='IMAGENET1K_V1')      

        return cnn_model
    
    def _get_conv_output_size(self):
        # Crear una entrada dummy para pasarla a través de la CNN
        dummy_input = torch.zeros(1, *self.input_size)  # Ajustar según la entrada esperada por la CNN
        output = self.cnn(dummy_input)
        return int(torch.prod(torch.tensor(output.shape[1:])))
    
    def forward(self, x):
        """ batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)
        
        cnn_output = self.cnn(x)
        cnn_output = cnn_output.view(batch_size, seq_len, -1)
        
        if self.dropout_layer:
            cnn_output = self.dropout_layer(cnn_output)
        
        final_out = cnn_output.view(batch_size, -1)
        return final_out """
        cnn_output = self.cnn(x)
        
        if self.dropout_layer:
            cnn_output = self.dropout_layer(cnn_output)
        
        final_out = cnn_output.view(cnn_output.size(0), -1)
        return final_out


# Definir las redes del actor y crítico
class Actor(nn.Module):
    def __init__(self, cnn_name, output_size, input_size=(3, 224, 224), dropout=0.5, bias=True, cnn_train=True):
        super(Actor, self).__init__()
        self.cnn = CNN(cnn_name, input_size, dropout, bias, cnn_train)
        conv_output_size = self.cnn.conv_output_size
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, 256)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(256, output_size)
    
    def forward(self, x):
        x = self.cnn(x)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return torch.tanh(self.fc3(x))

class Critic(nn.Module):
    def __init__(self, cnn_name, output_size, input_size=(3, 224, 224), dropout=0.5, bias=True, cnn_train=True):
        super(Critic, self).__init__()
        self.cnn = CNN(cnn_name, input_size, dropout, bias, cnn_train)
        conv_output_size = self.cnn.conv_output_size
        self.fc1 = nn.Linear(conv_output_size + output_size, 256)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, 256)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(256, 1)
    
    def forward(self, state, action): #Revisar si esto es optimo
        state_features = self.cnn(state)
        state_features_flat = state_features.view(state_features.size(0), -1)  # Aplanar el tensor de estado VER SI HAY FORMA MAS EFICIENTE DE HACERLO
        action_flat = action.view(action.size(0), -1)  # Aplanar el tensor de acción
        x = torch.cat([state_features_flat, action_flat], dim=-1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)
    
# Definir el buffer de experiencia
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size) # Es una forma de data shuffling

    def __len__(self):
        return len(self.buffer)
    
# Función para calcular la recompensa en cada paso
def calculate_reward(variables, rewards, previous_location):
    """Calcula la recompensa basándose en las variables del entorno
    Args:
        variables (dict): Diccionario con las variables del entorno
        rewards (dict): Diccionario con los pesos de las recompensas y penalizaciones
        previous_location (dict): Diccionario con la ubicación anterior del auto
    Returns:
        float: Recompensa calculada"""

    speed = variables["speed"]
    rpms = variables["rpms"]
    track_position = variables["track_position"]
    laps = variables["laps"]
    tyres_out = variables["tyres_out"]
    car_damage = variables["car_damage"]

    reward_speed_weight = rewards["reward_speed_weight"]
    reward_track_position_weight = rewards["reward_track_position_weight"]
    reward_laps_weight = rewards["reward_laps_weight"]
    penalty_low_rpms = rewards["penalty_low_rpms"]
    penalty_backwards = rewards["penalty_backwards"]
    penalty_tyres_out = rewards["penalty_tyres_out"]
    penalty_car_damage = rewards["penalty_car_damage"]
    threshold_speed = rewards["threshold_speed"]
    threshold_rpms = rewards["threshold_rpms"]
    threshold_checkpoint = rewards["threshold_checkpoint"]

    previous_checkpoint = previous_location["previous_checkpoint"]
    previous_position = previous_location["previous_position"]
    previous_lap = previous_location["previous_lap"]

    reward = 0.0
    # Recompensas
    if laps > previous_lap: # Si se ha completado una vuelta
        reward += reward_laps_weight
        previous_location["previous_checkpoint"] = 0.0
        previous_location["previous_position"] = 0.0
        previous_location["previous_lap"] = laps

    if track_position == 1.0: # La posición en la pista es 1.0 cuando inicia
        track_position = 0.0

    position_difference = track_position - previous_position
    checkpoint_difference = track_position - previous_checkpoint
    
    
    if speed < threshold_speed:
        reward -= reward_speed_weight  # No recompensar por velocidad si es menor al umbral
    else:
        reward += reward_speed_weight

    if checkpoint_difference > threshold_checkpoint: # Si se ha avanzado en la pista
        print(f"Se ha alcanzado un checkpoint: {track_position}")
        reward += reward_track_position_weight
        previous_checkpoint = track_position

    # Penalizaciones
    reward += penalty_tyres_out * tyres_out if tyres_out > 0 else 0 # Penalizar por ruedas salidas de la pista
    reward += penalty_car_damage if car_damage > 0 else 0           # Penalizar por daño al auto
    reward += penalty_low_rpms if rpms < threshold_rpms else 0      # Penalizar por quedarse quieto
    reward += penalty_backwards if position_difference < 0 else 0   # Penalizar por ir hacia atrás

    done = tyres_out == 4 or car_damage > 0  # El episodio termina si el auto está fuera de la pista o dañado 

    return reward, done

def save_model(actor, critic1, critic2, target_critic1, target_critic2,
               actor_optimizer, critic1_optimizer, critic2_optimizer,
               episode, model_name, save_dir):
    model_save_path = os.path.join(save_dir, f"{model_name}_{episode}.pth")
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
    print(f"Modelo guardado en el episodio {episode} en {model_save_path}")

def load_models(load_model, model_path, actor, critic1, critic2, target_critic1, target_critic2,
                actor_optimizer, critic1_optimizer, critic2_optimizer):
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

def capture_and_process(region, transform, device):
    img = capture_screen(region)
    preprocessed_img = Image.fromarray(img.astype(np.uint8)).convert('RGB')
    return transform(preprocessed_img).unsqueeze(0).to(device)