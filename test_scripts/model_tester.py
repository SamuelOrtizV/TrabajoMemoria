import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from model import SimpleRNN, RacingDataset
import os
import torchvision.transforms.functional as F
from torchvision import transforms, models
import time
import pandas as pd

# Definición de parámetros
#pretrained_cnn = models.efficientnet_v2_s(weights='IMAGENET1K_V1')
pretrained_cnn = models.efficientnet_b0(weights='IMAGENET1K_V1') 
hidden_size = 256
output_size = 9
input_size = (3, 426, 240)
num_layers = 1
dropout = 0
bias = True

seq_len = 5
batch_size = 1
num_epochs = 20
learning_rate = 0.001

# Optimización y función de pérdida
criterion = nn.CrossEntropyLoss()

# Definir las rutas de los directorios de datos y de guardado de modelos
test_data_dir = r"C:\Users\PC\Documents\GitHub\TrabajoMemoria\validation_dataset"
model_path = r"C:\Users\PC\Documents\GitHub\TrabajoMemoria\trained_models\model_EffNet_b0_256_epoch_2.pth"

# Cargar el modelo guardado
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleRNN(pretrained_cnn, hidden_size, output_size, input_size, num_layers, dropout, bias)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Definir las transformaciones
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Cambia el tamaño de las imágenes a 72x128 (height x width)
    transforms.ToTensor(),         # Convierte las imágenes a tensores
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizar la imagen
])

# Preparar el conjunto de prueba
test_dataset = RacingDataset(data_dir = test_data_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Función para evaluar el modelo
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    start_time = time.time()
    sequence_buffer_test = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            print(f"Progreso: {(i+1)/len(test_loader)*100:.2f}%", end='\r')

            # Añadir la secuencia al buffer
            if len(sequence_buffer_test) < seq_len: # Si el buffer no está lleno aún, se añade la imagen al buffer y se continúa con la siguiente iteración
                sequence_buffer_test.append(images)
                continue

            # Eliminar la imagen más antigua del buffer
            sequence_buffer_test.pop(0)
            sequence_buffer_test.append(images)

            # Apilar las imágenes en una sola dimensión
            input_test_sequence = torch.stack(sequence_buffer_test) # (seq_len, batch_size, channels, height, width)

            # Cambiar la forma de input_sequence para que el batch size sea 1
            input_test_sequence = input_test_sequence.permute(1,0, 2, 3, 4) # (batch_size, seq_len, channels, height, width)


            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(input_test_sequence)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                # Todas las salidas
                #print(outputs)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                probabilities_list = probabilities.tolist()[0]
                rounded_probabilities = [round(prob, 2) for prob in probabilities_list]
                prediction = torch.argmax(outputs, dim=1).item()
                label = labels.item()
                print(f"{rounded_probabilities} {prediction} VS {label}")

                # Predicción y cálculo de precisión
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    
    end_time = time.time()-start_time
    end_time = time.strftime("%H:%M:%S", time.gmtime(end_time))
    accuracy = 100 * correct / total

    print(f'Accuracy of the model on the test set: {accuracy:.2f}%\n',
          f'Test Loss: {test_loss/len(test_loader):.4f}\n',
          f'Time: {end_time}')

# Ejecutar la evaluación
evaluate_model(model, test_loader)