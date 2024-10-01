import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from model import SimpleRNN, RacingDataset
import os
import torchvision.transforms.functional as F
from torchvision import transforms, models
import time
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

# Definición de parámetros
#pretrained_cnn = models.efficientnet_v2_s(weights='IMAGENET1K_V1') 
pretrained_cnn = models.efficientnet_b0(weights='IMAGENET1K_V1') 
hidden_size = 256 # Número de neuronas en la capa oculta 512 usado por Iker
output_size = 9 # Número de clases (W, A, S, D, WA, WD, SA, SD, NONE)
input_size = (3, 224, 224)  # 16:9 ratio
num_layers = 1
dropout = 0 # Se necesita num_layers > 1 para que el dropout tenga efecto
bias = True
cnn_train = True # Si se entrena la parte CNN o no

seq_len = 5 # Número de imágenes a considerar en la secuencia
batch_size = 16 # Número de secuencias a considerar en paralelo
num_epochs = 20 # Número de veces que se recorrerá el dataset
learning_rate = 0.001 

# Definir las rutas de los directorios de datos y de guardado de modelos
train_data_dir = "./datasets/train_dataset"
validation_data_dir = "./datasets/validation_dataset"
save_dir = "./trained_models"

model_name = "Oval_model_EffNet_b0_256_epoch"

os.makedirs(save_dir, exist_ok=True)

# Definir el escritor de TensorBoard para visualización
writer = SummaryWriter(log_dir="./runs/exp2") 

# Inicializar el modelo
model = SimpleRNN(pretrained_cnn, hidden_size, output_size, input_size, num_layers, dropout, bias, cnn_train)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Definir las transformaciones
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Cambia el tamaño de las imágenes a 72x128 (height x width)
    transforms.ToTensor(),         # Convierte las imágenes a tensores
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizar la imagen
])

# Cargar los datasets
train_dataset = RacingDataset(data_dir= train_data_dir, transform=transform)
test_dataset = RacingDataset(data_dir= validation_data_dir, transform=transform)

# Crear los dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
validation_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Optimización y función de pérdida
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

print("Iniciando entrenamiento...")

start_time = time.time()    # Tiempo de inicio del entrenamiento

# Ciclo de entrenamiento
for epoch in range(num_epochs): 
    
    # Iniciar buffer de secuencia
    sequence_buffer_train = []
    sequence_buffer_validation = []

    predictions_list = [] 
    labels_list = [] 
    T_o_F_list = []
    
    model.train()
    running_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):

        print(f"Trabajando en Epoch {epoch+1}. Progreso: {(i+1)/len(train_loader)*100:.2f}%", end='\r')       

        # Añadir la secuencia al buffer
        if len(sequence_buffer_train) < seq_len: # Si el buffer no está lleno aún, se añade la imagen al buffer y se continúa con la siguiente iteración
            sequence_buffer_train.append(images)
            continue

        # Eliminar la imagen más antigua del buffer
        sequence_buffer_train.pop(0)
        sequence_buffer_train.append(images)        

        # Apilar las imágenes en una sola dimensión
        input_sequence = torch.stack(sequence_buffer_train) # (seq_len, batch_size, channels, height, width)

        # Cambiar la forma de input_sequence para que el batch size sea 1
        input_sequence = input_sequence.permute(1,0, 2, 3, 4) # (batch_size, seq_len, channels, height, width)
        
        input_sequence, labels = input_sequence.to(device), labels.to(device) # Mover los datos al dispositivo
        
        optimizer.zero_grad()
        
        # Habilitar precisión mixta
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(input_sequence)
            loss = criterion(outputs, labels) # Solo se considera la última etiqueta de la secuencia
        
        # Backpropagation y optimización
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

        # Registrar las predicciones para la matriz de confusión
        _, predicted = torch.max(outputs, 1)
        predictions_list.append(predicted.cpu().numpy())
        labels_list.append(labels.cpu().numpy())
        T_o_F_list.append(labels.cpu().numpy() == predicted.cpu().numpy())
    
    # Guardar las predicciones en un archivo CSV
    predictions_df = pd.DataFrame({
    'Labels': labels_list,
    'Predictions': predictions_list,
    'T o F': T_o_F_list
    })

    predictions_df.to_csv(f'predictions_{model_name}_{epoch+1}.csv', index=False)

    # Registrar la pérdida en TensorBoard
    writer.add_scalar('Loss/train', running_loss / len(train_loader), epoch)

    # Guardar el modelo después de cada epoch
    torch.save(model.state_dict(), os.path.join(save_dir, f'{model_name}_{epoch+1}.pth'))

    print("\nValidando modelo...")


    # Validación en el conjunto de datos de prueba
    model.eval()  # Establecer el modo de evaluación
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(validation_loader):
            
            print(f"Validando en Epoch {epoch+1}. Progreso: {(i+1)/len(validation_loader)*100:.2f}%", end='\r')
            
            # Añadir la secuencia al buffer
            if len(sequence_buffer_validation) < seq_len:
                sequence_buffer_validation.append(images)
                continue

            # Eliminar la imagen más antigua del buffer
            sequence_buffer_validation.pop(0)
            sequence_buffer_validation.append(images)

            # Apilar las imágenes en una sola dimensión
            input_test_sequence = torch.stack(sequence_buffer_validation) # (seq_len, batch_size, channels, height, width)

            # Cambiar la forma de input_sequence para que el batch size sea 1
            input_test_sequence = input_test_sequence.permute(1,0, 2, 3, 4) # (batch_size, seq_len, channels, height, width)

            input_test_sequence, labels = input_test_sequence.to(device), labels.to(device) # Mover los datos al dispositivo

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(input_test_sequence)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                # Predicción y cálculo de precisión
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    writer.add_scalar('Loss/test', test_loss / len(validation_loader), epoch)
    writer.add_scalar('Accuracy/test', accuracy, epoch)

    end_time = time.time() - start_time  # Tiempo de finalización del epoch
    # Convertir end_time a formato hh:mm:ss
    end_time = time.strftime("%H:%M:%S", time.gmtime(end_time))

    print(f'\nEpoch [{epoch+1}/{num_epochs}], '
          f'Train Loss: {running_loss/len(train_loader):.4f}, '
          f'Test Loss: {test_loss/len(validation_loader):.4f}, '
          f'Accuracy: {accuracy:.2f}% '
          f'Time: {end_time}')

print("Entrenamiento completado.")

writer.close()  # Cerrar el escritor de TensorBoard
