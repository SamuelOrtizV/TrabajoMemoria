import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from model import RacingDataset 
from torch.utils.data import DataLoader

seq_len = 5  # Longitud de la secuencia

print("Cargango...", end="\r")

# Definir una transformación simple (opcional)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Redimensionar las imágenes
    transforms.ToTensor()  # Convertir las imágenes a tensores
])

# Crear una instancia del dataset
data_dir = './validation_dataset'
dataset = RacingDataset(data_dir, seq_len = seq_len, transform=transform)

dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Comprobar una muestra del datase

IMAGE_INDEX = 5



for i, (images, label) in enumerate(dataloader):
    if i == IMAGE_INDEX:
        print(f"Imagen {i}:")
        print(f"Etiqueta: {label.item()}")
        plt.figure(figsize=(10, 10))
        for j in range(seq_len):
            plt.subplot(1, seq_len, j + 1)
            plt.imshow(images[0, j].permute(1, 2, 0))
            plt.axis('off')
        plt.show

        break