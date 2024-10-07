import torch
from torchvision import transforms
import matplotlib.pyplot as plt
#from model import RacingDataset 
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import time
from IPython.display import clear_output, display


class RacingDataset(Dataset):
    def __init__(self, data_dir, seq_len, transform=None):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.transform = transform
        self.image_paths = sorted(os.listdir(data_dir))
        self.data_dimension = self.get_data_dimension()

    def load_data(self, idx):
        return Image.open(os.path.join(self.data_dir, self.image_paths[idx]))
    
    def get_data_dimension(self):
        image = self.load_data(0)
        return image.size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        images_seq = []
        img_names = []           
        # Comenzar desde idx - self.seq_len + 1 y terminar en idx
        for i in range(self.seq_len):
            current_idx = idx - self.seq_len + 1 + i

            image = self.load_data(current_idx)
            img_name = self.image_paths[current_idx]

            """ if self.load_data(current_idx) is not None:
                image = self.load_data(current_idx)
                img_name = self.image_paths[current_idx]
            else:
                image = Image.new('RGB', self.data_dimension, (0, 0, 0))
                img_name = "Black Img"  """     

            #print(img_name)

            img_names.append(img_name)
            images_seq.append(image)

        #print(img_names)
        label = int(self.image_paths[idx].split('_')[1].split(".")[0])

        #print(len(img_names))

        if self.transform:
            images_seq = [self.transform(image) for image in images_seq]

        # Convertir la lista de imágenes a un tensor de tamaño (seq_len, Channels, Height, Width)

        if len(images_seq) > 0:
            images_seq = torch.stack(images_seq)
        else:
            raise RuntimeError("La lista de imágenes está vacía, no se puede apilar.")

        #print(label)
        return images_seq, label, img_names
    

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

IMAGE_INDEX = 0

# Mostrar la primera imagen de la secuencia

# Tensor de tamaño (seq_len, Channels, Height, Width)

for i, (images, label, img_names) in enumerate(dataloader):
    if True: #i == IMAGE_INDEX:
        #print(f"Imagen {i}:")
        print(f"Etiqueta: {label.item()}", end="\r")
        #print(f"Forma del tensor de imágenes: {images.shape}")
        plt.figure(figsize=(10, 10))
        for j in range(seq_len):
            plt.subplot(1, seq_len, j + 1)
            plt.imshow(images[0][j].permute(1, 2, 0))
            plt.title(img_names[j][0])
            plt.axis('off')
        #plt.show()
        #time.sleep(1)
        plt.pause(1)  # Pausa para actualizar el gráfico
        plt.clf()   # Limpiar la figura
        clear_output(wait=True)
        