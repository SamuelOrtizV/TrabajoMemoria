import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
from PIL import Image


# Definición del dataset personalizado
class RacingDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        #self.seq_len = seq_len
        self.transform = transform
        self.image_paths = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir)])
        
    def __len__(self):
        return len(self.image_paths) #- self.seq_len + 1
    
    """ def __getitem__(self, idx):
        images = []
        labels = []
        for i in range(self.seq_len):
            img_path = self.image_paths[idx + i]
            image = Image.open(img_path).convert('L')
            if self.transform:
                image = self.transform(image)
            images.append(image)
            
            # Extraer la etiqueta del nombre de la imagen
            label = int(os.path.basename(img_path).split('_')[1].split(".")[0])  # Asume que la etiqueta está en el nombre despues del id y antes de la extensión
            labels.append(label)
        
        return torch.stack(images), labels[-1]  # Devuelve las imágenes y la última etiqueta """
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = int(os.path.basename(image_path).split('_')[1].split(".")[0])  # Asume que la etiqueta está en el nombre despues del id y antes de la extensión

        image = Image.open(image_path).convert('L')  # Convertir a escala de grises si es necesario
        if self.transform:
            image = self.transform(image)

        return image, label

# Definición del modelo
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        
        # Capa de convolución para extraer características de las imágenes
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        conv_output_size = self._get_conv_output_size(input_size)
        self.rnn = nn.RNN(conv_output_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def _get_conv_output_size(self, input_size):
        dummy_input = torch.zeros(1, 1, *input_size)
        output = self.conv(dummy_input)
        return int(torch.prod(torch.tensor(output.shape[1:])))
    
    def forward(self, x):
        batch_size, seq_len, _, _, _ = x.size()
        conv_out = []
        for t in range(seq_len):
            conv_out.append(self.conv(x[:, t, :, :, :]).view(batch_size, -1))
        conv_out = torch.stack(conv_out, dim=1)
        rnn_out, _ = self.rnn(conv_out)
        final_out = self.fc(rnn_out[:, -1, :])
        return final_out


