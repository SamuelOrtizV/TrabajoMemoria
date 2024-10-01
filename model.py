import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
from PIL import Image

class RacingDataset(Dataset):
    def __init__(self, data_dir, seq_len, transform=None):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.transform = transform
        self.image_raw_paths = os.listdir(data_dir) 
        self.image_paths = sorted(self.image_raw_paths, key=lambda x: int(x.split('_')[0])) # Ordenar las imágenes por número de secuencia
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

            img_names.append(img_name)
            images_seq.append(image)

        label = int(self.image_paths[idx].split('_')[1].split(".")[0])

        if self.transform:
            images_seq = [self.transform(image) for image in images_seq] #IDEA: Aplicar transformaciones a cada imagen de la secuencia de imágenes en lugar de a la secuencia completa

        # Convertir la lista de imágenes a un tensor de tamaño (seq_len, Channels, Height, Width)

        if len(images_seq) > 0:
            images_seq = torch.stack(images_seq)
        else:
            raise RuntimeError("La lista de imágenes está vacía, no se puede apilar.")

        return images_seq, label#, img_names #Se puede quitar img_names si no se necesita
    
# Definición del modelo    
class SimpleRNN(nn.Module):
    """
    Modelo de red neuronal recurrente simple que utiliza una CNN preentrenada para extraer características de las imágenes
    y una capa RNN para procesar la secuencia de características.

    Args:
        pretrained_cnn (torch.nn.Module): Modelo de CNN preentrenado.
        hidden_size (int): Tamaño del estado oculto de la RNN.
        output_size (int): Tamaño de la salida del modelo.
        input_size (tuple, optional): Tamaño de la entrada de las imágenes (canales, altura, ancho). Máximo tamaño es 384x384.
        num_layers (int, optional): Número de capas de la RNN. 
        dropout (float, optional): Probabilidad de dropout.
        bias (bool, optional): Si se incluye sesgo en las capas lineales.
    """
    def __init__(self, pretrained_cnn, hidden_size, output_size, input_size=(3, 224, 224), num_layers=1, dropout=0, bias=True, cnn_train = True):
        super(SimpleRNN, self).__init__()
        self.cnn = pretrained_cnn
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bias = bias
        
        # Usar el modelo preentrenado
        self.cnn.classifier = nn.Identity()  # Quitar la última capa de clasificación
        #self.cnn.eval()  # Poner el modelo en modo evaluación
        
        if not cnn_train:
            for param in self.cnn.parameters():# Desactivar el gradiente para las capas convolucionales preentrenadas, de esta forma no se actualizarán los pesos durante el entrenamiento
                param.requires_grad = False
        
        # Obtener el tamaño de la salida de la CNN
        conv_output_size = self._get_conv_output_size()
        
        # Definir la capa RNN y la capa totalmente conectada
        self.rnn = nn.RNN(conv_output_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout, bias=bias)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def _get_conv_output_size(self):
        # Crear una entrada dummy para pasarla a través de la CNN
        dummy_input = torch.zeros(1, *self.input_size)  # Ajustar según la entrada esperada por la CNN segun los parametros ()
        output = self.cnn(dummy_input)
        return int(torch.prod(torch.tensor(output.shape[1:])))
    
    def forward(self, x):
        batch_size, seq_len, _, _, _ = x.size()
        conv_out = []
        for t in range(seq_len):
            # Pasar cada imagen a través de la CNN preentrenada
            cnn_output = self.cnn(x[:, t, :, :, :])
            conv_out.append(cnn_output.view(batch_size, -1))
        conv_out = torch.stack(conv_out, dim=1)
        rnn_out, _ = self.rnn(conv_out)
        final_out = self.fc(rnn_out[:, -1, :])
        return final_out