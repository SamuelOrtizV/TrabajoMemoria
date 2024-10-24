import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
    
class RacingDataset(Dataset):
    def __init__(self, data_dir, seq_len, input_size=(224, 224), controller = False):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.input_size = input_size
        self.controller = controller
        self.transform = self.set_transform()
        self.image_raw_paths = os.listdir(data_dir) 
        self.image_paths = sorted(self.image_raw_paths, key=lambda x: int(x.split('_')[0])) # Ordenar las imágenes por número de secuencia
        self.data_dimension = self.get_data_dimension()      
    

    def load_data(self, idx):
        return Image.open(os.path.join(self.data_dir, self.image_paths[idx]))
    
    def get_data_dimension(self):
        image = self.load_data(0)
        return image.size
    
    def set_transform(self):
        transform = transforms.Compose([
            transforms.Resize((self.input_size[0], int(self.input_size[0]*(9/16)))),  # Cambia el tamaño de las imágenes a (height x width)
            transforms.ToTensor(),         # Convierte las imágenes a tensores
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizar la imagen
        ])
        return transform

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

        if self.controller:
            steering = float(self.image_paths[idx].split("_")[1].rsplit('.', 1)[0].split(" ")[0])
            throttle = float(self.image_paths[idx].split("_")[1].rsplit('.', 1)[0].split(" ")[1])
            label = torch.tensor([steering, throttle], dtype=torch.float32)
        else:
            label = label = [int(self.image_paths[idx].split('_')[1].split(".")[0])]
            
        if self.transform:
            images_seq = [self.transform(image) for image in images_seq] #IDEA: Aplicar transformaciones a cada imagen de la secuencia de imágenes en lugar de a la secuencia completa

        # Convertir la lista de imágenes a un tensor de tamaño (seq_len, Channels, Height, Width)

        if len(images_seq) > 0:
            images_seq = torch.stack(images_seq)
        else:
            raise RuntimeError("La lista de imágenes está vacía, no se puede apilar.")

        return images_seq, label#, img_names #Se puede quitar img_names si no se necesita

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
    def __init__(self, cnn_name, output_size, input_size=(3, 224, 224), dropout=0, bias=True, cnn_train=True):
        super(CNN, self).__init__()
        self.cnn_name = cnn_name
        self.output_size = output_size
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
        conv_output_size = self._get_conv_output_size()
        
        # Definir la capa totalmente conectada
        self.fc = nn.Linear(conv_output_size, output_size, bias=bias)
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
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)
        
        cnn_output = self.cnn(x)
        cnn_output = cnn_output.view(batch_size, seq_len, -1)
        
        if self.dropout_layer:
            cnn_output = self.dropout_layer(cnn_output)
        
        final_out = self.fc(cnn_output.view(batch_size, -1))
        return final_out

class CNN_RNN(nn.Module):
    """
    Modelo de red neuronal recurrente simple que utiliza una CNN preentrenada para extraer características de las imágenes
    y una capa RNN para procesar la secuencia de características.

    Args:
        cnn_name (str): Nombre de la CNN a utilizar. Opciones: 'resnet18', 'efficientnet-b0', 'vgg11', etc.
        hidden_size (int): Tamaño del estado oculto de la RNN.
        output_size (int): Tamaño de la salida del modelo.
        input_size (tuple, optional): Tamaño de la entrada de las imágenes (canales, altura, ancho). Máximo tamaño es 384x384.
        num_layers (int, optional): Número de capas de la RNN. 
        dropout (float, optional): Probabilidad de dropout.
        bias (bool, optional): Si se incluye sesgo en las capas lineales.
    """
    def __init__(self, cnn_name, hidden_size, output_size, input_size=(3, 224, 224), num_layers=1, dropout=0, bias=True, cnn_train = True):
        super(CNN_RNN, self).__init__()
        self.cnn_name = cnn_name
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bias = bias
        
        # Obtener la CNN preentrenada
        self.cnn = self.get_cnn()

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
    
    def get_cnn(self):
        # Obtener la clase de la CNN
        cnn_call_method = getattr(models, self.cnn_name)
        # Instanciar el modelo
        cnn_model = cnn_call_method(weights='IMAGENET1K_V1')        

        return cnn_model
    
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
       
class CNN_LSTM_STATE(nn.Module):
    """
    Modelo de red neuronal con LSTM que utiliza una CNN preentrenada para extraer características de las imágenes
    y una capa LSTM para procesar la secuencia de características.

    Args:
        cnn_name (str): Nombre de la CNN a utilizar. Ejemplos: 'resnet18', 'efficientnet-b0', 'vgg11', etc.
        hidden_size (int): Tamaño del estado oculto de la LSTM.
        output_size (int): Tamaño de la salida del modelo.
        input_size (tuple, optional): Tamaño de la entrada de las imágenes (canales, altura, ancho).
        num_layers (int, optional): Número de capas de la LSTM. 
        dropout (float, optional): Probabilidad de dropout.
        bias (bool, optional): Si se incluye sesgo en las capas lineales.
        cnn_train (bool, optional): Si la CNN se entrena o no.
    """
    def __init__(self, cnn_name, hidden_size, output_size, input_size=(3, 224, 224), num_layers=1, dropout=0, bias=True, cnn_train=True):
        super(CNN_LSTM_STATE, self).__init__()
        self.cnn_name = cnn_name
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bias = bias
        
        # Obtener la CNN preentrenada
        self.cnn = self.get_cnn()

        # Usar el modelo preentrenado
        self.cnn.classifier = nn.Identity()  # Quitar la última capa de clasificación
        
        if not cnn_train:
            for param in self.cnn.parameters():
                param.requires_grad = False
        
        # Obtener el tamaño de la salida de la CNN
        conv_output_size = self._get_conv_output_size()
        
        # Definir la capa LSTM y la capa totalmente conectada
        self.lstm = nn.LSTM(conv_output_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout, bias=bias)
        self.fc = nn.Linear(hidden_size, output_size)
    
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
    
    def forward(self, x, hidden_state=None):
        batch_size, seq_len, _, _, _ = x.size()
        conv_out = []
        for t in range(seq_len):
            # Pasar cada imagen a través de la CNN preentrenada
            cnn_output = self.cnn(x[:, t, :, :, :])
            conv_out.append(cnn_output.view(batch_size, -1))
        
        conv_out = torch.stack(conv_out, dim=1)  # Tamaño: (batch_size, seq_len, conv_output_size)
        
        # Pasar la salida de la CNN a la LSTM
        lstm_out, hidden_state = self.lstm(conv_out, hidden_state)  # hidden_state incluye (hidden, cell)
        
        # Tomar la última salida de la secuencia y pasarla por la capa fully connected
        final_out = self.fc(lstm_out[:, -1, :])
        
        return final_out, hidden_state

    def init_hidden(self, batch_size):
        """
        Inicializa el estado oculto (hidden state) y el estado de celda (cell state) a ceros.
        """
        weight = next(self.parameters()).data
        return (weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to(weight.device),
                weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to(weight.device))
