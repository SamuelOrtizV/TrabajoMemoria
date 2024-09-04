

# check if cuda is available

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
""" 
print(device) """

import torch
print(torch.cuda.is_available())
print(torch.version.cuda)