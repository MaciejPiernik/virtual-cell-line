import torch

from torch import nn

from BaseSimulator import BaseSimulator


class Autoencoder(BaseSimulator):
    def __init__(self, layer_dims, dropout_rate=0.3):
        super(Autoencoder, self).__init__()
        
        encoder_layers = []
        for i in range(1, len(layer_dims)):
            encoder_layers.append(nn.Linear(layer_dims[i-1], layer_dims[i]))
            encoder_layers.append(nn.Dropout(dropout_rate))
            encoder_layers.append(nn.ReLU())
            
        self.encoder = nn.Sequential(*encoder_layers)
        
        decoder_layers = []
        for i in range(len(layer_dims) - 1, 0, -1):
            decoder_layers.append(nn.Linear(layer_dims[i], layer_dims[i-1]))
            decoder_layers.append(nn.Dropout(dropout_rate))
            decoder_layers.append(nn.ReLU())

        self.decoder = nn.Sequential(*decoder_layers)


    def forward(self, x, mask=None):
        if mask is not None:
            x = x * (1 - mask)

        x = torch.relu(self.encoder(x))
        x = self.decoder(x)

        return x
