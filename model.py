import torch.nn as nn
import torch

# Define an AE encoder
class Encoder(torch.nn.Module):
    def __init__(self, latent_dim, conv_dims):
        super(Encoder, self).__init__()
        n_layers = len(conv_dims)
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(nn.Conv2d(3, conv_dims[0], 3, stride=2, padding=1))
        for i in range(n_layers-1):
            self.conv_layers.append(nn.Conv2d(conv_dims[i], conv_dims[i+1], 3, stride=2, padding=1))
        # self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(4608, latent_dim)

    def forward(self, x):
        for layer in self.conv_layers:
            x = torch.relu(layer(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return x
    
# Define an AE decoder
class Decoder(torch.nn.Module):
    def __init__(self, latent_dim, conv_dims):
        super(Decoder, self).__init__()
        n_layers = len(conv_dims)
        self.fc1 = nn.Linear(latent_dim, 4608)
        self.conv_layers = nn.ModuleList()
        for i in range(n_layers-1):
            self.conv_layers.append(nn.ConvTranspose2d(conv_dims[i], conv_dims[i+1], 3, stride=2, padding=1, output_padding=1))
        self.conv_layers.append(nn.ConvTranspose2d(conv_dims[-1], 3, 3, stride=2, padding=1, output_padding=1))

    def forward(self, x):

        x = torch.relu(self.fc1(x))
        x = x.view(-1, self.conv_layers[0].in_channels, 6, 6)
        for layer in self.conv_layers:
            x = torch.relu(layer(x))
        return x
    
# Define an AE model
class AE(torch.nn.Module):
    def __init__(self, latent_dim, conv_dims = [16, 32, 64, 128]):
        super(AE, self).__init__()
        self.encoder = Encoder(latent_dim, conv_dims)
        self.decoder = Decoder(latent_dim, conv_dims[::-1])

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x