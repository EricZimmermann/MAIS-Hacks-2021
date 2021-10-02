import torch
from torch import nn
from torch.nn import functional as F
from typing import List
from torch import Tensor

'''
    Implements a convolutional VAE as specified in 
    https://colab.research.google.com/github/smartgeometry-ucl/dl4g/blob/master/variational_autoencoder.ipynb#scrollTo=l7b-GgGi8Ilp
    based on https://arxiv.org/pdf/1312.6114.pdf
    
    Adapted to classify latent space jointly with reconstructions
'''

def vae_cls_loss(recon_x, x, mu, logvar, weight, cls, labels):
    recon_loss = F.binary_cross_entropy(recon_x, x)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    cls_loss = F.cross_entropy(cls, labels)
    total_loss = 5*recon_loss + kl_loss + cls_loss
    return total_loss, recon_loss, kl_loss, cls_loss

class Encoder(nn.Module):
    def __init__(self, c, dim_z):
        super(Encoder, self).__init__()
        self.c = c
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=c, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=c*2, out_channels=c*4, kernel_size=4, stride=2, padding=1)
        self.fc_mu = nn.LazyLinear(dim_z)
        self.fc_logvar = nn.LazyLinear(dim_z)
            
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        return x_mu, x_logvar

class Decoder(nn.Module):
    def __init__(self, c):
        super(Decoder, self).__init__()
        self.c = c
        self.fc = nn.LazyLinear(c*4*4*4)
        self.conv3 = nn.ConvTranspose2d(in_channels=c*4, out_channels=c*2, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(in_channels=c*2, out_channels=c, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=c, out_channels=1, kernel_size=4, stride=2, padding=1)
            
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), self.c*4, 4, 4)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.conv1(x))
        return x
    
class VAE(nn.Module):
    def __init__(self, c, dim_z, ncls):
        super(VAE, self).__init__()
        self.encoder = Encoder(c, dim_z)
        self.decoder = Decoder(c)
        self.classifier = nn.Sequential(nn.LazyLinear(128),
                                        nn.ReLU(True),
                                        nn.LazyLinear(ncls))
    
    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        cls = self.classifier(latent_mu)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar, cls
    
    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu