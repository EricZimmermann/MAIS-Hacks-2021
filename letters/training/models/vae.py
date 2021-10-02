import torch
from torch import nn
from torch.nn import functional as F
from typing import List
from torch import Tensor

def variational_classifier_loss(sample, reconstruction, classes, mu, log_var, kld_weight, cls_weight):
    recon_loss = F.mse_loss(reconstruction, sample)
    kld_loss = torch.mean(-0.5*torch.sum(1+log_var-mu**2-log_var.exp(), dim=1), dim=0)
    cls_loss = F.cross_entropy_loss(mu, classes)
    total_loss = recon_loss + kld_weight*kld_loss + cls_weight*cls_loss
    return [total_loss, recon_loss, -kld_loss, cls_loss]


class VAEClassifier(nn.Module):

    '''
        Classic convolutional VAE ported over and adapted from
        https://github.com/AntixK/PyTorch-VAE
        Additional classification head added
    '''
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List,
                 n_cls) -> None:
        super(VAEClassifier, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(True)))
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)

        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU()))

        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())
        
        self.classifier = nn.LazyLinear(n_cls)

    def encode(self, input: Tensor) -> List[Tensor]:
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor) -> List[Tensor]:
        mu, log_var = self.encode(input)
        cls = self.classifier(mu)
        z = self.reparameterize(mu, log_var)
        output = self.decode(z)
        return  output, mu, log_var, cls