"""
A generic pytorch VAE. Based on https://github.com/pytorch/examples/blob/master/vae/main.py.

NOTE: NOT TESTED/USED may be wrong.
"""

import torch
from torch import nn, optim
from torch.nn import functional as F

class VAE(nn.Module):
  def __init__(self, input_dim, hidden_dim=256, latent_dim=64, num_hidden=2):
    super(VAE, self).__init__()

    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.latent_dim = latent_dim
    self.num_hidden = 2

    layer_sizes = [input_dim] + [hidden_dim] * num_hidden
    self.encoder = []
    for i, o in zip(layer_sizes[:-1], layer_sizes[1:]):
      self.encoder.append(nn.Linear(i, o))
      self.encoder.append(nn.ReLU())
    self.encoder = nn.Sequential(*self.encoder)

    self.mu = nn.Linear(hidden_dim, latent_dim)
    self.sigma = nn.Linear(hidden_dim, latent_dim)

    layer_sizes = [latent_dim] + [hidden_dim] * num_hidden + [input_dim]
    self.decoder = []
    for i, o in zip(layer_sizes[:-1], layer_sizes[1:]):
      self.decoder.append(nn.Linear(i, o))
      self.decoder.append(nn.ReLU())
    self.decoder.pop()
    self.decoder = nn.Sequential(*self.decoder)

  def encode(self, x):
    h = self.encoder(x)
    return self.mu(h), self.sigma(h)

  def reparameterize(self, mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mu + eps*std

  def decode(self, z):
    return self.decoder(z)

  def forward(self, x):
    mu, logvar = self.encode(x)
    z = self.reparameterize(mu, logvar)
    return self.decode(z), mu, logvar


# Mean reconstruction + KL divergence losses summed over all elements and batch
def vae_loss_function(recon_x, x, mu, logvar):
    MSE = torch.sum((recon_x - x)**2, -1)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), -1)

    return (MSE + KLD).mean()