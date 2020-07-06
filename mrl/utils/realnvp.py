# Adapted from: https://github.com/wjy5446/pytorch-Real-NVP

import torch
import torch.nn as nn
import numpy as np

class Loss(nn.Module):
    def __init__(self, prior):
        super(Loss, self).__init__()
        self.prior = prior

    def __call__(self, z, sum_log_det_jacobians):
        log_p = self.prior.log_prob(z)
        return -(log_p + sum_log_det_jacobians).mean()


class RealNVP(nn.Module):
    def __init__(self, prior=None, input_channel=2, lr=1e-3, num_layer_pairs=3, dev=None):
        super().__init__()

        if dev is None:
          self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
          self.dev = dev

        layers = []

        self.input_channel = input_channel

        for _ in range(num_layer_pairs):
          layers += [CouplingLayer("01", self.input_channel),
                     CouplingLayer("10", self.input_channel)]

        self.layers = nn.Sequential(*layers).to(self.dev)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        if prior is None:
            self.prior = torch.distributions.MultivariateNormal(torch.zeros(input_channel).to(self.dev),
                                                       torch.eye(input_channel).to(self.dev))
        else:
            self.prior = prior

    def forward(self, x, reverse=False):
        if not reverse:
            sum_log_det_jacobians = x.new_zeros(x.size(0))

            z = x
            for layer in self.layers:
              z, log_det_jacobians = layer(z, reverse)
              sum_log_det_jacobians += log_det_jacobians

            return z, sum_log_det_jacobians
        else:
            output = x
            for layer in list(self.layers)[::-1]:
              output = layer(output, reverse)

            return output

    def fit(self, data, epochs=10):
        if isinstance(data, torch.Tensor):
            data = data.to(self.dev)
        else:
            data = torch.from_numpy(data).to(self.dev)
        loss_log_det_jacobians = Loss(self.prior)
        for epoch in range(epochs):
            self.train()

            z, sum_log_det_jacobian = self(data)
            loss = loss_log_det_jacobians(z, sum_log_det_jacobian)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def sample(self, num_samples=1000):
        z = self.prior.sample((num_samples,))
        x = self(z, reverse=True)
        return x.detach().cpu().numpy()

    def score_samples(self, x):
        z, sum_log_det_jacobian = self(torch.from_numpy(x).to(self.dev))

        log_pz = self.prior.log_prob(z)

        log_px = log_pz + sum_log_det_jacobian

        log_px = log_px.detach().cpu().numpy()

        return log_px


class CouplingLayer(nn.Module):
    def __init__(self, mask_type, input_channel, dev = None):
        super().__init__()
        self.function_s_t = Function_s_t(input_channel)
        self.mask_type = mask_type
        self.input_channel = input_channel

        if dev is None:
          self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
          self.dev = dev

        d = self.input_channel // 2
        if '01' in self.mask_type:
            self.mask = torch.tensor([[0.0]*d + (self.input_channel - d)*[1.0]]).to(self.dev)
        else:
            self.mask = torch.tensor([[1.0]*d + (self.input_channel - d)*[0.0]]).to(self.dev)

    def forward(self, x, reverse=False):

        if not reverse:
            # masked half of x
            x1 = x * self.mask
            s,t = self.function_s_t(x1, self.mask)

            # z_1:d = x_1:d
            # z_d+1:D = exp(s(x_1:d)) * x_d+1:D + m(x_1:d)
            y = x1 + ((-self.mask+1.0) * (x*torch.exp(s)+t))

            # calculation of jacobians
            log_det_jacobian = torch.sum(s, 1)

            return y, log_det_jacobian
        else:
            # masked half of y
            x1 = x * self.mask
            s,t = self.function_s_t(x1, self.mask)

            # x_1:d = z_1:d
            # x_d+1:D = z_d+1:D - m(z_1:d) * exp(-s(z_1:d))
            y = x1 + (-self.mask+1.0) * ((x-t) * torch.exp(-s))

            return y

class Function_s_t(nn.Module):
    ############################################
    # scale, translation function
    ############################################
    def __init__(self, input_channel, channel=256):
        super().__init__()
        self.input_channel = input_channel
        layers = []

        layers += [
            nn.Linear(input_channel, channel),
            nn.LeakyReLU(),
            nn.Linear(channel, channel),
            nn.LeakyReLU(),
            nn.Linear(channel, input_channel*2)]

        self.model = nn.Sequential(*layers)
        self.w_scale = torch.rand(1, requires_grad=True)

    def forward(self, x, mask):
        x = self.model(x)

        #######################################
        # scale function : first half dimension
        # translation function : second half dimension
        #######################################
        s = x[:,:self.input_channel] * (-mask+1)
        t = x[:,self.input_channel:] * (-mask+1)

        s = nn.Tanh()(s)

        return s, t