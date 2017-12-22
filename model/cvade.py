from math import tanh

import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.weight_norm import weight_norm

from nn.utils import GumbelSoftmax


class CVaDE(nn.Module):
    def __init__(self, latent_size, num_clusters, free_bits=0.2):
        super(CVaDE, self).__init__()

        self.latent_size = latent_size
        self.num_clusters = num_clusters

        self.x_to_hidden = nn.Sequential(
            weight_norm(nn.Linear(784, 500)),
            nn.SELU(),
            weight_norm(nn.Linear(500, 2000)),
            nn.SELU()
        )

        self.hidden_to_z = nn.Sequential(
            weight_norm(nn.Linear(2000, 100)),
            nn.SELU(),
            weight_norm(nn.Linear(100, self.latent_size * 2))
        )

        self.hidden_to_cat = nn.Sequential(
            weight_norm(nn.Linear(2000, 50)),
            nn.SELU(),

            weight_norm(nn.Linear(50, num_clusters))
        )

        self.z_to_x = nn.Sequential(
            weight_norm(nn.Linear(self.latent_size, 2000)),
            nn.SELU(),
            weight_norm(nn.Linear(2000, 500)),
            nn.SELU(),
            weight_norm(nn.Linear(500, 500)),
            nn.SELU(),
            weight_norm(nn.Linear(500, 784))
        )

        self.p_c_logits = nn.Parameter(t.ones(num_clusters))
        self.p_z_mu_logvar = nn.Parameter(t.zeros(num_clusters, self.latent_size * 2))

        self.free_bits = nn.Parameter(t.FloatTensor([free_bits]), requires_grad=False)

    def forward(self, input):
        """
        :param input: An float tensor with shape of [batch_size, 784]
        :return: An float tensor with shape of [batch_size, 784] and kl-divergence estimation
        """

        batch_size, _ = input.size()

        cuda = input.is_cuda

        hidden = self.x_to_hidden(input)

        q_mu_logvar = self.hidden_to_z(hidden)
        mu, logvar = q_mu_logvar[:, :self.latent_size], q_mu_logvar[:, self.latent_size:]
        std = t.exp(0.5 * logvar)

        eps = Variable(t.randn(batch_size, self.latent_size))
        if cuda:
            eps = eps.cuda()
        z = eps * std + mu

        cat_logits = self.hidden_to_cat(hidden)
        kl_cat = self._kl_cat(cat_logits)

        cat = GumbelSoftmax(cat_logits, 0.3, hard=False)
        p_mu_logvar = t.mm(cat, self.p_z_mu_logvar)

        kl_z = self._kl_gauss(mu, logvar, p_mu_logvar[:, :self.latent_size], p_mu_logvar[:, self.latent_size:])

        kld = kl_cat + kl_z
        kld = t.max(t.stack([kld, self.free_bits.expand_as(kld)], 1), 1)[0].mean()

        result = self.z_to_x(z)

        return result, kld

    def loss(self, input, criterion, eval=False):

        batch_size, *_ = input.size()

        input = input.view(batch_size, -1)

        if eval:
            self.eval()
        else:
            self.train()

        out, kld = self(input)

        nll = criterion(out, input) / batch_size

        return nll, kld

    def sample(self, cuda):

        size = 25
        params = self.p_z_mu_logvar.repeat(size, 1)

        '''
        First row of samplings corresponding to sampling from mean of p(z|c)
        '''
        params[:size, self.latent_size:] = Variable(t.zeros(size, self.latent_size))

        mu, logvar = params[:, :self.latent_size], params[:, self.latent_size:]
        std = t.exp(0.5 * logvar)

        eps = Variable(t.randn(*mu.size()))
        if cuda:
            eps = eps.cuda()
        z = eps * std + mu

        result = F.sigmoid(self.z_to_x(z))
        result = result.view(size, self.num_clusters, 28, 28)
        result = result.transpose(1, 2).contiguous().view(size, 28, 28 * self.num_clusters)
        return result.view(-1, 28 * self.num_clusters).cpu().data

    def learnable_parameters(self):
        for par in self.parameters():
            if par.requires_grad:
                yield par

    def _kl_cat(self, posterior):
        """
        :param posterior: An float tensor with shape of [batch_size, num_clusters] where each row contains q(c|x, z)
        :return: KL-Divergence estimation for cat latent variables as E_{c ~ q(c|x, z)} [ln(q(c)/p(c))]
        """

        prior = F.softmax(self.p_c_logits, dim=0).expand_as(posterior)
        posterior = F.softmax(posterior, dim=1)
        return (posterior * t.log(posterior / (prior + 1e-12)) + 1e-12).sum(1)

    @staticmethod
    def _kl_gauss(mu, logvar, mu_c, logvar_c):
        return 0.5 * (logvar_c - logvar + t.exp(logvar) / (t.exp(logvar_c) + 1e-8) +
                      t.pow(mu - mu_c, 2) / (t.exp(logvar_c) + 1e-8) - 1).sum(1)
