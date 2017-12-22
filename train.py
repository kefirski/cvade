import argparse
import os
from os import listdir

import torch.nn.functional as F
from torch.autograd import Variable
import imageio
import torch as t
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.optim import Adam
from torchvision import datasets

from model import CVaDE

if __name__ == "__main__":

    if not os.path.exists('sampling'):
        os.mkdir('sampling')

    parser = argparse.ArgumentParser(description='CDVAE')
    parser.add_argument('--num-epochs', type=int, default=10, metavar='NI',
                        help='num epochs (default: 10)')
    parser.add_argument('--batch-size', type=int, default=80, metavar='BS',
                        help='batch size (default: 40)')
    parser.add_argument('--num-clusters', type=int, default=10, metavar='NC',
                        help='num clusters (default: 10)')
    parser.add_argument('--use-cuda', type=bool, default=False, metavar='CUDA',
                        help='use cuda (default: False)')
    parser.add_argument('--learning-rate', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--save', type=str, default='trained_model', metavar='TS',
                        help='path where save trained model to (default: "trained_model")')
    args = parser.parse_args()

    dataset = datasets.MNIST(root='data/',
                             transform=transforms.Compose([
                                 transforms.ToTensor()]),
                             download=True,
                             train=True)
    dataloader = t.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = CVaDE(50, num_clusters=args.num_clusters, free_bits=0.5)
    if args.use_cuda:
        model = model.cuda()

    optimizer = Adam(model.learnable_parameters(), args.learning_rate, eps=1e-6)

    criterion = nn.BCEWithLogitsLoss(size_average=False)

    for epoch in range(args.num_epochs):
        for iteration, (input, _) in enumerate(dataloader):

            input = Variable(input)
            if args.use_cuda:
                input = input.cuda()

            optimizer.zero_grad()

            nll, kld = model.loss(input, criterion, eval=False)
            loss = nll + kld

            loss.backward()
            optimizer.step()

            if iteration % 10 == 0:
                print('epoch {}, iteration {}, loss {}'.format(epoch, iteration, loss.cpu().data.numpy()[0]))
                print(F.softmax(model.p_c_logits, dim=0).cpu().data.numpy())

                sampling = model.sample()
                vutils.save_image(sampling, 'sampling/vae_{}.png'.format(epoch * len(dataloader) + iteration))

    samplings = [f for f in listdir('sampling')]
    samplings = [imageio.imread('sampling/' + path) for path in samplings for _ in range(2)]
    imageio.mimsave('sampling/movie.gif', samplings)
