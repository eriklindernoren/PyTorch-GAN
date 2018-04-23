import argparse
import os
import numpy as np
import math
import itertools

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from datasets import MNISTM

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs('images', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=16, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--n_residual_blocks', type=int, default=6, help='number of residual blocks in generator')
parser.add_argument('--latent_dim', type=int, default=10, help='dimensionality of the noise input')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--n_critic', type=int, default=5, help='number of training steps for discriminator per iter')
parser.add_argument('--clip_value', type=float, default=0.01, help='lower and upper clip value for disc. weights')
parser.add_argument('--sample_interval', type=int, default=200, help='interval betwen image samples')
opt = parser.parse_args()
print(opt)

# Calculate output of image discriminator (PatchGAN)
patch = int(opt.img_size / 2**4)
patch = (1, patch, patch)

cuda = True if torch.cuda.is_available() else False

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class ResidualBlock(nn.Module):
    def __init__(self, in_features=64, out_features=64):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, 1, 1),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, 3, 1, 1),
            nn.BatchNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.l1 = nn.Sequential(nn.Conv2d(opt.channels, 64, 3, 1, 1), nn.ReLU(inplace=True))

        resblocks = []
        for _ in range(opt.n_residual_blocks):
            resblocks.append(ResidualBlock())
        self.resblocks = nn.Sequential(*resblocks)

        self.l2 = nn.Sequential(nn.Conv2d(64, opt.channels, 3, 1, 1), nn.Tanh())


    def forward(self, img):
        out = self.l1(img)
        out = self.resblocks(out)
        img_ = self.l2(out)

        return img_

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def block(in_features, out_features, normalization=True):
            """Discriminator block"""
            layers = [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.LeakyReLU(0.2, inplace=True) ]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_features))
            return layers

        self.model = nn.Sequential(
            *block(opt.channels, 64, normalization=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512)
        )

        in_size = opt.img_size // 2**4
        self.output_layer = nn.Sequential(nn.Linear(512*in_size**2, 1))

    def forward(self, img):
        feature_repr = self.model(img)
        feature_repr = feature_repr.view(feature_repr.size(0), -1)
        validity = self.output_layer(feature_repr)

        return validity

# Loss function
cycle_loss = torch.nn.L1Loss()

# Loss weights
lambda_adv = 1
lambda_cycle = 10

# Initialize generator and discriminator
G_AB = Generator()
G_BA = Generator()
D_A = Discriminator()
D_B = Discriminator()

if cuda:
    G_AB.cuda()
    G_BA.cuda()
    D_A.cuda()
    D_B.cuda()
    cycle_loss.cuda()

# Initialize weights
G_AB.apply(weights_init_normal)
G_BA.apply(weights_init_normal)
D_A.apply(weights_init_normal)
D_B.apply(weights_init_normal)

# Configure data loader
os.makedirs('../../data/mnist', exist_ok=True)
dataloader_A = torch.utils.data.DataLoader(
    datasets.MNIST('../../data/mnist', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.Resize(opt.img_size),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ])),
    batch_size=opt.batch_size, shuffle=True)

os.makedirs('../../data/mnistm', exist_ok=True)
dataloader_B = torch.utils.data.DataLoader(
    MNISTM('../../data/mnistm', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.Resize(opt.img_size),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ])),
    batch_size=opt.batch_size, shuffle=True)

# Optimizers
optimizer_G = torch.optim.Adam( itertools.chain(G_AB.parameters(), G_BA.parameters()),
                                lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):

    # Batch iterator
    data_iterator_A = iter(dataloader_A)
    data_iterator_B = iter(dataloader_B)

    for i in range(len(data_iterator_A) // opt.n_critic):
        # Train discriminator for n_critic times
        for _ in range(opt.n_critic):
            optimizer_G.zero_grad()

            (imgs_A, _) = data_iterator_A.next()
            (imgs_B, _) = data_iterator_B.next()

            batch_size = imgs_A.size(0)

            # Adversarial ground truths
            valid = Variable(FloatTensor(batch_size, 1).fill_(-1.0), requires_grad=False)
            fake = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)

            # Configure input
            imgs_A = Variable(imgs_A.type(FloatTensor).expand(batch_size, 3, opt.img_size, opt.img_size))
            imgs_B = Variable(imgs_B.type(FloatTensor))

            # ----------------------
            #  Train Discriminators
            # ----------------------

            optimizer_D_A.zero_grad()
            optimizer_D_B.zero_grad()

            # Generate a batch of images
            fake_A = G_BA(imgs_B)
            fake_B = G_AB(imgs_A)

            #----------
            # Domain A
            #----------

            real_validity_A = D_A(imgs_A)
            real_validity_A.backward(valid)
            fake_validity_A = D_A(fake_A)
            fake_validity_A.backward(fake)

            #----------
            # Domain B
            #----------

            real_validity_B = D_B(imgs_B)
            real_validity_B.backward(valid)
            fake_validity_B = D_B(fake_B)
            fake_validity_B.backward(fake)

            # Total loss
            D_A_loss = real_validity_A - fake_validity_A
            D_B_loss = real_validity_B - fake_validity_B

            optimizer_D_A.step()
            optimizer_D_B.step()

            # Clip weights of discriminators
            for d in [D_A, D_B]:
                for p in d.parameters():
                    p.data.clamp_(-opt.clip_value, opt.clip_value)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # Translate images to opposite domain
        fake_A = G_BA(imgs_B)
        fake_B = G_AB(imgs_A)

        # Reconstruct images
        recov_A = G_BA(fake_B)
        recov_B = G_AB(fake_A)

        # Adversarial loss
        validity_A = lambda_adv / 2 * D_A(fake_A)
        validity_A.backward(valid)
        validity_B = lambda_adv / 2 * D_B(fake_B)
        validity_B.backward(valid)

        # Cycle-consistency loss
        cycle_A = lambda_cycle / 2 * cycle_loss(recov_A, imgs_A)
        cycle_A.backward()
        cycle_B = lambda_cycle / 2 * cycle_loss(recov_B, imgs_B)
        cycle_B.backward()

        optimizer_G.step()

        # Total losses
        G_adv = validity_A + validity_B
        G_cycle = cycle_A + cycle_B

        print ("[Epoch %d/%d] [Batch %d/%d] [D_A loss: %f] [D_B loss: %f] [G loss: %f, cycle: %f]" % (epoch, opt.n_epochs,
                                                        i * opt.n_critic, len(dataloader_A),
                                                        D_A_loss.data[0], D_B_loss.data[0],
                                                        G_adv.data[0], G_cycle.data[0]))

        batches_done = len(dataloader_A) * epoch + i * opt.n_critic
        if batches_done % opt.sample_interval == 0:
            n_samples = 10
            # Concatenate samples by column
            real_A = torch.cat(imgs_A.data[:n_samples], -1)
            real_B = torch.cat(imgs_B.data[:n_samples], -1)
            fake_A = torch.cat(fake_A.data[:n_samples], -1)
            fake_B = torch.cat(fake_B.data[:n_samples], -1)
            recov_A = torch.cat(recov_A.data[:n_samples], -1)
            recov_B = torch.cat(recov_B.data[:n_samples], -1)
            # Concatenate translations by row
            ABA = torch.cat((real_A, fake_B, recov_A), -2)
            BAB = torch.cat((real_B, fake_A, recov_B), -2)
            # Save image
            save_image(torch.cat((ABA, BAB), -2), 'images/%d.png' % batches_done, nrow=2, normalize=True)
