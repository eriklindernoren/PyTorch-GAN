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

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs('images', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=1000, help='interval between image sampling')
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        layers = [  nn.Linear(opt.img_size**2, 512),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(512, 512),
                    nn.BatchNorm1d(512),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(512, opt.latent_dim)]

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        latent = self.model(img_flat)
        return latent

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        layers = [  nn.Linear(opt.latent_dim, 512),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(512, 512),
                    nn.BatchNorm1d(512),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(512, opt.img_size**2),
                    nn.Tanh()]

        self.model = nn.Sequential(*layers)

    def forward(self, noise):
        img_flat = self.model(noise)
        img = img_flat.view(img_flat.shape[0], opt.channels, opt.img_size, opt.img_size)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        layers = [  nn.Linear(opt.latent_dim, 512),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(512, 256),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(256, 1),
                    nn.Sigmoid() ]

        self.model = nn.Sequential(*layers)

    def forward(self, latent):
        validity = self.model(latent)

        return validity

# Initialize generator and discriminator
encoder = Encoder()
decoder = Decoder()
discriminator = Discriminator()

if cuda:
    encoder.cuda()
    decoder.cuda()
    discriminator.cuda()

# Initialize weights
encoder.apply(weights_init_normal)
decoder.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
os.makedirs('../../data/mnist', exist_ok=True)
mnist_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../../data/mnist', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.Resize(opt.img_size),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ])),
    batch_size=opt.batch_size, shuffle=True)

# Use binary cross-entropy loss
adversarial_loss = torch.nn.BCELoss()
pixelwise_loss = torch.nn.L1Loss()

# Optimizers
optimizer_G = torch.optim.Adam( itertools.chain(encoder.parameters(), decoder.parameters()),
                                lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def sample_image(n_row, epoch):
    """Saves a grid of generated digits"""
    # Sample noise
    z = Variable(Tensor(np.random.normal(0, 1, (n_row**2, opt.latent_dim))))
    gen_imgs = decoder(z)
    save_image(gen_imgs.data, 'images/%d.png' % epoch, nrow=n_row, normalize=True)

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(mnist_loader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        if cuda:
            imgs = imgs.type(torch.cuda.FloatTensor)

        real_imgs = Variable(imgs)

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        encoded_imgs = encoder(real_imgs)
        decoded_imgs = decoder(encoded_imgs)

        # Loss measures generator's ability to fool the discriminator
        g_loss =    0.001 * adversarial_loss(discriminator(encoded_imgs), valid) + \
                    0.999 * pixelwise_loss(decoded_imgs, real_imgs)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as discriminator ground truth
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(z), valid)
        fake_loss = adversarial_loss(discriminator(encoded_imgs.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)

        d_loss.backward()
        optimizer_D.step()

        print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, opt.n_epochs, i, len(mnist_loader),
                                                            d_loss.data.cpu().numpy()[0], g_loss.data.cpu().numpy()[0]))

    sample_image(10, epoch)
