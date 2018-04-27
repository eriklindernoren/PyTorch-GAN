import argparse
import os
import numpy as np
import math
import scipy
import itertools

import mnistm

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
parser.add_argument('--batch_size', type=int, default=32, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=28, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=400, help='interval betwen image samples')
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# Shared input layers between generators 1 and 2
G = nn.Sequential(
    nn.Linear(opt.latent_dim, 128),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Linear(128, 256),
    nn.BatchNorm1d(256),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Linear(256, 512),
    nn.BatchNorm1d(512),
    nn.LeakyReLU(0.2, inplace=True)
)
# Output layers of generator 1
G1 = nn.Sequential(
    nn.Linear(512, 1024),
    nn.BatchNorm1d(1024),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Linear(1024, int(np.prod(img_shape))),
    nn.Tanh()
)
# Output layers of generator 2
G2 = nn.Sequential(
    nn.Linear(512, 1024),
    nn.BatchNorm1d(1024),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Linear(1024, int(np.prod(img_shape))),
    nn.Tanh()
)

def generator1(noise):
    img_emb = G(noise)
    img = G1(img_emb)
    img = img.view(img.shape[0], *img_shape)
    return img

def generator2(noise):
    img_emb = G(noise)
    img = G2(img_emb)
    img = img.view(img.shape[0], *img_shape)
    return img

# Shared output layers between discriminator 1 and 2
D = nn.Sequential(
    nn.LeakyReLU(0.2, inplace=True),
    nn.Linear(256, 1),
    nn.Sigmoid()
)
# Input layers of discriminator 1
D1 = nn.Sequential(
    nn.Linear(int(np.prod(img_shape)), 512),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Linear(512, 256)
)
# Input layers of discriminator 2
D2 = nn.Sequential(
    nn.Linear(int(np.prod(img_shape)), 512),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Linear(512, 256)
)

def discriminator1(img):
    out = D1(img.contiguous().view(img.shape[0], -1))
    validity = D(out)
    return validity

def discriminator2(img):
    out = D2(img.contiguous().view(img.shape[0], -1))
    validity = D(out)
    return validity

networks = [G, G1, G2, D, D1, D2]

# Loss function
adversarial_loss = torch.nn.BCELoss()

if cuda:
    for net in networks:
        net.cuda()

# Initialize weights
for net in networks:
    net.apply(weights_init_normal)

# Configure data loader
os.makedirs('../../data/mnist', exist_ok=True)
dataloader1 = torch.utils.data.DataLoader(
    datasets.MNIST('../../data/mnist', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.Resize(opt.img_size),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ])),
    batch_size=opt.batch_size, shuffle=True)

os.makedirs('../../data/mnistm', exist_ok=True)
dataloader2 = torch.utils.data.DataLoader(
    mnistm.MNISTM('../../data/mnistm', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.Resize(opt.img_size),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ])),
    batch_size=opt.batch_size, shuffle=True)

G_params = itertools.chain(G.parameters(), G1.parameters(), G2.parameters())
D_params = itertools.chain(D.parameters(), D1.parameters(), D2.parameters())
# Optimizers
optimizer_G = torch.optim.Adam(G_params, lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(D_params, lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, ((imgs1, _), (imgs2, _)) in enumerate(zip(dataloader1, dataloader2)):

        batch_size = imgs1.shape[0]

        # Adversarial ground truths
        valid = Variable(Tensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        imgs1 = Variable(imgs1.type(Tensor).expand(imgs1.size(0), 3, opt.img_size, opt.img_size))
        imgs2 = Variable(imgs2.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))

        # Generate a batch of images
        gen_imgs1 = generator1(z)
        gen_imgs2 = generator2(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss1 = adversarial_loss(discriminator1(gen_imgs1), valid)
        g_loss2 = adversarial_loss(discriminator2(gen_imgs2), valid)

        g_loss = (g_loss1 + g_loss2) / 2

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss1 = adversarial_loss(discriminator1(imgs1), valid)
        fake_loss1 = adversarial_loss(discriminator1(gen_imgs1.detach()), fake)
        d_loss1 = (real_loss1 + fake_loss1) / 2

        real_loss2 = adversarial_loss(discriminator2(imgs2), valid)
        fake_loss2 = adversarial_loss(discriminator2(gen_imgs2.detach()), fake)
        d_loss2 = (real_loss2 + fake_loss2) / 2

        d_loss = (d_loss1 + d_loss2) / 2

        d_loss.backward()
        optimizer_D.step()

        print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, opt.n_epochs, i, len(dataloader1),
                                                            d_loss.item(), g_loss.item()))

        batches_done = epoch * len(dataloader1) + i
        if batches_done % opt.sample_interval == 0:
            gen_imgs = torch.cat((gen_imgs1.data, gen_imgs2.data), 0)
            save_image(gen_imgs, 'images/%d.png' % batches_done, nrow=8, normalize=True)
