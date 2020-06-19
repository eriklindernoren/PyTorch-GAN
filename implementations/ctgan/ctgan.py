import argparse
import os
import numpy as np
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=100, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--lambda_gp", type=int, default=10, help="loss weight for gradient penalty")
parser.add_argument("--lambda_ct", type=int, default=10, help="loss weight for consistency term")
parser.add_argument("--lambda_ct_M", type=int, default=0, help="hyperparameter M for consistency term")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=2, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False


class Generator(nn.Module):
    def __init__(self, input_size):
        super(Generator, self).__init__()

        strides = [1, 2, 2, 2]
        padding = [0, 1, 1, 1]
        channels = [input_size,
                    256, 128, 64, 1]  # 1表示一维
        kernels = [4, 3, 4, 4]

        model = []
        for i, stride in enumerate(strides):
            model.append(
                nn.ConvTranspose2d(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    stride=stride,
                    kernel_size=kernels[i],
                    padding=padding[i]
                )
            )

            if i != len(strides) - 1:
                model.append(
                    nn.BatchNorm2d(channels[i + 1], 0.8)
                )
                model.append(
                    nn.LeakyReLU(.2)
                )
            else:
                model.append(
                    nn.Tanh()
                )

        self.main = nn.Sequential(*model)

    def forward(self, x):
        x = self.main(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, input_size=1):
        super(Discriminator, self).__init__()

        strides = [2, 2, 2, 1]
        padding = [1, 1, 1, 0]
        channels = [input_size,
                    64, 128, 256, 1]  # 1表示一维
        kernels = [4, 4, 4, 3]

        model = []
        for i, stride in enumerate(strides):
            model.append(
                nn.Conv2d(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    stride=stride,
                    kernel_size=kernels[i],
                    padding=padding[i]
                )
            )
            model.append(
                nn.LeakyReLU(0.2)
            )
            model.append(
                nn.Dropout2d(.1)
            )

        self.main = nn.Sequential(*(model[:-3]))
        self.final = nn.Sequential(
            *model[-3:]
        )

    def forward(self, x):
        x_ = self.main(x)  # D_(x)
        x = self.final(x_)
        return torch.squeeze(x), x_.view(x.shape[0], -1)


# Loss weight for gradient penalty
lambda_gp = opt.lambda_gp
# Loss weight for consistency term
lambda_ct = opt.lambda_ct
# hyperparameter M for consistency term
M = opt.lambda_ct_M
# dimensionality of the latent space
latent_dim = opt.latent_dim

# Initialize generator and discriminator
G = Generator(latent_dim)
D = Discriminator()

if cuda:
    G.cuda()
    D.cuda()

# Configure data loader
os.makedirs("../../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(G.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(D.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

mse = nn.MSELoss()

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def gp_loss(D, real_x, fake_x, cuda=False):
    if cuda:
        alpha = torch.rand((real_x.shape[0], 1, 1, 1)).cuda()
    else:
        alpha = torch.rand((real_x.shape[0], 1, 1, 1))
    x_ = (alpha * real_x + (1-alpha) * fake_x).requires_grad_(True)
    y_ = D(x_)[0]
    # cal f'(x)
    grad = autograd.grad(
        outputs=y_,
        inputs=x_,
        grad_outputs=torch.ones_like(y_),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad = grad.view(x_.shape[0], -1)
    gp = ((grad.norm(2, dim=1) - 1) ** 2).mean()
    return gp


# ----------
#  Training
# ----------

batches_done = 0
for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Configure input
        x = Variable(imgs.type(Tensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Sample noise as generator input
        z = Variable(torch.randn((x.shape[0], latent_dim, 1, 1)).type(Tensor))

        G_imgs = G(z)

        D_fake1, D_fake1_ = D(G_imgs)

        D_real1, D_real1_ = D(x)
        D_real2, D_real2_ = D(x)

        D_real_loss = -torch.mean(D_real1)
        D_fake_loss = torch.mean(D_fake1)

        adv_loss = D_real_loss + D_fake_loss

        CT_loss = mse(D_real1, D_real2) + 0.1 * mse(D_real1_, D_real2_) - M

        if CT_loss > 0:
            D_loss = adv_loss + lambda_gp * gp_loss(D, x, G_imgs, cuda=True) + lambda_ct * CT_loss
        else:
            D_loss = adv_loss + lambda_gp * gp_loss(D, x, G_imgs, cuda=True)
        
        optimizer_D.zero_grad()
        D_loss.backward(retain_graph=True)
        optimizer_D.step()

        # Train the generator every n_critic steps
        if i % opt.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            G_loss = -torch.mean(D_fake1)

            optimizer_G.zero_grad()
            G_loss.backward(retain_graph=True)
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), D_loss.item(), G_loss.item())
            )

            if batches_done % opt.sample_interval == 0:
                save_image(G_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

            batches_done += opt.n_critic
