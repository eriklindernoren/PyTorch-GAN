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

from models import *
from datasets import *
from utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs('images', exist_ok=True)
os.makedirs('saved_models', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--dataset_name', type=str, default="edges2shoes", help='name of the dataset')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch from which to start lr decay')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_height', type=int, default=128, help='size of image height')
parser.add_argument('--img_width', type=int, default=128, help='size of image width')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--latent_dim', type=int, default=8, help='dimensionality of latent representation')
parser.add_argument('--sample_interval', type=int, default=1000, help='interval between sampling of images from generators')
parser.add_argument('--checkpoint_interval', type=int, default=-1, help='interval between model checkpoints')
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_height, opt.img_width)

# Loss functions
adversarial_loss = torch.nn.MSELoss()
pixelwise_loss = torch.nn.L1Loss()
latent_loss = torch.nn.L1Loss()
kl_loss = torch.nn.KLDivLoss()

cuda = True if torch.cuda.is_available() else False

# Calculate output of image D_VAE (PatchGAN)
patch_h, patch_w = int(opt.img_height / 2**3), int(opt.img_width / 2**3)
# Discriminators output two patch shapes
patch1 = (opt.batch_size, 1, patch_h, patch_w)
patch2 = (opt.batch_size, 1, patch_h // 2, patch_w // 2)

# Initialize generator, encoder and discriminators
generator = Generator(opt.latent_dim, img_shape)
encoder = Encoder(opt.latent_dim)
D_VAE = Discriminator()
D_LR = Discriminator()

if cuda:
    generator = generator.cuda()
    encoder.cuda()
    D_VAE = D_VAE.cuda()
    D_LR = D_LR.cuda()
    adversarial_loss.cuda()
    pixelwise_loss.cuda()
    latent_loss.cuda()
    kl_loss.cuda()

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load('saved_models/generator_%d.pth'))
    encoder.load_state_dict(torch.load('saved_models/encoder_%d.pth'))
    D_VAE.load_state_dict(torch.load('saved_models/D_VAE_%d.pth'))
    D_LR.load_state_dict(torch.load('saved_models/D_LR_%d.pth'))
else:
    # Initialize weights
    generator.apply(weights_init_normal)
    D_VAE.apply(weights_init_normal)
    D_LR.apply(weights_init_normal)

# Loss weights
lambda_pixel = 10
lambda_latent = 0.5
lambda_kl = 0.01

# Optimizers
optimizer_E = torch.optim.Adam(encoder.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_VAE = torch.optim.Adam(D_VAE.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_LR = torch.optim.Adam(D_LR.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Learning rate update schedulers
lr_scheduler_E = torch.optim.lr_scheduler.LambdaLR(optimizer_E, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_VAE = torch.optim.lr_scheduler.LambdaLR(optimizer_D_VAE, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_LR = torch.optim.lr_scheduler.LambdaLR(optimizer_D_LR, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
input_A = Tensor(opt.batch_size, opt.channels, opt.img_height, opt.img_width)
input_B = Tensor(opt.batch_size, opt.channels, opt.img_height, opt.img_width)
# Adversarial ground truths
valid1 = Variable(Tensor(np.ones(patch1)), requires_grad=False)
valid2 = Variable(Tensor(np.ones(patch2)), requires_grad=False)
fake1 = Variable(Tensor(np.zeros(patch1)), requires_grad=False)
fake2 = Variable(Tensor(np.zeros(patch2)), requires_grad=False)

# Dataset loader
transforms_ = [ transforms.Resize((opt.img_height, opt.img_width*2), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
dataloader = DataLoader(ImageDataset("../../data/%s" % opt.dataset_name, transforms_=transforms_),
                        batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)


# Progress logger
logger = Logger(opt.n_epochs, len(dataloader), opt.sample_interval, n_samples=opt.latent_dim)

# ----------
#  Training
# ----------

def reparameterization(mu, logvar):
    std = torch.exp(logvar / 2)
    sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), opt.latent_dim))))
    z = sampled_z * std + mu
    return z

for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):

        # Set model input
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))

        # -----------------------------
        #  Train Generator and Encoder
        # -----------------------------

        optimizer_E.zero_grad()
        optimizer_G.zero_grad()

        # Produce output using encoding of B (cVAE-GAN)
        mu, logvar = encoder(real_B)
        encoded_z = reparameterization(mu, logvar)
        fake_B = generator(real_A, encoded_z)

        # Produce output using sampled z (cLR-GAN)
        sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), opt.latent_dim))))
        _fake_B = generator(real_A, sampled_z)

        # Pixelwise loss of translated image by VAE
        loss_pixel = pixelwise_loss(fake_B, real_B)

        # Kullback-Leibler divergence of encoded B
        loss_kl = torch.sum(0.5 * (mu**2 + torch.exp(logvar) - logvar - 1))

        # Discriminators evaluate generated samples
        VAE_validity1, VAE_validity2 = D_VAE(fake_B)
        LR_validity1, LR_validity2 = D_LR(_fake_B)

        # Adversarial losses
        loss_VAE_GAN =  (adversarial_loss(VAE_validity1, valid1) + \
                        adversarial_loss(VAE_validity2, valid2)) / 2
        loss_LR_GAN =   (adversarial_loss(LR_validity1, valid1) + \
                        adversarial_loss(LR_validity2, valid2)) / 2

        # Shared losses between encoder and generator
        loss_GE =   loss_VAE_GAN + \
                    loss_LR_GAN + \
                    lambda_pixel * loss_pixel + \
                    lambda_kl * loss_kl

        loss_GE.backward()
        optimizer_E.step()

        # Latent L1 loss
        _mu, _ = encoder(generator(real_A, sampled_z))
        loss_latent = lambda_latent * latent_loss(_mu, sampled_z)

        loss_latent.backward()
        optimizer_G.step()

        # --------------------------------
        #  Train Discriminator (cVAE-GAN)
        # --------------------------------

        optimizer_D_VAE.zero_grad()

        # Real loss
        pred_real1, pred_real2 = D_VAE(real_B)
        loss_real = (adversarial_loss(pred_real1, valid1) + \
                    adversarial_loss(pred_real2, valid2)) / 2

        # Fake loss (D_LR evaluates samples produced by encoded B)
        pred_gen1, pred_gen2 = D_VAE(fake_B.detach())
        loss_fake = (adversarial_loss(pred_gen1, fake1) + \
                    adversarial_loss(pred_gen2, fake2)) / 2

        # Total loss
        loss_D_VAE = (loss_real + loss_fake) / 2

        loss_D_VAE.backward()
        optimizer_D_VAE.step()

        # -------------------------------
        #  Train Discriminator (cLR-GAN)
        # -------------------------------

        optimizer_D_LR.zero_grad()

        # Real loss
        pred_real1, pred_real2 = D_LR(real_B)
        loss_real = (adversarial_loss(pred_real1, valid1) + \
                    adversarial_loss(pred_real2, valid2)) / 2

        # Fake loss (D_LR evaluates samples produced by sampled z)
        pred_gen1, pred_gen2 = D_LR(_fake_B.detach())
        loss_fake = (adversarial_loss(pred_gen1, fake1) + \
                    adversarial_loss(pred_gen2, fake2)) / 2

        # Total loss
        loss_D_LR = 0.5 * (loss_real + loss_fake)

        loss_D_LR.backward()
        optimizer_D_LR.step()

        # --------------
        #  Log Progress
        # --------------

        logger.log({'loss_D_VAE': loss_D_VAE, 'loss_D_LR': loss_D_LR,
                    'loss_G': loss_GE, 'loss_pixel': loss_pixel, 'loss_latent': loss_latent},
                    images={'real_B': real_B, 'fake_B': fake_B, 'real_A': real_A},
                    epoch=epoch, batch=i)

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_VAE.step()
    lr_scheduler_D_LR.step()

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), 'saved_models/generator_%d.pth' % epoch)
        torch.save(encoder.state_dict(), 'saved_models/encoder_%d.pth' % epoch)
        torch.save(D_VAE.state_dict(), 'saved_models/D_VAE_%d.pth' % epoch)
        torch.save(D_LR.state_dict(), 'saved_models/D_LR_%d.pth' % epoch)
