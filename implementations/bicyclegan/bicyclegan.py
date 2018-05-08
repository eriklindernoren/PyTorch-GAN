import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--dataset_name', type=str, default="edges2shoes", help='name of the dataset')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_height', type=int, default=256, help='size of image height')
parser.add_argument('--img_width', type=int, default=256, help='size of image width')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--latent_dim', type=int, default=8, help='dimensionality of latent representation')
parser.add_argument('--sample_interval', type=int, default=1000, help='interval between sampling of images from generators')
parser.add_argument('--checkpoint_interval', type=int, default=-1, help='interval between model checkpoints')
opt = parser.parse_args()
print(opt)

os.makedirs('images/%s' % opt.dataset_name, exist_ok=True)
os.makedirs('saved_models/%s' % opt.dataset_name, exist_ok=True)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

img_shape = (opt.channels, opt.img_height, opt.img_width)

# Loss functions
adversarial_loss = torch.nn.MSELoss()
pixelwise_loss = torch.nn.L1Loss()
latent_loss = torch.nn.L1Loss()
kl_loss = torch.nn.KLDivLoss()

cuda = True if torch.cuda.is_available() else False

# Calculate outputs of multilevel PatchGAN
patch1 = (opt.batch_size, 1, opt.img_height // 2**3, opt.img_width // 2**3)
patch2 = (opt.batch_size, 1, opt.img_height // 2**4, opt.img_width // 2**4)

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
    generator.load_state_dict(torch.load('saved_models/%s/generator_%d.pth' % (opt.dataset_name, opt.epoch)))
    encoder.load_state_dict(torch.load('saved_models/%s/encoder_%d.pth' % (opt.dataset_name, opt.epoch)))
    D_VAE.load_state_dict(torch.load('saved_models/%s/D_VAE_%d.pth' % (opt.dataset_name, opt.epoch)))
    D_LR.load_state_dict(torch.load('saved_models/%s/D_LR_%d.pth' % (opt.dataset_name, opt.epoch)))
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

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Adversarial ground truths
valid1 = Variable(Tensor(np.ones(patch1)), requires_grad=False)
valid2 = Variable(Tensor(np.ones(patch2)), requires_grad=False)
fake1 = Variable(Tensor(np.zeros(patch1)), requires_grad=False)
fake2 = Variable(Tensor(np.zeros(patch2)), requires_grad=False)

# Dataset loader
transforms_ = [ transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
dataloader = DataLoader(ImageDataset("../../data/%s" % opt.dataset_name, transforms_=transforms_),
                            batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)
val_dataloader = DataLoader(ImageDataset("../../data/%s" % opt.dataset_name, transforms_=transforms_, mode='val'),
                            batch_size=5, shuffle=True, num_workers=1)

def sample_images(batches_done):
    """Saves a generated sample from the validation set"""
    imgs = next(iter(val_dataloader))
    real_A = Variable(imgs['A'].type(Tensor))
    sampled_z = Variable(Tensor(np.random.normal(0, 1, (5, opt.latent_dim))))
    fake_B = generator(real_A, sampled_z)
    real_B = Variable(imgs['B'].type(Tensor))
    img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), 0)
    save_image(img_sample, 'images/%s/%s.png' % (opt.dataset_name, batches_done), nrow=5, normalize=True)

# ----------
#  Training
# ----------

def reparameterization(mu, logvar):
    std = torch.exp(logvar / 2)
    sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), opt.latent_dim))))
    z = sampled_z * std + mu
    return z

prev_time = time.time()
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):

        # Set model input
        real_A = Variable(batch['A'].type(Tensor))
        real_B = Variable(batch['B'].type(Tensor))

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

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [D VAE_loss: %f, LR_loss: %f] [G loss: %f, pixel: %f, latent: %f] ETA: %s" %
                                                        (epoch, opt.n_epochs,
                                                        i, len(dataloader),
                                                        loss_D_VAE.item(), loss_D_LR.item(),
                                                        loss_GE.item(), loss_pixel.item(),
                                                        loss_latent.item(), time_left))

        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)


    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), 'saved_models/%s/generator_%d.pth' % (opt.dataset_name, epoch))
        torch.save(encoder.state_dict(), 'saved_models/%s/encoder_%d.pth' % (opt.dataset_name, epoch))
        torch.save(D_VAE.state_dict(), 'saved_models/%s/D_VAE_%d.pth' % (opt.dataset_name, epoch))
        torch.save(D_LR.state_dict(), 'saved_models/%s/D_LR_%d.pth' % (opt.dataset_name, epoch))
