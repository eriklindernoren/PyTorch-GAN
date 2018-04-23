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
parser.add_argument('--dataset_name', type=str, default="facades", help='name of the dataset')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch from which to start lr decay')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_height', type=int, default=128, help='size of image height')
parser.add_argument('--img_width', type=int, default=128, help='size of image width')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=500, help='interval between sampling of images from generators')
parser.add_argument('--checkpoint_interval', type=int, default=-1, help='interval between model checkpoints')
parser.add_argument('--generator_type', type=str, default='unet', help="'resnet' or 'unet'")
parser.add_argument('--n_residual_blocks', type=int, default=6, help='number of residual blocks in resnet generator')
opt = parser.parse_args()
print(opt)

# Loss functions
criterion_GAN = torch.nn.MSELoss()
criterion_translation = torch.nn.L1Loss()

cuda = True if torch.cuda.is_available() else False

# Calculate output of image discriminator (PatchGAN)
patch_h, patch_w = int(opt.img_height / 2**4), int(opt.img_width / 2**4)
patch = (opt.batch_size, 1, patch_h, patch_w)

# Initialize generator and discriminator
generator = GeneratorResNet(resblocks=opt.n_residual_blocks) if opt.generator_type == 'resnet' else GeneratorUNet()
discriminator = Discriminator()

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_GAN.cuda()
    criterion_translation.cuda()

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load('saved_models/generator_%d.pth'))
    discriminator.load_state_dict(torch.load('saved_models/discriminator_%d.pth'))
else:
    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

# Loss weight of L1 pixel-wise loss between translated image and real image
lambda_trans = 100

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
input_A = Tensor(opt.batch_size, opt.channels, opt.img_height, opt.img_width)
input_B = Tensor(opt.batch_size, opt.channels, opt.img_height, opt.img_width)
# Adversarial ground truths
valid = Variable(Tensor(np.ones(patch)), requires_grad=False)
fake = Variable(Tensor(np.zeros(patch)), requires_grad=False)

# Dataset loader
transforms_ = [ transforms.Resize((opt.img_height, opt.img_width*2), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
dataloader = DataLoader(ImageDataset("../../data/%s" % opt.dataset_name, transforms_=transforms_),
                        batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)


# Progress logger
logger = Logger(opt.n_epochs, len(dataloader), opt.sample_interval)

# ----------
#  Training
# ----------

for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):

        # Set model input
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # GAN loss
        fake_A = generator(real_B)
        pred_fake = discriminator(fake_A, real_B)
        loss_GAN = criterion_GAN(pred_fake, valid)
        loss_trans = criterion_translation(fake_A, real_A)

        # Total loss
        loss_G = loss_GAN + lambda_trans * loss_trans

        loss_G.backward()

        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Real loss
        pred_real = discriminator(real_A, real_B)
        loss_real = criterion_GAN(pred_real, valid)

        # Fake loss
        pred_fake = discriminator(fake_A.detach(), real_B)
        loss_fake = criterion_GAN(pred_fake, fake)

        # Total loss
        loss_D = 0.5 * (loss_real + loss_fake)

        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------

        logger.log({'loss_G': loss_G, 'loss_G_trans': loss_trans, 'loss_D': loss_D},
                    images={'real_B': real_B, 'fake_A': fake_A, 'real_A': real_A},
                    epoch=epoch, batch=i)

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D.step()

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), 'saved_models/generator_%d.pth' % epoch)
        torch.save(discriminator.state_dict(), 'saved_models/discriminator_%d.pth' % epoch)
