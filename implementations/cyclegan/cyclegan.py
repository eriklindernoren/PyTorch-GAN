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
parser.add_argument('--dataset_name', type=str, default="horse2zebra", help='name of the dataset')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch from which to start lr decay')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_height', type=int, default=256, help='size of image height')
parser.add_argument('--img_width', type=int, default=256, help='size of image width')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=100, help='interval between sampling of images from generators')
parser.add_argument('--checkpoint_interval', type=int, default=-1, help='interval between model checkpoints')
parser.add_argument('--generator_type', type=str, default='resnet', help="'resnet' or 'unet'")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False

# Calculate output of image discriminator (PatchGAN)
patch_h, patch_w = int(opt.img_height / 2**3), int(opt.img_width / 2**3)
patch = (opt.batch_size, 1, patch_h, patch_w)

# Initialize generator and discriminator
G_AB = GeneratorResNet() if opt.generator_type == 'resnet' else GeneratorUNet()
G_BA = GeneratorResNet() if opt.generator_type == 'resnet' else GeneratorUNet()
D_A = Discriminator()
D_B = Discriminator()

if cuda:
    G_AB = G_AB.cuda()
    G_BA = G_BA.cuda()
    D_A = D_A.cuda()
    D_B = D_B.cuda()

if opt.epoch != 0:
    # Load pretrained models
    G_AB.load_state_dict(torch.load('saved_models/G_AB_%d.pth'))
    G_BA.load_state_dict(torch.load('saved_models/G_BA_%d.pth'))
    D_A.load_state_dict(torch.load('saved_models/D_A_%d.pth'))
    D_B.load_state_dict(torch.load('saved_models/D_B_%d.pth'))
else:
    # Initialize weights
    G_AB.apply(weights_init_normal)
    G_BA.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

# Loss weights
lambda_cyc = 10
lambda_id = 0.0 * lambda_cyc

# Optimizers
optimizer_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()),
                                lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
input_A = Tensor(opt.batch_size, opt.channels, opt.img_height, opt.img_width)
input_B = Tensor(opt.batch_size, opt.channels, opt.img_height, opt.img_width)
# Adversarial ground truths
valid = Variable(Tensor(np.ones(patch)), requires_grad=False)
fake = Variable(Tensor(np.zeros(patch)), requires_grad=False)

# Buffers of previously generated samples
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Dataset loader
transforms_ = [ transforms.Resize(int(opt.img_height*1.12), Image.BICUBIC),
                transforms.RandomCrop((opt.img_height, opt.img_width)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
dataloader = DataLoader(ImageDataset("../../data/%s" % opt.dataset_name, transforms_=transforms_, unaligned=True),
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

        # Identity loss
        loss_id_B = criterion_identity(G_AB(real_B), real_B)
        loss_id_A = criterion_identity(G_BA(real_A), real_A)

        loss_identity = 0.5 * (loss_id_A + loss_id_B)

        # GAN loss
        fake_B = G_AB(real_A)
        pred_fake = D_B(fake_B)
        loss_GAN_AB = criterion_GAN(pred_fake, valid)

        fake_A = G_BA(real_B)
        pred_fake = D_A(fake_A)
        loss_GAN_BA = criterion_GAN(pred_fake, valid)

        loss_GAN = 0.5 * (loss_GAN_AB + loss_GAN_BA)

        # Cycle loss
        recov_A = G_BA(fake_B)
        loss_cycle_A = criterion_cycle(recov_A, real_A)

        recov_B = G_AB(fake_A)
        loss_cycle_B = criterion_cycle(recov_B, real_B)

        loss_cycle = 0.5 * (loss_cycle_A + loss_cycle_B)

        # Total loss
        loss_G =    loss_GAN + \
                    lambda_cyc * loss_cycle + \
                    lambda_id * loss_identity

        loss_G.backward()

        optimizer_G.step()

        # -----------------------
        #  Train Discriminator A
        # -----------------------

        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = D_A(real_A)
        loss_real = criterion_GAN(pred_real, valid)

        # Fake loss (on batch of previously generated samples)
        fake_A_ = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = D_A(fake_A_.detach())
        loss_fake = criterion_GAN(pred_fake, fake)

        # Total loss
        loss_D_A = 0.5 * (loss_real + loss_fake)
        loss_D_A.backward()

        optimizer_D_A.step()

        # -----------------------
        #  Train Discriminator B
        # -----------------------

        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = D_B(real_B)
        loss_real = criterion_GAN(pred_real, valid)

        # Fake loss (on batch of previously generated samples)
        fake_B_ = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = D_B(fake_B_.detach())
        loss_fake = criterion_GAN(pred_fake, fake)

        # Total loss
        loss_D_B = (loss_real + loss_fake)*0.5
        loss_D_B.backward()

        optimizer_D_B.step()

        # --------------
        #  Log Progress
        # --------------

        logger.log({'loss_G': loss_G,
                    'loss_G_id': loss_identity,
                    'loss_G_GAN': loss_GAN,
                    'loss_G_cycle': loss_cycle,
                    'loss_D': (loss_D_A + loss_D_B)},
                    images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A,
                            'fake_B': fake_B, 'recov_A': recov_A, 'recov_B': recov_B},
                    epoch=epoch, batch=i)

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(G_AB.state_dict(), 'saved_models/G_AB_%d.pth' % epoch)
        torch.save(G_BA.state_dict(), 'saved_models/G_BA_%d.pth' % epoch)
        torch.save(D_A.state_dict(), 'saved_models/D_A_%d.pth' % epoch)
        torch.save(D_B.state_dict(), 'saved_models/D_B_%d.pth' % epoch)
