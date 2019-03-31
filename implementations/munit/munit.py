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
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="edges2shoes", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=128, help="size of image height")
parser.add_argument("--img_width", type=int, default=128, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval saving generator samples")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between saving model checkpoints")
parser.add_argument("--n_downsample", type=int, default=2, help="number downsampling layers in encoder")
parser.add_argument("--n_residual", type=int, default=3, help="number of residual blocks in encoder / decoder")
parser.add_argument("--dim", type=int, default=64, help="number of filters in first encoder layer")
parser.add_argument("--style_dim", type=int, default=8, help="dimensionality of the style code")
opt = parser.parse_args()
print(opt)

cuda = torch.cuda.is_available()

# Create sample and checkpoint directories
os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

criterion_recon = torch.nn.L1Loss()

# Initialize encoders, generators and discriminators
Enc1 = Encoder(dim=opt.dim, n_downsample=opt.n_downsample, n_residual=opt.n_residual, style_dim=opt.style_dim)
Dec1 = Decoder(dim=opt.dim, n_upsample=opt.n_downsample, n_residual=opt.n_residual, style_dim=opt.style_dim)
Enc2 = Encoder(dim=opt.dim, n_downsample=opt.n_downsample, n_residual=opt.n_residual, style_dim=opt.style_dim)
Dec2 = Decoder(dim=opt.dim, n_upsample=opt.n_downsample, n_residual=opt.n_residual, style_dim=opt.style_dim)
D1 = MultiDiscriminator()
D2 = MultiDiscriminator()

if cuda:
    Enc1 = Enc1.cuda()
    Dec1 = Dec1.cuda()
    Enc2 = Enc2.cuda()
    Dec2 = Dec2.cuda()
    D1 = D1.cuda()
    D2 = D2.cuda()
    criterion_recon.cuda()

if opt.epoch != 0:
    # Load pretrained models
    Enc1.load_state_dict(torch.load("saved_models/%s/Enc1_%d.pth" % (opt.dataset_name, opt.epoch)))
    Dec1.load_state_dict(torch.load("saved_models/%s/Dec1_%d.pth" % (opt.dataset_name, opt.epoch)))
    Enc2.load_state_dict(torch.load("saved_models/%s/Enc2_%d.pth" % (opt.dataset_name, opt.epoch)))
    Dec2.load_state_dict(torch.load("saved_models/%s/Dec2_%d.pth" % (opt.dataset_name, opt.epoch)))
    D1.load_state_dict(torch.load("saved_models/%s/D1_%d.pth" % (opt.dataset_name, opt.epoch)))
    D2.load_state_dict(torch.load("saved_models/%s/D2_%d.pth" % (opt.dataset_name, opt.epoch)))
else:
    # Initialize weights
    Enc1.apply(weights_init_normal)
    Dec1.apply(weights_init_normal)
    Enc2.apply(weights_init_normal)
    Dec2.apply(weights_init_normal)
    D1.apply(weights_init_normal)
    D2.apply(weights_init_normal)

# Loss weights
lambda_gan = 1
lambda_id = 10
lambda_style = 1
lambda_cont = 1
lambda_cyc = 0

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(Enc1.parameters(), Dec1.parameters(), Enc2.parameters(), Dec2.parameters()),
    lr=opt.lr,
    betas=(opt.b1, opt.b2),
)
optimizer_D1 = torch.optim.Adam(D1.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D2 = torch.optim.Adam(D2.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D1 = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D1, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D2 = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D2, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Configure dataloaders
transforms_ = [
    transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

dataloader = DataLoader(
    ImageDataset("../../data/%s" % opt.dataset_name, transforms_=transforms_),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

val_dataloader = DataLoader(
    ImageDataset("../../data/%s" % opt.dataset_name, transforms_=transforms_, mode="val"),
    batch_size=5,
    shuffle=True,
    num_workers=1,
)


def sample_images(batches_done):
    """Saves a generated sample from the validation set"""
    imgs = next(iter(val_dataloader))
    img_samples = None
    for img1, img2 in zip(imgs["A"], imgs["B"]):
        # Create copies of image
        X1 = img1.unsqueeze(0).repeat(opt.style_dim, 1, 1, 1)
        X1 = Variable(X1.type(Tensor))
        # Get random style codes
        s_code = np.random.uniform(-1, 1, (opt.style_dim, opt.style_dim))
        s_code = Variable(Tensor(s_code))
        # Generate samples
        c_code_1, _ = Enc1(X1)
        X12 = Dec2(c_code_1, s_code)
        # Concatenate samples horisontally
        X12 = torch.cat([x for x in X12.data.cpu()], -1)
        img_sample = torch.cat((img1, X12), -1).unsqueeze(0)
        # Concatenate with previous samples vertically
        img_samples = img_sample if img_samples is None else torch.cat((img_samples, img_sample), -2)
    save_image(img_samples, "images/%s/%s.png" % (opt.dataset_name, batches_done), nrow=5, normalize=True)


# ----------
#  Training
# ----------

# Adversarial ground truths
valid = 1
fake = 0

prev_time = time.time()
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):

        # Set model input
        X1 = Variable(batch["A"].type(Tensor))
        X2 = Variable(batch["B"].type(Tensor))

        # Sampled style codes
        style_1 = Variable(torch.randn(X1.size(0), opt.style_dim, 1, 1).type(Tensor))
        style_2 = Variable(torch.randn(X1.size(0), opt.style_dim, 1, 1).type(Tensor))

        # -------------------------------
        #  Train Encoders and Generators
        # -------------------------------

        optimizer_G.zero_grad()

        # Get shared latent representation
        c_code_1, s_code_1 = Enc1(X1)
        c_code_2, s_code_2 = Enc2(X2)

        # Reconstruct images
        X11 = Dec1(c_code_1, s_code_1)
        X22 = Dec2(c_code_2, s_code_2)

        # Translate images
        X21 = Dec1(c_code_2, style_1)
        X12 = Dec2(c_code_1, style_2)

        # Cycle translation
        c_code_21, s_code_21 = Enc1(X21)
        c_code_12, s_code_12 = Enc2(X12)
        X121 = Dec1(c_code_12, s_code_1) if lambda_cyc > 0 else 0
        X212 = Dec2(c_code_21, s_code_2) if lambda_cyc > 0 else 0

        # Losses
        loss_GAN_1 = lambda_gan * D1.compute_loss(X21, valid)
        loss_GAN_2 = lambda_gan * D2.compute_loss(X12, valid)
        loss_ID_1 = lambda_id * criterion_recon(X11, X1)
        loss_ID_2 = lambda_id * criterion_recon(X22, X2)
        loss_s_1 = lambda_style * criterion_recon(s_code_21, style_1)
        loss_s_2 = lambda_style * criterion_recon(s_code_12, style_2)
        loss_c_1 = lambda_cont * criterion_recon(c_code_12, c_code_1.detach())
        loss_c_2 = lambda_cont * criterion_recon(c_code_21, c_code_2.detach())
        loss_cyc_1 = lambda_cyc * criterion_recon(X121, X1) if lambda_cyc > 0 else 0
        loss_cyc_2 = lambda_cyc * criterion_recon(X212, X2) if lambda_cyc > 0 else 0

        # Total loss
        loss_G = (
            loss_GAN_1
            + loss_GAN_2
            + loss_ID_1
            + loss_ID_2
            + loss_s_1
            + loss_s_2
            + loss_c_1
            + loss_c_2
            + loss_cyc_1
            + loss_cyc_2
        )

        loss_G.backward()
        optimizer_G.step()

        # -----------------------
        #  Train Discriminator 1
        # -----------------------

        optimizer_D1.zero_grad()

        loss_D1 = D1.compute_loss(X1, valid) + D1.compute_loss(X21.detach(), fake)

        loss_D1.backward()
        optimizer_D1.step()

        # -----------------------
        #  Train Discriminator 2
        # -----------------------

        optimizer_D2.zero_grad()

        loss_D2 = D2.compute_loss(X2, valid) + D2.compute_loss(X12.detach(), fake)

        loss_D2.backward()
        optimizer_D2.step()

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] ETA: %s"
            % (epoch, opt.n_epochs, i, len(dataloader), (loss_D1 + loss_D2).item(), loss_G.item(), time_left)
        )

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D1.step()
    lr_scheduler_D2.step()

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(Enc1.state_dict(), "saved_models/%s/Enc1_%d.pth" % (opt.dataset_name, epoch))
        torch.save(Dec1.state_dict(), "saved_models/%s/Dec1_%d.pth" % (opt.dataset_name, epoch))
        torch.save(Enc2.state_dict(), "saved_models/%s/Enc2_%d.pth" % (opt.dataset_name, epoch))
        torch.save(Dec2.state_dict(), "saved_models/%s/Dec2_%d.pth" % (opt.dataset_name, epoch))
        torch.save(D1.state_dict(), "saved_models/%s/D1_%d.pth" % (opt.dataset_name, epoch))
        torch.save(D2.state_dict(), "saved_models/%s/D2_%d.pth" % (opt.dataset_name, epoch))
