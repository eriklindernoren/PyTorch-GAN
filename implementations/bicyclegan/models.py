import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

from torchvision.models import resnet18

##############################
#           U-NET
##############################

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        model = [   nn.Conv2d(in_size, out_size, 3, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        if normalize:
            model += [nn.BatchNorm2d(out_size, 0.8)]

        if dropout:
            model += [nn.Dropout(dropout)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        model = [   nn.Upsample(scale_factor=2),
                    nn.Conv2d(in_size, out_size, 3, stride=1, padding=1),
                    nn.BatchNorm2d(out_size, 0.8),
                    nn.LeakyReLU(0.2, inplace=True) ]
        if dropout:
            model += [nn.Dropout(dropout)]

        self.model = nn.Sequential(*model)

    def forward(self, x, skip_input):
        x = self.model(x)
        out = torch.cat((x, skip_input), 1)
        return out

class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        channels, self.h, self.w = img_shape

        self.fc = nn.Linear(latent_dim, self.h * self.w)

        self.down1 = UNetDown(channels+1, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256, dropout=0.5)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 256, dropout=0.5)
        self.up4 = UNetUp(512, 128)
        self.up5 = UNetUp(256, 64)


        final = [   nn.Upsample(scale_factor=2),
                    nn.Conv2d(128, channels, 3, 1, 1),
                    nn.Tanh() ]
        self.final = nn.Sequential(*final)

    def forward(self, x, z):
        # Propogate noise through fc layer and reshape to img shape
        z_ = self.fc(z).view(z.size(0), 1, self.h, self.w)
        d1 = self.down1(torch.cat((x, z_), 1))
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        u1 = self.up1(d6, d5)
        u2 = self.up2(u1, d4)
        u3 = self.up3(u2, d3)
        u4 = self.up4(u3, d2)
        u5 = self.up5(u4, d1)

        return self.final(u5)

##############################
#        Encoder
##############################

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()

        resnet18_model = resnet18(pretrained=True)

        # Extracts features at the last fully-connected
        self.feature_extractor = nn.Sequential(*list(resnet18_model.children())[:-1])
        # Final fully-connected layer
        self.fc = nn.Linear(512*2*2, latent_dim)

    def forward(self, img):
        out = self.feature_extractor(img)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


##############################
#        Discriminator
##############################

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride, normalize):
            """Returns layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, 2, False),
            *discriminator_block(64, 128, 2, True),
            *discriminator_block(128, 256, 2, True),
            *discriminator_block(256, 512, 2, True),
            nn.Conv2d(512, 1, 3, 1, 1)
        )

    def forward(self, img):
        # Concatenate image and condition image by channels to produce input
        return self.model(img)
