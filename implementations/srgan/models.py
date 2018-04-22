import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import vgg19

class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()

        vgg19_model = vgg19(pretrained=True)

        self.feature_extractor = nn.Sequential(*list(vgg19_model.children())[:-1])

    def forward(self, img):
        return self.feature_extractor(img)


##############################
#           RESNET
##############################

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class GeneratorResNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=9):
        super(GeneratorResNet, self).__init__()

        # Initial convolution block
        self.pre_blocks = nn.Sequential(
            nn.ReflectionPad2d(4),
            nn.Conv2d(in_channels, 64, 9),
            nn.ReLU(inplace=True)
        )

        res_blocks = []
        # Residual blocks
        for _ in range(n_residual_blocks):
            res_blocks += [ResidualBlock(64)]

        self.res_blocks = nn.Sequential(*res_blocks)

        self.post_blocks = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.InstanceNorm2d(64)
        )

        upsampling = []
        # Upsampling
        in_features = 64
        for out_features in [256, 256]:
            upsampling += [ nn.Upsample(scale_factor=2),
                            nn.Conv2d(in_features, out_features, 3, 1, 1),
                            nn.InstanceNorm2d(out_features),
                            nn.ReLU(inplace=True) ]
            in_features = out_features

        upsampling += [ nn.ReflectionPad2d(4),
                        nn.Conv2d(out_features, out_channels, 9),
                        nn.Tanh()]

        self.upsampling = nn.Sequential(*upsampling)

    def forward(self, x):
        out1 = self.pre_blocks(x)
        out = self.res_blocks(out1)
        out = self.post_blocks(out)
        out = torch.add(out, out1)
        out = self.upsampling(out)
        return out

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride, normalize):
            """Returns layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for out_filters, stride, normalize in [ (64, 2, False),
                                                (128, 2, True),
                                                (256, 2, True),
                                                (512, 2, True)]:
            layers.extend(discriminator_block(in_filters, out_filters, stride, normalize))
            in_filters = out_filters

        # Output layer
        layers.append(nn.Conv2d(out_filters, 1, 3, 1, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)
