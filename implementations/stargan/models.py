import torch.nn as nn
import torch.nn.functional as F
import torch

##############################
#           RESNET
##############################

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features, affine=True, track_running_stats=True),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features, affine=True, track_running_stats=True)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class GeneratorResNet(nn.Module):
    def __init__(self, img_shape=(3, 128, 128), res_blocks=9, c_dim=5):
        super(GeneratorResNet, self).__init__()
        channels, img_size, _ = img_shape

        # Initial convolution block
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(channels + c_dim, 64, 7),
                    nn.InstanceNorm2d(64, affine=True, track_running_stats=True),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        curr_dim = 64
        for _ in range(2):
            model += [  nn.Conv2d(curr_dim, curr_dim*2, 4, stride=2, padding=1),
                        nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True),
                        nn.ReLU(inplace=True) ]
            curr_dim *= 2

        # Residual blocks
        for _ in range(res_blocks):
            model += [ResidualBlock(curr_dim)]

        # Upsampling
        for _ in range(2):
            model += [  nn.ConvTranspose2d(curr_dim, curr_dim//2, 4, stride=2, padding=1),
                        nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True),
                        nn.ReLU(inplace=True) ]
            curr_dim = curr_dim // 2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(curr_dim, channels, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x, c):
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat((x, c), 1)
        return self.model(x)


##############################
#        Discriminator
##############################

class Discriminator(nn.Module):
    def __init__(self, img_shape=(3, 128, 128), c_dim=5):
        super(Discriminator, self).__init__()
        channels, img_size, _ = img_shape

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
        )

        self.out1 = nn.Conv2d(512, 1, 3, padding=1)
        self.out2 = nn.Conv2d(512, c_dim, 3, padding=1)

    def forward(self, img):
        feature_repr = self.model(img)
        out_adv = self.out1(feature_repr)
        out_cls = self.out2(feature_repr)
        return out_adv, out_cls
