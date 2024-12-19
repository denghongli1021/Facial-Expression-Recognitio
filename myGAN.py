import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
import os
from PIL import Image
import itertools
import torch.autograd as autograd



# Step 1: Define the Generator and Discriminator for myGAN

class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(True),

            ResidualBlock(dim_in=64, dim_out=64),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(128, affine=True, track_running_stats=True),
            nn.ReLU(True),

            ResidualBlock(dim_in=128, dim_out=128),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(256, affine=True, track_running_stats=True),
            nn.ReLU(True),
            # nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.InstanceNorm2d(512, affine=True, track_running_stats=True),
            # nn.ReLU(True),

            # Bottleneck layers.
            ResidualBlock(dim_in=256, dim_out=256),
            ResidualBlock(dim_in=256, dim_out=256),
            # ResidualBlock(dim_in=256, dim_out=256),
            # ResidualBlock(dim_in=256, dim_out=256),
            # ResidualBlock(dim_in=256, dim_out=256),
            # ResidualBlock(dim_in=256, dim_out=256),

            # nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.InstanceNorm2d(256, affine=True, track_running_stats=True),
            # nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(128, affine=True, track_running_stats=True),
            nn.ReLU(True),

            ResidualBlock(dim_in=128, dim_out=128),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(64, affine=True, track_running_stats=True),
            nn.ReLU(True),

            ResidualBlock(dim_in=64, dim_out=64),

            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=True),
            # nn.Conv2d(128, 3, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            # nn.LeakyReLU(0.01, inplace=False),

            # nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            # # nn.BatchNorm2d(128),
            # nn.LeakyReLU(0.01, inplace=False),
            # nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            # # nn.BatchNorm2d(256),
            # nn.LeakyReLU(0.01, inplace=False),

            # nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            # nn.LeakyReLU(0.01, inplace=False),

            # nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            # nn.LeakyReLU(0.01, inplace=False),

            # # nn.Conv2d(1024, 2048, kernel_size=4, stride=2, padding=1),
            # # nn.LeakyReLU(0.01, inplace=False),



            # nn.Conv2d(1024, 1, kernel_size=3, stride=1, padding=1, bias=False)
            # # nn.Sigmoid()


            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # nn.Conv2d(1024, 2048, kernel_size=4, stride=2, padding=1),
            # nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(1024, 1, kernel_size=4, stride=1, padding=1, bias=False)
            # nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0)
        )

    def forward(self, x):
        return self.main(x).view(-1)