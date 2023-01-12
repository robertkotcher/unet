import torch
import torch.nn as nn
import torchvision.transforms.functional as F

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down_sample_after):
        super().__init__()
        print(f"next layer IN: {in_channels} OUT: {out_channels}")
        self.c0 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same')
        self.b0 = nn.BatchNorm2d(out_channels)
        self.r0 = nn.ReLU()
        self.c1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same')
        self.b1 = nn.BatchNorm2d(out_channels)
        self.r1 = nn.ReLU()

        self.down_sample_after = down_sample_after
        self.mp = nn.MaxPool2d(2)
        

    def forward(self, x):
        x = self.c0(x)
        x = self.b0(x)
        x = self.r0(x)
        x = self.c1(x)
        x = self.b1(x)
        x = self.r1(x)

        if self.down_sample_after:
            return x, self.mp(x)
        else:
            return x

# XXXXX I'm not sure how to get Upsample to give me half as many channels.... problem for another
# day... I'll just use ConvTranspose2d right now
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.double_conv = DownBlock(in_channels, out_channels, False)

    # y is a copy of the feature map from contracting path
    def forward(self, x, y):
        x = self.up(x)
        x = torch.cat([y, x], dim=1)
        return self.double_conv(x)


# UNet always has 4 max pools
class UNet(nn.Module):
    # This constructor is set up to match the original unet paper.
    # "side" refers to the width and height, which for now will match
    def __init__(self, in_channels=3, out_channels=1, side=572):
        super().__init__()
        self.side=side
        self.in_channels=in_channels
        self.out_channels=out_channels

        self.down0 = DownBlock(in_channels, 64, True)
        self.down1 = DownBlock(64, 128, True)
        self.down2 = DownBlock(128, 256, True)
        self.down3 = DownBlock(256, 512, True)

        self.middle = DownBlock(512, 1024, False)

        self.up0 = UpBlock(1024, 512)
        self.up1 = UpBlock(512, 256)
        self.up2 = UpBlock(256, 128)
        self.up3 = UpBlock(128, 64)

        self.last_double = DownBlock(64, 64, False)
        
        self.last_single = nn.Conv2d(64, out_channels, kernel_size=1, padding='same')

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # validate
        if self.side != x.shape[2] or self.side != x.shape[3]:
            raise Exception(f"side is {self.side}, but input image had shape {x.shape}")

        x_cat_3, x0 = self.down0(x)
        x_cat_2, x1 = self.down1(x0)
        x_cat_1, x2 = self.down2(x1)
        x_cat_0, x3 = self.down3(x2)

        x = self.middle(x3)

        x = self.up0(x, x_cat_0)
        x = self.up1(x, x_cat_1)
        x = self.up2(x, x_cat_2)
        x = self.up3(x, x_cat_3)

        x = self.last_double(x)
        x = self.last_single(x)
        return self.sigmoid(x)
        
