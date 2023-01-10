import torch
import torch.nn as nn
import torchvision.transforms.functional as F

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down_sample_after):
        super().__init__()
        print(f"next layer IN: {in_channels} OUT: {out_channels}")
        self.operations = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, groups=1, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, groups=1, kernel_size=3, padding='same'),
            nn.ReLU(),
        )
        if down_sample_after:
            self.operations.append(
                nn.MaxPool2d(2),
            )

    def forward(self, x):
        self.operations(x)

# UpBlock uses Upsample and will always result in 2x width and height
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.double_conv = DownBlock(in_channels, out_channels, False)

    # y is a copy of the feature map from contracting path
    def forward(self, x, y):
        x = self.up(x)
        x = torch.cat([y, x], dim=1)
        return self.double_conv(x)


class UNet(nn.Module):
    # This constructor is set up to match the original unet paper.
    # "side" refers to the width and height, which for now will match
    def __init__(self, in_channels=3, out_channels=1, side=572, depth=4):
        super().__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.depth=depth

        # we'll construct these - they contain details about the different
        # expanding and contracting layers, as well as the middle one
        self.downs = []
        self.middle = None
        self.ups = []
        self.last = None

        # validate that side and depth work together
        if side % (2 * depth) != 0:
            raise Exception(f"dims {side}x{side} not compatible with depth {depth}")

        # 1. Encoding layers
        before_channels = in_channels
        after_channels = 64
        for i in range(depth):
            self.downs.append(DownBlock(before_channels, after_channels, True))
            if i < depth - 1:
                before_channels = after_channels
                after_channels = after_channels * 2

        # 2. Middle layer at bottom of the U
        self.middle = DownBlock(after_channels, after_channels, False)

        # 3. Decoding layers
        before_channels, after_channels = after_channels, before_channels
        for i in range(depth):
            self.ups.append(UpBlock(before_channels, after_channels))
            if i < depth - 1:
                before_channels = after_channels
                after_channels = int(after_channels / 2)
        
        # 4. Last
        self.last_double = DownBlock(after_channels, after_channels, False)
        self.last_single = nn.Conv2d(after_channels, out_channels, kernel_size=1, padding='same'),

    def forward(self, x):
        # validate
        if self.side != x.shape[2] or self.side != x.shape[3]:
            raise Exception(f"side is {self.side}, but input image had shape {x.shape}")

        # 1. Encoding layers
        downs = []
        for d in self.downs:
            x = d(x)
            downs.append(x)

        # 2. Middle layer
        x = self.middle(x)

        # 3. Decoding layers
        idx = 0
        for u in self.ups:
            x = u(x, downs[idx])
            idx + 1

        # 4. Last
        x = self.last_double(x)
        return self.last_single(x)

        
