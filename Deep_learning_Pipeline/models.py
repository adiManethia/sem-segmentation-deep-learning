# models.py
# This file defines our segmentation heroes: the U-Net and Attention U-Net!
# These are the brains that learn to spot grain boundaries in SEM images.
# Think of them as detectives piecing together a puzzle from our synthetic clues.

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import BASE_CHANNELS, DEVICE

# A building block with two convolutional layers—our basic tool!
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        """Double convolution block with ReLU and batch normalization.
        
        Args:
            in_ch (int): Number of input channels.
            out_ch (int): Number of output channels.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),  # First conv layer
            nn.BatchNorm2d(out_ch),                   # Normalize it
            nn.ReLU(inplace=True),                    # Activate!
            nn.Conv2d(out_ch, out_ch, 3, padding=1), # Second conv
            nn.BatchNorm2d(out_ch),                   # Normalize again
            nn.ReLU(inplace=True),                    # Another activation
        )

    def forward(self, x): 
        return self.net(x)  # Pass through the layers

# The classic U-Net for segmentation
class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=BASE_CHANNELS):
        """U-Net architecture for binary segmentation.
        
        Args:
            in_ch (int): Number of input channels (default: 1 for grayscale).
            out_ch (int): Number of output channels (default: 1 for binary mask).
            base (int): Base number of channels (from config).
        """
        super().__init__()
        self.d1 = DoubleConv(in_ch, base)        # First down block
        self.d2 = DoubleConv(base, base*2)       # Second down block
        self.d3 = DoubleConv(base*2, base*4)     # Third down block
        self.d4 = DoubleConv(base*4, base*8)     # Fourth down block
        self.pool = nn.MaxPool2d(2)              # Downsampling
        self.bottleneck = DoubleConv(base*8, base*16)  # Middle layer
        self.u4 = nn.ConvTranspose2d(base*16, base*8, 2, 2)  # First up
        self.u3 = nn.ConvTranspose2d(base*8, base*4, 2, 2)   # Second up
        self.u2 = nn.ConvTranspose2d(base*4, base*2, 2, 2)   # Third up
        self.u1 = nn.ConvTranspose2d(base*2, base, 2, 2)     # Final up
        self.up4 = DoubleConv(base*16, base*8)   # Up block 4
        self.up3 = DoubleConv(base*8, base*4)    # Up block 3
        self.up2 = DoubleConv(base*4, base*2)    # Up block 2
        self.up1 = DoubleConv(base*2, base)      # Up block 1
        self.out = nn.Conv2d(base, out_ch, 1)    # Output layer

    def forward(self, x):
        c1 = self.d1(x); p1 = self.pool(c1)      # Down 1
        c2 = self.d2(p1); p2 = self.pool(c2)     # Down 2
        c3 = self.d3(p2); p3 = self.pool(c3)     # Down 3
        c4 = self.d4(p3); p4 = self.pool(c4)     # Down 4
        bn = self.bottleneck(p4)                 # Bottleneck
        u4 = self.u4(bn); u4 = torch.cat([u4, c4], dim=1); u4 = self.up4(u4)  # Up 4
        u3 = self.u3(u4); u3 = torch.cat([u3, c3], dim=1); u3 = self.up3(u3)  # Up 3
        u2 = self.u2(u3); u2 = torch.cat([u2, c2], dim=1); u2 = self.up2(u2)  # Up 2
        u1 = self.u1(u2); u1 = torch.cat([u1, c1], dim=1); u1 = self.up1(u1)  # Up 1
        return self.out(u1)  # Final output

# Attention block to focus on important features
class AttentionBlock(nn.Module):
    def __init__(self, g_ch, x_ch, inter_ch):
        """Attention gate to highlight relevant features.
        
        Args:
            g_ch (int): Number of gate channels.
            x_ch (int): Number of input channels.
            inter_ch (int): Number of intermediate channels.
        """
        super().__init__()
        self.Wg = nn.Sequential(nn.Conv2d(g_ch, inter_ch, 1, bias=False), nn.BatchNorm2d(inter_ch))
        self.Wx = nn.Sequential(nn.Conv2d(x_ch, inter_ch, 1, bias=False), nn.BatchNorm2d(inter_ch))
        self.psi = nn.Sequential(nn.Conv2d(inter_ch, 1, 1, bias=True), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        psi = self.relu(self.Wg(g) + self.Wx(x))  # Combine and activate
        psi = self.psi(psi)  # Sigmoid for attention
        return x * psi  # Apply attention to input

# Attention U-Net with extra focus
class AttnUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=BASE_CHANNELS):
        """Attention U-Net with gated attention for better boundary detection.
        
        Args:
            in_ch (int): Number of input channels (default: 1).
            out_ch (int): Number of output channels (default: 1).
            base (int): Base number of channels (from config).
        """
        super().__init__()
        self.d1 = DoubleConv(in_ch, base)        # First down block
        self.d2 = DoubleConv(base, base*2)       # Second down block
        self.d3 = DoubleConv(base*2, base*4)     # Third down block
        self.d4 = DoubleConv(base*4, base*8)     # Fourth down block
        self.pool = nn.MaxPool2d(2)              # Downsampling
        self.bottleneck = DoubleConv(base*8, base*16)  # Middle layer
        self.u4 = nn.ConvTranspose2d(base*16, base*8, 2, 2)  # First up
        self.att4 = AttentionBlock(g_ch=base*8, x_ch=base*8, inter_ch=base*4)  # Attention 4
        self.up4 = DoubleConv(base*16, base*8)   # Up block 4
        self.u3 = nn.ConvTranspose2d(base*8, base*4, 2, 2)   # Second up
        self.att3 = AttentionBlock(g_ch=base*4, x_ch=base*4, inter_ch=base*2)  # Attention 3
        self.up3 = DoubleConv(base*8, base*4)    # Up block 3
        self.u2 = nn.ConvTranspose2d(base*4, base*2, 2, 2)   # Third up
        self.att2 = AttentionBlock(g_ch=base*2, x_ch=base*2, inter_ch=base)  # Attention 2
        self.up2 = DoubleConv(base*4, base*2)    # Up block 2
        self.u1 = nn.ConvTranspose2d(base*2, base, 2, 2)     # Final up
        self.att1 = AttentionBlock(g_ch=base, x_ch=base, inter_ch=base//2)  # Attention 1
        self.up1 = DoubleConv(base*2, base)      # Up block 1
        self.out = nn.Conv2d(base, out_ch, 1)    # Output layer

    def forward(self, x):
        c1 = self.d1(x); p1 = self.pool(c1)      # Down 1
        c2 = self.d2(p1); p2 = self.pool(c2)     # Down 2
        c3 = self.d3(p2); p3 = self.pool(c3)     # Down 3
        c4 = self.d4(p3); p4 = self.pool(c4)     # Down 4
        bn = self.bottleneck(p4)                 # Bottleneck
        u4 = self.u4(bn); c4a = self.att4(u4, c4)  # Up 4 with attention
        u4 = torch.cat([u4, c4a], dim=1); u4 = self.up4(u4)
        u3 = self.u3(u4); c3a = self.att3(u3, c3)  # Up 3 with attention
        u3 = torch.cat([u3, c3a], dim=1); u3 = self.up3(u3)
        u2 = self.u2(u3); c2a = self.att2(u2, c2)  # Up 2 with attention
        u2 = torch.cat([u2, c2a], dim=1); u2 = self.up2(u2)
        u1 = self.u1(u2); c1a = self.att1(u1, c1)  # Up 1 with attention
        u1 = torch.cat([u1, c1a], dim=1); u1 = self.up1(u1)
        return self.out(u1)  # Final output

def get_model(name="unet"):
    """Get the appropriate model based on the given name.
    
    Args:
        name (str): Model name, either 'unet' or 'attnunet' (default: 'unet').
    
    Returns:
        nn.Module: The selected model instance.
    
    Raises:
        ValueError: If the name is not 'unet' or 'attnunet'.
    """
    name = name.lower().strip()
    if name == "unet":
        return UNet().to(DEVICE)
    elif name == "attnunet":
        return AttnUNet().to(DEVICE)
    else:
        raise ValueError("name must be 'unet' or 'attnunet'")