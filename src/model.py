import torch
from torch import nn
from torch.nn import functional as F

def block(a,b):
    return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.ReLU(),nn.Conv2d(b,b,3,padding=1),nn.ReLU())

class Segmenter(nn.Module):
    """Same weights and widths in both variants; only additive skip routing changes."""
    def __init__(self,skip=False):
        super().__init__(); self.skip=skip
        self.e1=block(3,8); self.e2=block(8,16); self.b=block(16,32)
        self.p2=nn.Conv2d(32,16,1);self.d2=block(16,16)
        self.p1=nn.Conv2d(16,8,1);self.d1=block(8,8);self.head=nn.Conv2d(8,2,1)
    def forward(self,x):
        e1=self.e1(x);e2=self.e2(F.max_pool2d(e1,2));b=self.b(F.max_pool2d(e2,2))
        u2=self.p2(F.interpolate(b,scale_factor=2,mode='bilinear',align_corners=False))
        d2=self.d2(u2+e2 if self.skip else u2)
        u1=self.p1(F.interpolate(d2,scale_factor=2,mode='bilinear',align_corners=False))
        return self.head(self.d1(u1+e1 if self.skip else u1))
