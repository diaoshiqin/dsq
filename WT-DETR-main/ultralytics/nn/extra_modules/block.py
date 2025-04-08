import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



__all__ = ['WaveletUnPool','wtd']

class wtd(nn.Module):
    def __init__(self,out_ch):
        super(wtd, self).__init__()
        ll = np.array([[0.5, 0.5], [0.5, 0.5]])
        lh = np.array([[-0.5, -0.5], [0.5, 0.5]])
        hl = np.array([[-0.5, 0.5], [-0.5, 0.5]])
        hh = np.array([[0.5, -0.5], [-0.5, 0.5]])
        filts = np.stack([ll[None,::-1,::-1], lh[None,::-1,::-1],
                            hl[None,::-1,::-1], hh[None,::-1,::-1]],
                            axis=0)
        self.weight = nn.Parameter(
            torch.tensor(filts).to(torch.get_default_dtype()),
            requires_grad=False)
        self.conv_bn_relu = nn.Sequential(
                                    nn.BatchNorm2d(out_ch),
                                    nn.ReLU(inplace=True),
                                    )
    def forward(self, x):
        C = x.shape[1]
        filters = torch.cat([self.weight,] * C, dim=0)
        y = F.conv2d(x, filters, groups=C, stride=2)
        y = self.conv_bn_relu(y)
        return y



class WaveletUnPool(nn.Module):
    def __init__(self,out_ch):
        super(WaveletUnPool, self).__init__()
        ll = np.array([[0.5, 0.5], [0.5, 0.5]])
        lh = np.array([[-0.5, -0.5], [0.5, 0.5]])
        hl = np.array([[-0.5, 0.5], [-0.5, 0.5]])
        hh = np.array([[0.5, -0.5], [-0.5, 0.5]])
        filts = np.stack([ll[None, ::-1, ::-1], lh[None, ::-1, ::-1],
                            hl[None, ::-1, ::-1], hh[None, ::-1, ::-1]],
                            axis=0)
        self.weight = nn.Parameter(
            torch.tensor(filts).to(torch.get_default_dtype()),
            requires_grad=False)
        self.conv_bn_relu = nn.Sequential(
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        C = torch.floor_divide(x.shape[1], 4)
        filters = torch.cat([self.weight, ] * C, dim=0)
        y = F.conv_transpose2d(x, filters, groups=C, stride=2)
        y = self.conv_bn_relu(y)
        return y


