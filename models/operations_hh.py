import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys


OPS = {
    'skip_connect': lambda C_in : Identity(),
    'conv_1x1': lambda C_in: Conv(C_in, C_in, 1, 1, 0),
    'conv_3x3': lambda C_in : Conv(C_in, C_in, 3, 1, 1),
    'conv_5x5': lambda C_in : Conv(C_in, C_in, 5, 1, 2),
    'conv_7x7': lambda C_in : Conv(C_in, C_in, 7, 1, 3),
    'sep_conv_3x3': lambda C_in: SepConv(C_in, C_in, 3, 1, 1),
    'sep_conv_5x5': lambda C_in: SepConv(C_in, C_in, 5, 1, 2),
    'sep_conv_7x7': lambda C_in: SepConv(C_in,C_in, 7, 1, 3),
    'dil_conv_3x3': lambda C_in: DilConv(C_in,C_in, 3, 1, 2, 2),
    'dil_conv_5x5': lambda C_in: DilConv(C_in,C_in, 5, 1, 4, 2),
    'rcab': lambda C: RCAB(C, 3, 3, bias=True, bn=True, act=nn.ReLU(True), res_scale=1),
    #'sub_pixel':lambda C, stride: SUBPIXEL(C, scale_factor=stride),
    'manual_sub_pixel':lambda C, stride: SUBPIXEL(C, scale_factor=2),
    'Deconvolution':lambda C, stride: Deconvolution(C,stride),
    'Bilinear':lambda stride: Bilinear(stride),
    'Nearest':lambda stride: Nearest(stride),
    'Linear':lambda stride: Linear(stride),
    'area':lambda stride: Area(stride),
}

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class Conv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super(Conv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
        )

    def forward(self, x):
        return self.op(x)

class ReLUConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super(ReLUConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            # nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)

class DilConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            # nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)

class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            # nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            # nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)

# class Skip(nn.Module):
#     def __init__(self, C_in, C_out, stride):
#         super(Skip, self).__init__()
#
#         if C_in!=C_out:
#             skip_conv = nn.Conv2d(C_in, C_out, kernel_size=1, stride=stride, padding=0, groups=1, bias=False),
#             # nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats))
#             self.op = Identity(1)
#             self.op = nn.Sequential(skip_conv, self.op)
#         else:
#             self.op = Identity(stride)
#
#     def forward(self,x):
#         return self.op(x)
#
# class Identity(nn.Module):
#     def __init__(self, stride):
#         super(Identity, self).__init__()
#         self.stride = stride
#
#     def forward(self, x):
#         if self.stride == 1:
#             return x
#         else:
#             return x[:, :, ::self.stride, ::self.stride]
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self,  n_feat, kernel_size, reduction,conv=default_conv,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            # if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

#upsample
class Bilinear(nn.Module):
    def __init__(self, stride):
        super(Bilinear, self).__init__()
        self.scale=stride

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale, mode='bilinear')

class Linear(nn.Module):
    def __init__(self, stride):
        super(Linear, self).__init__()
        self.scale=stride

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale, mode='linear')

class Area(nn.Module):
    def __init__(self, stride):
        super(Area, self).__init__()
        self.scale=stride

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale, mode='area')

class Nearest(nn.Module):
    def __init__(self, stride):
        super(Nearest, self).__init__()
        self.scale=stride

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale, mode='nearest')

class Deconvolution(nn.Module):
    def __init__(self, C, stride):
        super(Deconvolution, self).__init__()
        if stride==2:
            kernel_size=3
            output_padding=1
        elif stride==4:
            kernel_size=5
            output_padding = 1
        else:
            kernel_size=3
            output_padding = 0
        self.deconv=nn.ConvTranspose2d(C, C,kernel_size=kernel_size,stride=stride, padding=1,output_padding=output_padding)

    def forward(self, x):
        return self.deconv(x)


class Upsampler1(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                # if bn:
                #     m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            # if bn:
            #     m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler1, self).__init__(*m)

class SUBPIXEL(nn.Module):
    def __init__(self, C, scale_factor,conv=default_conv):
        super(SUBPIXEL, self).__init__()
        self.subpixel = nn.ModuleList()
        self.subpixel.append(Upsampler1(conv, scale_factor, C, act=False))

    def forward(self, x):
        return self.subpixel[0](x)
