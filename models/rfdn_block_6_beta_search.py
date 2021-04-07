import torch.nn as nn
from models.operations_hh import OPS
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import ipdb

PRIMITIVES = [
    'conv_1x1',
    'conv_3x3',
    'conv_5x5',
    'conv_7x7',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'sep_conv_7x7',
    'dil_conv_3x3',
    'dil_conv_5x5'
]
UPSAMPLING = [
    'Bilinear',
    'Nearest',
    'Deconvolution',
    'manual_sub_pixel'
]

def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)


class MixedLayer(nn.Module):
    def __init__(self, cin):
        super(MixedLayer, self).__init__()
        self.layers = nn.ModuleList()
        for primitive in PRIMITIVES:
            layer = OPS[primitive](cin)
            self.layers.append(layer)
    def forward(self, x, alpha):
        res = [a * layer(x) for a, layer in zip(alpha, self.layers)]
        res = sum(res)
        return res

def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu'):
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                  dilation=dilation, bias=bias, groups=groups)
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


class ShortcutBlock(nn.Module):
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))

def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)

def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

class ESA(nn.Module):
    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False) 
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3+cf)
        m = self.sigmoid(c4)
        
        return x * m


class RFDB(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(RFDB, self).__init__()
        self.dc = self.distilled_channels = in_channels//2
        self.rc = self.remaining_channels = in_channels
        self.c1_d = conv_layer(in_channels, self.dc, 1)
        self.c1_r = MixedLayer(in_channels)
        self.c2_d = conv_layer(self.remaining_channels, self.dc, 1)
        self.c2_r = MixedLayer(self.remaining_channels)
        self.c3_d = conv_layer(self.remaining_channels, self.dc, 1)
        self.c3_r = MixedLayer(self.remaining_channels)
        self.c4 = conv_layer(self.remaining_channels, self.dc, 3)
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer(self.dc*4, in_channels, 1)
        self.esa = ESA(in_channels, nn.Conv2d)

    def forward(self, input, alphas):
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input,alphas[0]))
        r_c1 = self.act(r_c1+input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1,alphas[1]))
        r_c2 = self.act(r_c2+r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.c3_r(r_c2,alphas[2]))
        r_c3 = self.act(r_c3+r_c2)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out_fused = self.esa(self.c5(out)) 

        return out_fused



def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)


def make_model(args, parent=False):
    model = RFDN()
    return model


class RFDN(nn.Module):
    def __init__(self, in_nc=3, nf=48, num_modules=6, out_nc=3, upscale=4):
        super(RFDN, self).__init__()

        self.fea_conv = conv_layer(in_nc, nf, kernel_size=3)

        self.B1 = RFDB(in_channels=nf)
        self.B2 = RFDB(in_channels=nf)
        self.B3 = RFDB(in_channels=nf)
        self.B4 = RFDB(in_channels=nf)
        self.B5 = RFDB(in_channels=nf)
        self.B6 = RFDB(in_channels=nf)
        self.c = conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')

        self.LR_conv = conv_layer(nf, nf, kernel_size=3)

        upsample_block = pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=2)
        self.scale_idx = 0

        num_ops = len(PRIMITIVES)
        self.alphas = Variable(1e-3 * torch.randn(3, num_ops).cuda(), requires_grad=True)
        self._arch_parameters = [self.alphas]

    def forward(self, input):
        out_fea = self.fea_conv(input)
        weights = F.softmax(self.alphas, dim=-1)
        out_B1 = self.B1(out_fea,weights)
        out_B2 = self.B2(out_B1,weights)
        out_B3 = self.B3(out_B2,weights)
        out_B4 = self.B4(out_B3,weights)
        out_B5 = self.B5(out_B4,weights)
        out_B6 = self.B6(out_B5,weights)
        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea

        output = self.upsampler(out_lr)

        return output

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):
        alphas = F.softmax(self.alphas, dim=-1)
        genotype = []
        for alpha in alphas:
            alpha_sorted = alpha.sort(descending=True)
            max_index = alpha_sorted[1][0]
            genotype.append(PRIMITIVES[max_index])
        return genotype


class RFDN_beta1(nn.Module):
    def __init__(self, in_nc=3, nf=48, num_modules=6, out_nc=3, upscale=4):
        super(RFDN_beta1, self).__init__()

        self.fea_conv = conv_layer(in_nc, nf, kernel_size=3)

        self.B1 = RFDB(in_channels=nf)
        self.B2 = RFDB(in_channels=nf)
        self.B3 = RFDB(in_channels=nf)
        self.B4 = RFDB(in_channels=nf)
        self.B5 = RFDB(in_channels=nf)
        self.B6 = RFDB(in_channels=nf)
        self.c2 = conv_block(nf * 2, nf, kernel_size=1, act_type='lrelu')
        self.c3 = conv_block(nf * 3, nf, kernel_size=1, act_type='lrelu')
        self.c4 = conv_block(nf * 4, nf, kernel_size=1, act_type='lrelu')
        self.c5 = conv_block(nf * 5, nf, kernel_size=1, act_type='lrelu')
        self.c6 = conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')

        self.LR_conv = conv_layer(nf, nf, kernel_size=3)

        upsample_block = pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=2)
        self.scale_idx = 0

        num_ops = len(PRIMITIVES)
        self.alphas = Variable(1e-3 * torch.randn(3, num_ops).cuda(), requires_grad=True)
        self.betas = []
        for i in range(2,6):
            self.betas.append(Variable(1e-3 * torch.randn(i).cuda(), requires_grad=True))
        #self._arch_parameters = [self.alphas,self.betas]
        self._arch_parameters = [self.alphas]
        self._arch_parameters.extend(self.betas)

    def forward(self, input):
        out_fea = self.fea_conv(input)
        weights = F.softmax(self.alphas, dim=-1)
        out_B1 = self.B1(out_fea,weights)
        out_B2 = self.B2(out_B1,weights)
        #ipdb.set_trace()
        beta_2 = F.softmax(self.betas[0], dim=-1)
        in_B3 = self.c2(torch.cat([beta_2[0] * out_B1, beta_2[1] * out_B2],dim=1))
        out_B3 = self.B3(in_B3,weights)

        beta_3 = F.softmax(self.betas[1], dim=-1)
        in_B4 = self.c3(torch.cat([beta_3[0] * out_B1, beta_3[1] * out_B2, beta_3[2] * out_B3],dim=1))
        out_B4 = self.B4(in_B4,weights)

        beta_4 = F.softmax(self.betas[2], dim=-1)
        in_B5 = self.c4(torch.cat([beta_4[0] * out_B1, beta_4[1] * out_B2, beta_4[2] * out_B3, beta_4[3] * out_B4],dim=1))
        out_B5 = self.B5(in_B5,weights)

        beta_5 = F.softmax(self.betas[3], dim=-1)
        in_B6 = self.c5(torch.cat([beta_5[0] * out_B1, beta_5[1] * out_B2, beta_5[2] * out_B3,
                                   beta_5[3] * out_B4, beta_5[4] * out_B5], dim=1))
        out_B6 = self.B6(in_B6,weights)

        out_B = self.c6(torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea

        output = self.upsampler(out_lr)

        return output

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):
        alphas = F.softmax(self.alphas, dim=-1)
        genotype = []
        for alpha in alphas:
            alpha_sorted = alpha.sort(descending=True)
            max_index = alpha_sorted[1][0]
            genotype.append(PRIMITIVES[max_index])

        for beta in self.betas:
            beta_s = F.softmax(beta,dim=-1)
            beta_sorted = beta_s.sort(descending=True)
            max_beta = beta_sorted[1][0]
            second_max_beta = beta_sorted[1][1]
            genotype.append(str([max_beta,second_max_beta]))

        return genotype


class RFDN_beta2(nn.Module):
    def __init__(self, in_nc=3, nf=48, num_modules=6, out_nc=3, upscale=4):
        super(RFDN_beta2, self).__init__()

        self.fea_conv = conv_layer(in_nc, nf, kernel_size=3)

        self.B1 = RFDB(in_channels=nf)
        self.B2 = RFDB(in_channels=nf)
        self.B3 = RFDB(in_channels=nf)
        self.B4 = RFDB(in_channels=nf)
        self.B5 = RFDB(in_channels=nf)
        self.B6 = RFDB(in_channels=nf)
        self.c2 = conv_block(nf * 2, nf, kernel_size=1, act_type='lrelu')
        self.c3 = conv_block(nf * 3, nf, kernel_size=1, act_type='lrelu')
        self.c4 = conv_block(nf * 4, nf, kernel_size=1, act_type='lrelu')
        self.c5 = conv_block(nf * 5, nf, kernel_size=1, act_type='lrelu')
        self.c6 = conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')

        self.LR_conv = conv_layer(nf, nf, kernel_size=3)

        upsample_block = pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=2)
        self.scale_idx = 0

        num_ops = len(PRIMITIVES)
        self.alphas = Variable(1e-3 * torch.randn(3, num_ops).cuda(), requires_grad=True)
        self.betas = []
        for i in range(2, 5):
            self.betas.append(Variable(1e-3 * torch.randn(i).cuda(), requires_grad=True))
        #self._arch_parameters = [self.alphas, self.betas]
        self._arch_parameters = [self.alphas]
        self._arch_parameters.extend(self.betas)

    def forward(self, input):
        out_fea = self.fea_conv(input)
        weights = F.softmax(self.alphas, dim=-1)
        out_B1 = self.B1(out_fea, weights)
        out_B2 = self.B2(out_B1, weights)

        in_B3 = self.c2(torch.cat([out_B1, out_B2],dim=1))
        out_B3 = self.B3(in_B3, weights)

        beta_3 = F.softmax(self.betas[0], dim=-1)
        in_B4 = self.c3(torch.cat([beta_3[0] * out_B1, beta_3[1] * out_B2, out_B3],dim=1))
        out_B4 = self.B4(in_B4, weights)

        beta_4 = F.softmax(self.betas[1], dim=-1)
        in_B5 = self.c4(torch.cat([beta_4[0] * out_B1, beta_4[1] * out_B2, beta_4[2] * out_B3, out_B4],dim=1))
        out_B5 = self.B5(in_B5, weights)

        beta_5 = F.softmax(self.betas[2], dim=-1)
        in_B6 = self.c5(torch.cat([beta_5[0] * out_B1, beta_5[1] * out_B2, beta_5[2] * out_B3,
                                   beta_5[3] * out_B4, out_B5],dim=1))
        out_B6 = self.B6(in_B6, weights)

        out_B = self.c6(torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea

        output = self.upsampler(out_lr)

        return output

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):
        alphas = F.softmax(self.alphas, dim=-1)
        genotype = []
        for alpha in alphas:
            alpha_sorted = alpha.sort(descending=True)
            max_index = alpha_sorted[1][0]
            genotype.append(PRIMITIVES[max_index])

        for beta in self.betas:
            beta_s = F.softmax(beta, dim=-1)
            beta_sorted = beta_s.sort(descending=True)
            max_beta = beta_sorted[1][0]
            second_max_beta = beta_sorted[1][1]
            genotype.append(str([max_beta, second_max_beta]))

        return genotype
