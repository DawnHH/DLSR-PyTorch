import torch, os, sys, cv2
import torch.nn as nn
from torch.nn import init
import functools
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as func
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch
import math
import vgg19

def LoG(img):
	weight = [
        [0,1,0],
        [1,-4,1],
        [0,1,0]
	]
	weight = np.array(weight)

	weight_np = np.zeros((1, 1, 3, 3))
	weight_np[0, 0, :, :] = weight
	weight_np = np.repeat(weight_np, img.shape[1], axis=1)
	weight_np = np.repeat(weight_np, img.shape[0], axis=0)

	weight = torch.from_numpy(weight_np).type(torch.FloatTensor).cuda()

	return func.conv2d(img, weight, padding=1)

def get_gaussian_kernel(kernel_size=3, sigma=2, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2*variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, bias=False)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter.cuda()


def HFEN(output, target):
	filter = get_gaussian_kernel(5,1.5,3)
	gradient_p = filter(target)
	gradient_o = filter(output)
	gradient_p = LoG(gradient_p)
	if torch.max(gradient_p) != 0:
		gradient_p = gradient_p/torch.max(gradient_p)
	gradient_o = LoG(gradient_o)
	if torch.max(gradient_o) !=0:
		gradient_o = gradient_o/torch.max(gradient_o)
	criterion = nn.L1Loss()
	return criterion(gradient_p, gradient_o)


def l1_norm(output, target):
	criterion = nn.L1Loss()
	return criterion(target,output)


def temporal_norm(output, target):
	criterion = nn.L1Loss()
	return criterion(target,output)

def feature_loss(gpu_ids,feature_criterion, output,target,netF):
	l_fea_type = feature_criterion

	if l_fea_type == 'l1':
		cri_fea = nn.L1Loss().cuda()
	elif l_fea_type == 'l2':
		cri_fea = nn.MSELoss().cuda()
	else:
		raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_fea_type))



	# if cri_fea:  # load VGG perceptual loss
	# 	netF = vgg19.define_F(gpu_ids, use_bn=False).cuda()
		# if opt['dist']:
		# 	self.netF = DistributedDataParallel(self.netF,
		# 										device_ids=[torch.cuda.current_device()])
		# else:
		# 	self.netF = DataParallel(self.netF)
		# feature loss
	real_fea = netF(target).detach()
	fake_fea = netF(output)
	l_g_fea = cri_fea(fake_fea, real_fea)
	return l_g_fea

def l1_perceptual_loss(output, target, gpu_ids,feature_criterion,netF):
	ls = l1_norm(output, target)
	lp = feature_loss(gpu_ids,feature_criterion, output,target,netF)
	lg = HFEN(output, target)
	#lt = temporal_norm(temporal_output, temporal_target)

	return 0.6*ls+0.3*lp+0.1*lg, ls, lp, lg
