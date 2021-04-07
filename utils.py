import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import math
import skimage


class AverageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform

def _data_transforms_cifar100(args):
  CIFAR_MEAN = [0.485, 0.456, 0.406]
  CIFAR_STD = [0.229, 0.224, 0.225]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform

def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

def print_model_parm_flops(model, input):
    # prods = {}
    # def save_prods(self, input, output):
    # print 'flops:{}'.format(self.__class__.__name__)
    # print 'input:{}'.format(input)
    # print '_dim:{}'.format(input[0].dim())
    # print 'input_shape:{}'.format(np.prod(input[0].shape))
    # grads.append(np.prod(input[0].shape))

    prods = {}

    def save_hook(name):
        def hook_per(self, input, output):
            # print 'flops:{}'.format(self.__class__.__name__)
            # print 'input:{}'.format(input)
            # print '_dim:{}'.format(input[0].dim())
            # print 'input_shape:{}'.format(np.prod(input[0].shape))
            # prods.append(np.prod(input[0].shape))
            prods[name] = np.prod(input[0].shape)
            # prods.append(np.prod(input[0].shape))

        return hook_per

    list_1 = []

    def simple_hook(self, input, output):
        list_1.append(np.prod(input[0].shape))

    list_2 = {}

    def simple_hook2(self, input, output):
        list_2['names'] = np.prod(input[0].shape)

    multiply_adds = False
    list_conv = []

    def conv_hook(self, input, output):
        batch_size, input_channels, input_time, input_height, input_width = input[0].size()
        output_channels, output_time, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2] * (
        self.in_channels / self.groups) * (2 if multiply_adds else 1)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_time * output_height * output_width
        list_conv.append(flops)

    list_linear = []

    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)

    list_bn = []

    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement())

    list_fc = []

    def fc_hook(self, input, output):
        list_bn.append(input[0].nelement())

    list_relu = []

    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    list_pooling = []

    # def pooling_hook(self, input, output):
    #   batch_size, input_channels, input_time,input_height, input_width = input[0].size()
    #  output_channels, output_time, output_height, output_width = output[0].size()

    # kernel_ops = self.kernel_size * self.kernel_size*self.kernel_size
    # bias_ops = 0
    # params = output_channels * (kernel_ops + bias_ops)
    # flops = batch_size * params * output_height * output_width * output_time

    # list_pooling.append(flops)



    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv3d):
                # net.register_forward_hook(save_hook(net.__class__.__name__))
                # net.register_forward_hook(simple_hook)
                # net.register_forward_hook(simple_hook2)
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm3d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
                # if isinstance(net, torch.nn.MaxPool3d) or isinstance(net, torch.nn.AvgPool2d):
                #   net.register_forward_hook(pooling_hook)
            return
        for c in childrens:
            foo(c)

    foo(model)
    output = model(input)
    total_flops = (sum(list_conv) + sum(list_linear))  # +sum(list_bn)+sum(list_relu))
    print('  + Number of FLOPs: %.5f(e9)' % (total_flops / 1e9))

def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path):
  model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)


def load_subgraph_from_model(sub_grah, ori_graph, selected_index, selected_alpha):
    ori_graph_paras = ori_graph.state_dict()
    sub_graph_paras = sub_grah.state_dict()

    # alpha = ori_graph._all_arch_parameters
    # selected_index, selected_alpha = sample_from_alpha(alpha, reduce_rate, with_noise)

    # if PC == True:
    #     selected_index[0] = torch.range(0, 15, dtype=torch.long).cuda()

    ind = 0
    pre_selected = [0, 1, 2]
    ############# load network parameters ##################
    for k, v in ori_graph_paras.items():
        if k in sub_graph_paras and len(v.shape) == 4:
            selected = selected_index[ind]
            if 'alpha' in k:
                sub_graph_paras[k] = v[selected, :, :, :]
            elif 'downsample' in k:
                sub_graph_paras[k] = v[pre_selected, :, :, :]
                a = selected_index[ind - 3]  #####for resnet34 there should be 3
                sub_graph_paras[k] = sub_graph_paras[k][:, a, :, :]
                # pre_selected = selected
            else:
                sub_graph_paras[k] = v[selected, :, :, :]
                sub_graph_paras[k] = sub_graph_paras[k][:, pre_selected, :, :]
                pre_selected = selected
                ind += 1
        elif k in sub_graph_paras:
            if 'fc' in k and len(v.shape) == 1:
                sub_graph_paras[k] = v
            elif 'fc' in k and len(v.shape) == 2:
                sub_graph_paras[k] = v[:, selected]
            else:
                sub_graph_paras[k] = v[pre_selected]
        else:
            print('Not Matching Key: {}'.format(k))

    sub_grah.load_state_dict(sub_graph_paras)
    ############# load network parameters ##################
    sub_grah._all_arch_parameters = selected_alpha

    return sub_grah

def sample_from_alpha_adaptive(alpha, keep_rate, with_noise):
    sampler_ind =[]
    sampler_alpha =[]
    sampler_len =[]
    cut_ind = []
    for paras in alpha:
        paras = paras.squeeze(3).squeeze(2).squeeze(1)
        # paras_abs = torch.abs(paras)

        paras_len = len(paras)
        noise = torch.log(-torch.log(torch.rand(paras_len ).cuda()))*0.05
        if with_noise:
            alpha_n = paras - noise
        else:
            alpha_n = paras
        alpha_soft = F.softmax(alpha_n, dim=-1)
        sorted_alpha = alpha_soft.sort(descending=True)

        def compute_len(alpha, keep_rate):
            # len = int(len(alpha)//reduce_rate)
            sum = 0
            ind = 0
            for i in alpha:
                sum+=i
                ind+=1
                if sum >= keep_rate:
                    return ind, sum

        reduced_len, summ = compute_len(sorted_alpha[0], keep_rate)

        max_alpha = sorted_alpha[0][:reduced_len]
        index = sorted_alpha[1][:reduced_len]
        cut_index = sorted_alpha[1][reduced_len:]

        sorted_index = index.sort()[0]
        sampler_ind.append(sorted_index)  # the index of the selceted parameters
        sampler_alpha.append(paras[sorted_index])  # the selected parameters which arranged sequentially
        sampler_len.append(reduced_len)
        cut_ind.append(cut_index)

        print('sum1:', sum(alpha_soft[sorted_index]), 'sum2:', summ)
    print('last_alpha:', max_alpha)
    return sampler_ind, sampler_alpha, sampler_len,cut_ind

def sample_from_alpha(alpha, reduce_rate, with_noise, PC=False):
    sampler_ind =[]
    sampler_alpha =[]
    for paras in alpha:
        paras = paras.squeeze(3).squeeze(2).squeeze(1)

        paras_len = len(paras)
        noise = torch.log(-torch.log(torch.rand(paras_len ).cuda()))*0.05
        if with_noise:
            alpha_n = paras - noise
        else:
            alpha_n = paras
        alpha_soft = F.softmax(alpha_n, dim=-1)
        sorted_alpha = alpha_soft.sort()

        reduced_len = int(len(alpha_soft)//reduce_rate)
        # if sampler_ind == [] and PC == True:
        #     reduced_len =int(len(alpha_soft))

        # max_alpha = sorted_alpha[0][-reduced_len:]
        index = sorted_alpha[1][-reduced_len:]

        sorted_index = index.sort()[0]
        sampler_ind.append(sorted_index)  # the index of the selceted parameters
        sampler_alpha.append(paras[sorted_index])  # the selected parameters which arranged sequentially

        print(sum(sorted_alpha[0][sorted_index]))


    return sampler_ind, sampler_alpha

def update_model_from_subgraph(sub_grah, ori_graph, selected_alpha):
    ori_graph_paras = ori_graph.state_dict()
    sub_graph_paras = sub_grah.state_dict()

    # alpha = ori_graph._all_arch_parameters
    # selected_alpha = sample_from_alpha(alpha, reduce_rate)

    ind = 0
    pre_selected = [0, 1, 2]

    for k, v in ori_graph_paras.items():
        if k in sub_graph_paras and len(v.shape) == 4:
            selected = selected_alpha[ind]
            if 'downsample' in k:
                a = selected_alpha[ind - 3]  #####for resnet34 there should be 3
                middle_paras = ori_graph_paras[k][pre_selected, :, :, :]
                middle_paras[:, a, :, :] = sub_graph_paras[k]
                ori_graph_paras[k][pre_selected, :, :, :] = middle_paras
            elif 'alpha' in k:
                ori_graph_paras[k][selected, :, :, :] = sub_graph_paras[k]
            else:
                middle_paras = ori_graph_paras[k][selected, :, :, :]
                middle_paras[:, pre_selected, :, :] = sub_graph_paras[k]
                ori_graph_paras[k][selected, :, :, :] = middle_paras
                pre_selected = selected
                ind += 1
        elif k in sub_graph_paras:
            if 'fc' in k and len(v.shape) == 1:
                ori_graph_paras[k] = sub_graph_paras[k]
            elif 'fc' in k and len(v.shape) == 2:
                ori_graph_paras[k][:, selected] = sub_graph_paras[k]
            else:
                ori_graph_paras[k][pre_selected] = sub_graph_paras[k]
        else:
            print('Not Matching Key: {}'.format(k))

    ori_graph.load_state_dict(ori_graph_paras)



    return ori_graph

def psnr_cul(sr,hr):
    diff = (sr - hr) / 255
    mse = diff.pow(2).mean()
    return -10 * math.log10(mse)

def calc_psnr(sr, hr, scale, rgb_range, dataset=None):
    if hr.nelement() == 1: return 0

    diff = (sr - hr) / rgb_range
    if dataset and dataset.dataset.benchmark:
        shave = scale
        if diff.size(1) > 1:
            gray_coeffs = [65.738, 129.057, 25.064]
            convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
            diff = diff.mul(convert).sum(dim=1)
    else:
        shave = scale + 6

    valid = diff[..., shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)
    
