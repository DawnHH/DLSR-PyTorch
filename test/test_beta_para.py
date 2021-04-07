from skimage.metrics import structural_similarity as compare_ssim
import os
import sys
import torch
import cv2
import torch.nn as nn
import numpy as np
import argparse
from torch import optim
from torch.autograd import Variable
import logging
import torch.backends.cudnn as cudnn
from models.beta1_para_loss import RFDN
from losses import loss_func
import utils
import data
import thop
from models import genotypes_rfdn
from thop import profile
from thop import clever_format
import ipdb

parser = argparse.ArgumentParser("Searched")
parser.add_argument('--name', type=str, default='search_upsampling', help='experiment name')
parser.add_argument('--save', type=str, default='/apdcephfs/private_hanhhhuang/test/code/NAS_SR/checkpoint', help='save_dir')
#parser.add_argument('--seed', type=int, default=0, help='random initial seed')
parser.add_argument('--genotypes', type=str, default='', help='genotype of a block')

#hardware
parser.add_argument('--n_threads', type=int, default=6,help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,help='number of GPUs')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

#data configuration
parser.add_argument('--dir_data', type=str, default='',help='dataset directory')
parser.add_argument('--data_test', type=str, default='Set5+Set14+B100+Urban100+DIV2K',help='test dataset name')
parser.add_argument('--data_range', type=str, default='801-810',help='train/test data range')
parser.add_argument('--ext', type=str, default='sep',help='dataset file extension')
parser.add_argument('--scale', type=str, default='2',help='super resolution scale')
parser.add_argument('--patch_size', type=int, default=96,help='output patch size')
parser.add_argument('--rgb_range', type=int, default=255,help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,help='number of color channels to use')
parser.add_argument('--test_every', type=int, default=1000,help='do test per every N batches')
parser.add_argument('--test_only', action='store_true',help='set this option to test the model')
parser.add_argument('--no_augment', action='store_true',help='do not use data augmentation')
#model
parser.add_argument('--model', default='',help='model name')
parser.add_argument('--restore_from', default=None,help='pretrained model name')


def print_options(args):
    message = '------------------------ Options ------------------------\n'
    for k, v in sorted(vars(args).items()):
        comment = ''
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '------------------------- End -------------------------'
    return message


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    cudnn.benchmark = True
    cudnn.enabled = True

    geno = eval("genotypes_rfdn.%s" % args.genotypes)
    model = RFDN(genotypes=geno)
    model = model.cuda()
    paras = utils.count_parameters_in_MB(model)
    logging.info("parameters: %f MB", paras)
    #inp = torch.randn(1,3,96,96).cuda()
    #macs,params = profile(model,inputs = (inp,))
    #macs, params = clever_format([macs, params], "%.3f")
    #print(macs,params)
    loader = data.Data(args)
    test_dataloader = loader.loader_test

    ckpt = torch.load(args.restore_from)
    model.load_state_dict(ckpt['state_dict'])
    #iter_num = start_epoch * len(dataset)
    logging.info("Pretrained model LOADED")
    save_path = os.path.join(args.save, args.name)
    test(model,test_dataloader,args.scale,save_path)
    inp = torch.randn(1,3,640,360).cuda()
    macs,params = profile(model,inputs = (inp,))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs,params)
def test(model,test_dataloader,scale,save_path):
    model.eval()
    with torch.no_grad():
        for idx_data, d in enumerate(test_dataloader):
            d.dataset.set_scale(0)
            psnr_a = utils.AverageMeter()
            #ssim_a = utils.AverageMeter()
            for lr, hr, filename in d:
                
                # lr = Variable(lr).cuda()
                # hr = Variable(hr).cuda()
                lr, hr = prepare(lr, hr)
                sr = model(lr)
                sr = utils.quantize(sr, args.rgb_range)


                n = lr.size(0)
                psnr = utils.calc_psnr(sr, hr, 2, args.rgb_range, dataset=d)
                #ipdb.set_trace()
                sr = sr.cpu().numpy().squeeze(0).transpose((1 ,2 ,0))
                #ipdb.set_trace()
                sr = cv2.cvtColor(sr,cv2.COLOR_BGR2RGB)
                cv2.imwrite('{}/{}/{}_sr.png'.format(save_path,args.data_test[idx_data],filename[0]), sr.astype(np.uint8))
                #hr = hr.cpu().numpy().squeeze(0).transpose((1, 2, 0))
                #hr = cv2.cvtColor(hr,cv2.COLOR_BGR2RGB)
                #ssim = compare_ssim(sr, hr, data_range=255, multichannel=True)
                #ipdb.set_trace()
                psnr_a.update(psnr, n)
                #ssim_a.update(ssim, n)

            logging.info('Dataset:%s psnr_test:%f', args.data_test[idx_data], psnr_a.avg)

def prepare(lr, hr):
    device = torch.device('cpu' if args.cpu else 'cuda')

    def _prepare(tensor):
        return tensor.to(device)

    res = []
    res.append(_prepare(lr))
    res.append(_prepare(hr))
    return res

if __name__ == '__main__':
    # setting logs
    args = parser.parse_args()
    args.scale = list(map(lambda x: int(x), args.scale.split('+')))
    args.data_test = args.data_test.split('+')

    if not os.path.exists(os.path.join(args.save, args.name)):
        os.makedirs(os.path.join(args.save, args.name))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, args.name,'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    message = print_options(args)
    logging.info(message)
    main()
