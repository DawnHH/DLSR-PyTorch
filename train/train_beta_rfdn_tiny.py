import os
import sys
import torch
import torch.nn as nn
import numpy as np
import argparse
from torch import optim
from torch.autograd import Variable
import logging
import torch.backends.cudnn as cudnn
from models.rfdn_6_train_beta21_tiny import RFDN
from losses import loss_func
import utils
import data
from models import genotypes_rfdn
#import ipdb
#import vgg19
#from prune_search_space import prune_model,gain_final_model,weights_mapback

parser = argparse.ArgumentParser("Searched")
parser.add_argument('--name', type=str, default='search_upsampling', help='experiment name')
parser.add_argument('--save', type=str, default='/apdcephfs/private_hanhhhuang/test/code/NAS_SR/checkpoint', help='save_dir')
parser.add_argument('--seed', type=int, default=0, help='random initial seed')
parser.add_argument('--epochs', type=int, default=700, help='epoch number')
parser.add_argument('--loss', type=str, default='l1', help='Loss type: l1, l2')
parser.add_argument('--lr', type=float, default=0.0003, help='learning_rate')
parser.add_argument('--lr_decay_step', type=str, default='200-400-600-800', help='Learning rate decay step')
parser.add_argument('--lr_decay_rate', type=float, default=0.5, help='Learning rate decay rate')
parser.add_argument('--momentum', type=float, default=0.9, help='')
parser.add_argument('--wd', type=float, default=3e-4, help='weight decay')
parser.add_argument('--batch_size', type=int, default=16,help='input batch size for training')
parser.add_argument('--feature_criterion', type=str, default='l1', help='feature loss criterion')
parser.add_argument('--genotypes', type=str, default='', help='genotype of a block')
parser.add_argument('--report_frequency', type=int, default=200, help='report_frequency')
parser.add_argument('--val_interval', type=int, default=10, help='val interval epoch')
parser.add_argument('--grad_clip', type=float, default=50, help='gradient clipping threshold (0 = no clipping)')
#hardware
parser.add_argument('--n_threads', type=int, default=6,help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,help='number of GPUs')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
#data configuration
parser.add_argument('--dir_data', type=str, default='',help='dataset directory')
parser.add_argument('--data_train', type=str, default='DIV2K',help='train dataset name')
parser.add_argument('--data_test', type=str, default='DIV2K',help='test dataset name')
parser.add_argument('--data_range', type=str, default='1-800/801-810/801-810',help='train/test data range')
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

    # file = open(os.path.join(args.save, args.name, 'configs'), 'w')
    # file.write(message)
    # file.close()


def save_checkpoint(state, filename):
    torch.save(state, filename)

def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    cudnn.benchmark = True
    cudnn.enabled = True

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    #logging.info("args = %s", args)

    geno = eval("genotypes_rfdn.%s" % args.genotypes)
    model = RFDN(genotypes=geno)
    model = model.cuda()
    paras = utils.count_parameters_in_MB(model)
    logging.info("parameters: %f MB" ,paras)
    #netF = vgg19.define_F(args.gpu_ids, use_bn=False).cuda()

    # criterion = nn.L1Loss()
    # criterion = criterion.cuda()
    optimizer_weight = optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.999), weight_decay=1e-8)
    # scheduler_warmup = optim.lr_scheduler.CosineAnnealingLR(
    #             optimizer_weight, float(args.epoch_warmup))
    # scheduler_w = optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer_weight, float(args.epochs))
    milestone = list(map(lambda x: int(x), args.lr_decay_step.split('-')))
    scheduler_w = torch.optim.lr_scheduler.MultiStepLR(optimizer_weight, milestone, args.lr_decay_rate)


    loader = data.Data(args)
    train_dataloader = loader.loader_train
    test_dataloader = loader.loader_test


    if args.restore_from is not None:
        ckpt = torch.load(args.restore_from)
        model_dict = model.state_dict()
        #ipdb.set_trace()
        pretrained_dict = {k: v for k, v in ckpt['state_dict'].items() if (k in model_dict.keys() and k.split(".")[0] != 'upsampler')}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        start_epoch = 0
        #model.load_state_dict(ckpt['state_dict'])
        #iter_num = start_epoch * len(dataset)
        logging.info("Pretrained X2 model LOADED")
    else:
        start_epoch = 0
        #iter_num = 0
    best_psnr = 32.0
    for epoch in range(start_epoch,args.epochs):

        lr = scheduler_w.get_lr()[0]
        logging.info('\nEpoch %s lr %e',epoch+1,lr)
        model.train() 

        model,loss_train = train(model,train_dataloader,optimizer_weight,epoch)
        psnr_test = test(model,test_dataloader,args.scale)
        if psnr_test > best_psnr:
            logging.info('SAVING MODEL AT EPOCH %s' % (epoch + 1))
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer_weight.state_dict(),
            }, '%s/%s/model_best.pt' % (args.save, args.name))
            best_psnr = psnr_test
        logging.info('epoch:%03d  psnr_test:%f  psnr_best:%f', epoch, psnr_test, best_psnr)
        #logging.info('Epoch:%03d loss_train:%.4f', epoch, loss_train)
        scheduler_w.step()



def train(model,train_dataloader,optimizer_weight,epoch):
    losses = utils.AverageMeter()
    train_dataloader.dataset.set_scale(0)
    for step, (lr, hr, _,) in enumerate(train_dataloader):
        model.train()
        n = lr.size(0)
        #lr = Variable(lr, requires_grad=False).cuda()
        #hr = Variable(hr, requires_grad=False).cuda()
        lr, hr = prepare(lr,hr)
        #ipdb.set_trace()
        optimizer_weight.zero_grad()
        sr = model(lr)
        #sr = utils.quantize(sr, args.rgb_range)
        total_loss, ls, lg = loss_func(sr, hr, args.loss)
        total_loss.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer_weight.step()

        losses.update(total_loss.item(), n)
        if step % args.report_frequency == 0:
            logging.info('[Epoch : {}] [{}/{}] Loss => {:.4f} , L1 => {:.4f} , HFEN => {:.4f} '.format( \
                epoch + 1, step + 1, len(train_dataloader), total_loss.item(), ls.item(), lg.item()))
    return model, losses.avg

def test(model,test_dataloader,scale):
    #losses = utils.AverageMeter()
    model.eval()
    with torch.no_grad():
        for idx_data, d in enumerate(test_dataloader):
            psnr_a = utils.AverageMeter()
            for idx_scale, scale in enumerate(scale):
                d.dataset.set_scale(idx_scale)
                for lr, hr, filename in d:
                    # lr = Variable(lr).cuda()
                    # hr = Variable(hr).cuda()
                    lr, hr = prepare(lr, hr)
                    #ipdb.set_trace()
                    sr = model(lr)
                    sr = utils.quantize(sr, args.rgb_range)
                    #loss = criterion(sr, hr)

                    n = lr.size(0)
                    psnr = utils.calc_psnr(sr,hr,scale,args.rgb_range,dataset=d)
                    #losses.update(loss.item(), n)
                    psnr_a.update(psnr, n)

        # if step % args.report_freq == 0:
        #   logging.info('test %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return psnr_a.avg

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
    args.data_train = args.data_train.split('+')
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
