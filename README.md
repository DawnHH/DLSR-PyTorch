# DLSR-PyTorch

This repository is an implementation of the paper "Lightweight Image Super-Resolution with Hierarchical and Differentiable Neural Search".

The code is based on:

+ EDSR-PyTorch: https://github.com/sanghyun-son/EDSR-PyTorch 
+ RFDN: https://github.com/njulj/RFDN 
+ PAN:  https://github.com/zhaohengyuan1/PAN 

## Dependencies

* Python 3.7
* PyTorch 1.2
* numpy
* skimage
* imageio
* matplotlib
* logging

## Search

The search script can be found at  : `script/RFDN_beta_NAS.sh`

Usage :

```
NAME= **YOUR TASK NAME**
GPUS=0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH: **YOUR ABSOLUTE PATH TO**/Anaconda3/lib/
**YOUR ABSOLUTE PATH TO**/python **YOUR ABSOLUTE PATH TO**/NAS_SR/search/RFDN6_para_loss.py --name $NAME \
      --save **YOUR ABSOLUTE PATH TO**/NAS_SR/checkpoint/ \
      --batch_size 64 \
      --gpu_ids $GPUS \
      --patch_size 64 \
      --seed 9 \
      --dir_data **YOUR ABSOLUTE PATH TO**/data/ \  
      --data_train DF2K \
      --data_test Set5 \
      --data_range 1-3000/3000-3450/3000-3450 \
      --rgb_range 255 \
      --ext sep > **YOUR ABSOLUTE PATH TO**/NAS_SR/training_logs/**file name**.log

```

## retrain

Copy the genotype to `models/genotypes_rfdn.py` in the format listed in the file. Or you may directly use the genotypes in the file. 

For network-level structure, you may slightly change the model in `models/beta1_para_loss.py`

The retrain script can be found at : `script/train_beta_para.sh`. The usage is basically the same as search, note to input the argument of **genotypes** 

The model with the best performance is saved in `checkpoint/**YOUR TASK NAME**` during training.

## test

The test script can be found at  : `script/test_beta_para.sh`. The benchmark datasets can be downloaded in [EDSR-PyTorch](https://github.com/sanghyun-son/EDSR-PyTorch ). The retrained models can be found in `checkpoint/best`. All files with 'tiny' in their name means the tiny model we build to compare with [TPSR](https://arxiv.org/abs/2007.04356), ECCV2020.

We run the  search and retrain process in single V100 GPU.

