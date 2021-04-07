NAME=test_tiny18
GPUS=0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/apdcephfs/private_hanhhhuang/Anaconda3/lib/
/apdcephfs/private_hanhhhuang/Anaconda3/bin/python /apdcephfs/private_hanhhhuang/test/code/NAS_SR/test_beta_tiny.py --name $NAME \
      --patch_size 256 \
      --data_test Set5+Set14+B100+Urban100 \
      --data_range 801-900 \
      --dir_data /apdcephfs/private_hanhhhuang/data/ \
      --scale 2 \
      --ext sep \
      --genotypes beta21 \
      --restore_from  /apdcephfs/private_hanhhhuang/test/code/NAS_SR/checkpoint/best/RFDN_beta21_tiny18_ps128_20210306_lr4_700.pt \
      --test_only 
