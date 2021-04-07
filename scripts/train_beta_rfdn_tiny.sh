NAME=RFDN_beta21_tiny18_ps128_20210306_lr4
GPUS=0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/apdcephfs/private_hanhhhuang/Anaconda3/lib/
nohup /apdcephfs/private_hanhhhuang/Anaconda3/bin/python /apdcephfs/private_hanhhhuang/test/code/NAS_SR/train_beta_rfdn_tiny.py --name $NAME \
      --save /apdcephfs/private_hanhhhuang/test/code/NAS_SR/checkpoint/ \
      --epochs 1000 \
      --batch_size 32 \
      --gpu_ids $GPUS \
      --patch_size 128 \
      --dir_data /apdcephfs/private_hanhhhuang/data/ \
      --data_train DF2K \
      --data_test Set5 \
      --data_range 1-3450/3000-3450/3000-3450 \
      --rgb_range 255 \
      --ext sep \
      --lr 0.0004 \
      --test_every 2000 \
      --genotypes beta21 > /apdcephfs/private_hanhhhuang/test/code/NAS_SR/training_logs/rfdn_beta21_tiny18_ps128_20210306_lr4.log
