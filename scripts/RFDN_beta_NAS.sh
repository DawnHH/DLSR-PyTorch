NAME=RFDN_beta1_para_loss_20210310
GPUS=0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/apdcephfs/private_hanhhhuang/Anaconda3/lib/
nohup /apdcephfs/private_hanhhhuang/Anaconda3/bin/python /apdcephfs/private_hanhhhuang/test/code/NAS_SR/RFDN6_para_loss.py --name $NAME \
      --save /apdcephfs/private_hanhhhuang/test/code/NAS_SR/checkpoint/ \
      --batch_size 64 \
      --gpu_ids $GPUS \
      --patch_size 64 \
      --seed 9 \
      --dir_data /apdcephfs/private_hanhhhuang/data/ \
      --data_train DF2K \
      --data_test Set5 \
      --data_range 1-3000/3000-3450/3000-3450 \
      --rgb_range 255 \
      --ext sep > /apdcephfs/private_hanhhhuang/test/code/NAS_SR/training_logs/beta/NAS_RFDN_beta1_para_loss_20210310.log
