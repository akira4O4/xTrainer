#bin/bash

#Launch Params
python_exec=python
all_machine=1 #total number of machine
all_gpu_in_one_machine=1 #one machine have all gpus
node_rank=0 #current gpu idx in all gpus
# Training_File=/home/seeking/llf/code/deep_learning_framework/exp/exp1/main.py
Training_File=/home/seeking/llf/code/deep_learning_framework/train.py
#单机多卡:使用多卡训练
${python_exec} -m torch.distributed.launch \
  --nnodes ${all_machine} \
  --nproc_per_node ${all_gpu_in_one_machine} \
  --use_env \
  ${Training_File} 
    

