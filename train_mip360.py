import os
from datetime import datetime
import random
current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
port = random.randint(10000, 30000)

data_root = '../../Data/Mip-NeRF360'
exp_name = f'../exps/reproduce/Mip-NeRF360-{current_time}'
mipnerf360_outdoor_scenes = ["bicycle", "flowers", "garden", "stump", "treehill"]
mipnerf360_indoor_scenes = ["room", "counter", "kitchen", "bonsai"]
gpu = 3

cmd_lis = []
for scene in (mipnerf360_outdoor_scenes+mipnerf360_indoor_scenes):
    
    source_args = " -s " + data_root + "/" + scene
    exp_args = " -m " + exp_name+"/"+scene
    
    # training
    train_args = source_args + exp_args + f" --eval --port {port}"
    # if scene in mipnerf360_outdoor_scenes:
    #     train_args += " -i images_4"
    # elif scene in mipnerf360_indoor_scenes:
    #     train_args += " -i images_2"
    cmd_lis.append(f"CUDA_VISIBLE_DEVICES={gpu} python train.py" + train_args)

    # rendering images
    cmd_lis.append(f"CUDA_VISIBLE_DEVICES={gpu} python render.py" + source_args + exp_args)

    # NVS metricd and visualization
    cmd_lis.append(f"CUDA_VISIBLE_DEVICES={gpu} python metrics.py" + exp_args + ' -f train')
    cmd_lis.append(f"CUDA_VISIBLE_DEVICES={gpu} python metrics.py" + exp_args + ' -f test')
    cmd_lis.append(f"CUDA_VISIBLE_DEVICES={gpu} python vis_outputs.py" + exp_args + ' -f train test')

# run cmd
for cmd in cmd_lis:
    print(cmd)
    os.system(cmd)