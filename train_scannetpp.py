import os
from datetime import datetime
import random
current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
port = random.randint(10000, 30000)

data_root = '/home/zhenhua2023/Proj/3Dv_Reconstruction/GS-Reconstruction/Data/ScanNetpp'
exp_name = f'../exps/reproduce/scannetpp-{current_time}'
scannetpp_scenes = ['8b5caf3398', '116456116b', '13c3e046d7', '0a184cf634', '578511c8a9', '21d970d8de']
gpu = 2

cmd_lis = []
for scene in scannetpp_scenes:
    
    source_args = " -s " + data_root + "/" + scene
    exp_args = " -m " + exp_name+"/"+scene
    
    # training
    train_args = source_args + exp_args + f" -r 2 --eval --port {port}"
    cmd_lis.append(f"CUDA_VISIBLE_DEVICES={gpu} python train.py" + train_args)

    # # rendering images
    cmd_lis.append(f"CUDA_VISIBLE_DEVICES={gpu} python render.py" + source_args + exp_args)

    # NVS metricd and visualization
    cmd_lis.append(f"CUDA_VISIBLE_DEVICES={gpu} python metrics.py" + exp_args + ' -f train')
    cmd_lis.append(f"CUDA_VISIBLE_DEVICES={gpu} python metrics.py" + exp_args + ' -f test')
    cmd_lis.append(f"CUDA_VISIBLE_DEVICES={gpu} python vis_outputs.py" + exp_args + ' -f train test')

# run cmd
for cmd in cmd_lis:
    print(cmd)
    os.system(cmd)