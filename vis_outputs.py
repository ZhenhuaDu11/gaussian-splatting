# concat all outputs
from argparse import ArgumentParser, Namespace
from tqdm import tqdm

import os
import cv2
import copy
import json

import matplotlib.pyplot as plt
import numpy as np

def convert_gray_to_cmap(img_gray, map_mode = 'jet', revert = True, vmax = None):
    '''Visualize point distances with 'hot_r' color map
    Args:
        cloud_source
        dists_to_target
    Return:
        cloud_visual: use color map, max
    '''
    img_gray = copy.deepcopy(img_gray)
    shape = img_gray.shape

    cmap = plt.get_cmap(map_mode)
    if vmax is not None:
        img_gray = img_gray / vmax
    else:
        img_gray = img_gray / (np.max(img_gray)+1e-6)
    if revert:
        img_gray = 1- img_gray      
    colors = cmap(img_gray.reshape(-1))[:, :3]

    # visualization
    colors = colors.reshape(shape+tuple([3]))*255
    return colors

# concatenate images
def write_image_lis(path_img_cat, lis_imgs, use_cmap = False, interval_img = 20, cat_mode = 'horizontal', color_space = 'BGR', wirte_imgs=True):
    '''Concatenate an image list to a single image and save it to the target path
    Args:
        cat_mode: horizonal/vertical
    '''
    img_cat = []
    for i in range(len(lis_imgs)):
        img = lis_imgs[i].squeeze()
        H, W = img.shape[:2]
        
        if use_cmap:
            img = convert_gray_to_cmap(img) if img.ndim==2 else img
        else:
            img = np.stack([img]*3, axis=-1) if img.ndim==2 else img
        
        if img.max() <= 1.0:
            img *= 255

        img_cat.append(img)
        if cat_mode == 'horizontal':
            img_cat.append(255 * np.ones((H, interval_img, 3)).astype('uint8'))
        elif cat_mode == 'vertical':
            img_cat.append(255 * np.ones((interval_img, W, 3)).astype('uint8'))
    
    if cat_mode == 'horizontal':
        img_cat = np.concatenate(img_cat[:-1], axis=1)
    elif cat_mode == 'vertical':
        img_cat = np.concatenate(img_cat[:-1], axis=0)
    else:
        raise NotImplementedError
    if color_space == 'RGB':
        img_cat = cv2.cvtColor(img_cat.astype(np.uint8), cv2.COLOR_BGR2RGB)
    
    if wirte_imgs:
        cv2.imwrite(path_img_cat, img_cat)
    else: 
        return img_cat

def write_image_info(image, image_info, interval_img=50, cat_mode= 'vertical'):
    H, W = image.shape[:2]
    if cat_mode == 'vertical':
        resize_image = np.concatenate([image, 255*np.ones((interval_img, W, 3))], axis=0)
        cv2.putText(resize_image, image_info,  (W//2-20, H+30), 
                    fontFace= cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale = 1, 
                    color = (0, 0, 255), 
                    thickness = 2)

    return resize_image

def concat_outputs(output_path, output_infos, output_metrics=None, resolution=1):
    '''
    concat output_maps to a lis_imgs
    inputs:
        - output_path: path of exps, e.g., '../exps/full/DTU/scan24/train/ours_30000'
        - output_infos: an array of output_maps, e.g., ['gt', 'renders', 'expect_depth']
        - output_metric: a dict of output_metrics, e.g., {'PSNR': [xxx, xxx], 'SSIM': [xxx, xxx]}
        - resize_shape: resize to shape
    outputs:
        - wirte lis_images
        - wirte lis_metrics
    '''
    print('Processing:', output_path)

    gt_images_list = os.listdir(f'{output_path}/{output_infos[0]}')
    H, W = cv2.imread(f'{output_path}/{output_infos[0]}/{gt_images_list[0]}').shape[:2]
    if resolution>1:
        H //= resolution
        W //= resolution

    output_lis_path = f'{output_path}/output_lis'
    os.makedirs(output_lis_path, exist_ok=True)

    for idx in tqdm(gt_images_list, desc='concating maps'):
         lis_imgs = []

         # write maps
         for info_idx in output_infos:
             output_map = cv2.imread(f'{output_path}/{info_idx}/{idx}')
             output_map_resize = cv2.resize(output_map, (W, H), interpolation=cv2.INTER_NEAREST)
             output_map_info = write_image_info(output_map_resize, info_idx)
             lis_imgs.append(output_map_info)
        
         lis_imgs_cat = write_image_lis(f'{output_lis_path}/{idx}', lis_imgs, wirte_imgs=False)
    
         # write metrics
         lis_imgs_metric = lis_imgs_cat.copy()
         for metric_name in output_metrics.keys():
             metric_idx = output_metrics[metric_name][idx]
             lis_imgs_metric = write_image_info(lis_imgs_metric,  f'{metric_name}:  {metric_idx}', interval_img=50)

         cv2.imwrite(f'{output_lis_path}/{idx}', lis_imgs_metric)

if __name__ == '__main__':
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--exp', type=str, default="ours_30000")
    parser.add_argument('--model_paths', '-m', nargs="+", type=str, default=['../exps/full/DTU/scan24'])
    parser.add_argument('--train_test_flag', '-f', nargs="+", type=str, default=['train'])
    args = parser.parse_args()
    
    exp = args.exp
    for model_path in args.model_paths:
        for train_test_split in args.train_test_flag:
        
            output_path = f'{model_path}/{train_test_split}/{exp}'
            output_infos = ['gt', 'renders', 'errors']
            output_metrics={}
            
            # nvs metric
            nvs_metric_path = f'{model_path}/per_view_{train_test_split}.json'
            with open(nvs_metric_path, 'r') as f:
                nvs_metric = json.load(f)[exp]
            for metric_name in nvs_metric.keys():
                output_metrics[metric_name] = nvs_metric[metric_name]

            concat_outputs(output_path, output_infos, output_metrics)
            


