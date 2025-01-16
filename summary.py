import os
import glob
import json
import wandb
import pandas as pd
import argparse
from argparse import Namespace

mipnerf360_outdoor_scenes = ["bicycle", "flowers", "garden", "stump", "treehill"]
mipnerf360_indoor_scenes = ["room", "counter", "kitchen", "bonsai"]

def summary_results(result_dirs, scene_lis, json_name = 'results_train', extension_name = 'summary', wandb=None):
    if len(scene_lis) == 0:
        print("No scenes to summarize.")
        return None
    else:
        print(f"Summarize {json_name}: " + ", ".join(scene_lis))

    # Initialize lists to store the data
    data = []

    # Read each JSON file and extract the metrics
    for scene in scene_lis:
        # find the result file
        result_file = os.path.join(result_dirs, scene, f'{json_name}.json')
    
        with open(result_file, 'r') as f:
            result = json.load(f)
        
        # Extract the metrics
        result_lis = {'Scene' : scene}
        result_lis.update(result['ours_30000'])
        
        # Append the data to the list
        data.append(result_lis)

        # write into wandb
        if wandb is not None:
            for key, value in result['ours_30000'].items():
                wandb.log({f'{scene}/{json_name}/{key}': value}, step = 30000)

    # Calculate average metrics
    df = pd.DataFrame(data)

    average_lis = {'Scene' : 'average'}
    for key in result['ours_30000']:
        average_lis[key] = df[key].mean()
        
        if wandb is not None:
            wandb.log({f'{extension_name}/{json_name}/{key}': df[key].mean()}, commit=False)
    
    data.append(average_lis)

    # Create a DataFrame from the collected data
    df = pd.DataFrame(data)

    # Sort the DataFrame by experiment name
    df = df.sort_values('Scene')

    # Save the table to a CSV file
    df.to_csv(f'{result_dirs}/{json_name}_{extension_name}.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process results from JSON files.")
    parser.add_argument("--source_name", "-s", help="source path", default="mipnerf360")
    parser.add_argument("--model_path", "-m", help="model path", default="../exps/full/mipnerf360")
    parser.add_argument("--scene_lis", nargs="+", type=str, default=[], help="scene list")
    parser.add_argument('--use_wandb', action='store_true', default=True)
    args = parser.parse_args()

    if len(args.scene_lis) == 0:
        scene_lis = os.listdir(args.model_path)
        scene_lis = [scene for scene in scene_lis if not scene.endswith('.csv')]
    else:
        scene_lis = args.scene_lis

    with open(os.path.join(args.model_path, scene_lis[0], 'cfg_args')) as cfg_file:
        cfgfile_string = cfg_file.read()
    args_cfgfile = eval(cfgfile_string)

    if args.use_wandb:
        wandb.login()
        run = wandb.init(
            # Set the project where this run will be logged
            project=args.source_name,
            name=f'3dgs-{args.model_path.split("/")[-2]}-{args.model_path.split("/")[-1]}',
            # Track hyperparameters and run metadata
            settings=wandb.Settings(start_method="fork"),
            config=vars(args_cfgfile)
        )
    else:
        wandb = None

    json_name_lis = ['results_train', 'results_test']

    for json_name in json_name_lis:
        ### Summary results of all scenes
        summary_results(args.model_path, scene_lis, json_name = json_name, wandb=wandb)

        ### Summart resuults of inddoor and outdoor scenes of Mip360 dataset
        if args.source_name == 'mipnerf360':
            mip360_outdoor_lis = [scene for scene in scene_lis if scene in mipnerf360_outdoor_scenes]
            mip360_indoor_lis = [scene for scene in scene_lis if scene in mipnerf360_indoor_scenes]

            summary_results(args.model_path, mip360_indoor_lis, json_name = json_name, extension_name = 'indoor', wandb=wandb)
            summary_results(args.model_path, mip360_outdoor_lis, json_name = json_name, extension_name = 'outdoor', wandb=wandb)