import os
import glob
import json
import pandas as pd
import argparse

def summary_results(result_dirs, scene_lis, json_name = 'results_train'):
    # Initialize lists to store the data
    data = []

    # Read each JSON file and extract the metrics
    for scene in scene_lis:
        # find the result file
        if ".csv" not in scene:
            result_file = os.path.join(result_dirs, scene, f'{json_name}.json')
        else: 
            continue
    
        with open(result_file, 'r') as f:
            result = json.load(f)
        
        # Extract the metrics
        result_lis = {'Scene' : scene}
        result_lis.update(result['ours_30000'])
        
        # Append the data to the list
        data.append(result_lis)

    # Calculate average metrics
    df = pd.DataFrame(data)

    average_lis = {'Scene' : 'average'}
    for key in result['ours_30000']:
        average_lis[key] = df[key].mean()
    
    data.append(average_lis)

    # Create a DataFrame from the collected data
    df = pd.DataFrame(data)

    # Sort the DataFrame by experiment name
    df = df.sort_values('Scene')

    # Save the table to a CSV file
    df.to_csv(f'{result_dirs}/{json_name}_summary.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process results from JSON files.")
    parser.add_argument("--model_path", "-m", help="model path", default="../exps/full/scannetpp-lambdadist-0")
    parser.add_argument("--scene_lis", nargs="+", type=str, default=[], help="scene list")
    args = parser.parse_args()
    
    if len(args.scene_lis) == 0:
        scene_lis = os.listdir(args.model_path)
    else:
        scene_lis = args.scene_lis

    for json_name in ['results_train', 'results_test']:
        summary_results(args.model_path, scene_lis, json_name = json_name)