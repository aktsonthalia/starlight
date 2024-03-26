import yaml
import random
import subprocess
import tempfile
import shutil
import argparse
from contextlib import suppress 
import os
import shutil

def load_with_diff(config_file_path, changes):

    # Load original YAML config file
    with open(config_file_path, 'r') as file:
        original = yaml.safe_load(file)

    # Apply changes
    for key, value in changes.items():
        if isinstance(value, dict) and key in original and isinstance(original[key], dict):
            apply_changes(original[key], value)
        else:
            original[key] = value

    # Write modified config to a temporary file
    temp_config_file = tempfile.mktemp(suffix='.yaml')
    with open(temp_config_file, 'w') as file:
        yaml.dump(original, file)

    return temp_config_file
    
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, help='Path to the main config file')
    parser.add_argument('--wandb_links_file', '-w', type=str, help='Path to the file to store wandb links')
    parser.add_argument('--num_held_out', '-o', type=int, help='Number of held out models')
    parser.add_argument('--num_anchors', '-a', type=int, help='Number of anchor models')
    args = parser.parse_args()
    MAIN_CONFIG_FILE = args.config
    WANDB_LINKS_FILE = args.wandb_links_file
    NUM_HELD_OUT = int(args.num_held_out)
    NUM_ANCHORS = int(args.num_anchors)
    HELD_OUT_SEEDS = [i for i in range(NUM_HELD_OUT)]
    ANCHOR_SEEDS = [i for i in range(NUM_HELD_OUT, NUM_HELD_OUT + NUM_ANCHORS)]

    wandb_links_dict = {
        "held_out": [],
        "anchors": [],
    }

    # Path to the original config file
    config_file_path = MAIN_CONFIG_FILE
    wandb_links = []
    
    for model_type, seeds in zip(['held_out', 'anchors'], [HELD_OUT_SEEDS, ANCHOR_SEEDS]):

        for new_seed in seeds:

            print(f"Running with seed = {new_seed}...")

            changes = {'seed': new_seed}
            if model_type == 'anchors':
                changes['eval'] = {
                    'held_out_anchors': [x.split('/')[-1] for x in wandb_links_dict['held_out']]
                }

            temp_config_file = load_with_diff(
                config_file_path, 
                changes,
            )

            try:
                # Invoke training script with the modified config file
                output = subprocess.run(
                    ['python', 'main.py', temp_config_file], 
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout

                # Find and extract the wandb link
                wandb_link = None
                for line in output.split('\n'):
                    if 'wandb url' in line and 'https' in line:
                        wandb_link = line.strip().split(" ")[-1]
                        break
                
                if wandb_link:
                    wandb_links_dict[model_type].append(wandb_link)
                    yaml.dump(wandb_links_dict, open(WANDB_LINKS_FILE, 'w'))

            except subprocess.CalledProcessError as e:
                print("Error running another script:", e)
            finally:
                # Clean up temporary config file
                os.remove(temp_config_file)
    

if __name__ == "__main__":
    main()
