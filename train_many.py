import yaml
import random
import subprocess
import tempfile
import shutil
import argparse

from contextlib import suppress 
import os
import shutil

MAIN_CONFIG_FILE = 'configs/cifar10_wrn_16_1.yaml'
SEEDS = [i for i in range(5)]
WANDB_LINKS_FILE = 'cifar10_wrn_16_1_anchors.txt'

def change_seed(config_file_path, new_seed):
    # Load YAML config file
    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)

    # Change seed in the config
    config['seed'] = new_seed

    # Write modified config to a temporary file
    temp_config_file = tempfile.mktemp(suffix='.yaml')
    with open(temp_config_file, 'w') as file:
        yaml.dump(config, file)

    return temp_config_file
    
def main():

    # Path to the original config file
    config_file_path = MAIN_CONFIG_FILE
    wandb_links = []
    
    for new_seed in SEEDS:

        print(f"Running with seed = {new_seed}...")
        # Change seed in config and get path to temporary config file
        temp_config_file = change_seed(config_file_path, new_seed)

        try:
            # Invoke training script with the modified config file
            output = subprocess.run(
                ['python', 'main.py', temp_config_file], 
                check=True,
                capture_output=True,
                text=True,
            ).stdout

            print(output)
            # Find and extract the wandb link
            wandb_link = None
            for line in output.split('\n'):
                if 'wandb url' in line and 'https' in line:
                    wandb_link = line.strip()
                    break
            
            if wandb_link:
                with open(WANDB_LINKS_FILE, 'w+') as f:
                    f.write(wandb_link + '\n')
            
        except subprocess.CalledProcessError as e:
            print("Error running another script:", e)
        finally:
            # Clean up temporary config file
            os.remove(temp_config_file)

   
    
if __name__ == "__main__":
    main()
