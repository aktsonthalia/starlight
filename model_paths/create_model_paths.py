# takes the name of the folder and creates lists of paths

import os
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--folder', '-f', type=str)
args = parser.parse_args()
folder = args.folder

def create_model_paths(folder, output_file):
    # Get the list of files in the folder
    files = os.listdir(folder)
    # Create a list of paths
    paths = [os.path.join(folder, file) for file in files]

    # Write the list of paths to a file
    with open(output_file, 'w') as file:
        for path in paths:
            file.write(path + '\n')

# remove / from the folder name's end
if folder[-1] == '/':
    folder = folder[:-1]

anchors_folder = os.path.join(folder, "anchors")
output_file = folder.split('/')[-1] + "_anchors.txt"
create_model_paths(anchors_folder, output_file)

held_out_folder = os.path.join(folder, "held_out")
output_file = folder.split('/')[-1] + "_held_out.txt"
create_model_paths(held_out_folder, output_file)