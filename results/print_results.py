import argparse
import yaml
from statistics import mean, stdev

model_pair_types = [
    'star_held_out',
    'anchor_held_out',
    'star_anchor',
]

parser = argparse.ArgumentParser(description='Print results')
parser.add_argument('--result_file', '-r', type=str, help='Path to the result file')
args = parser.parse_args()

with open(args.result_file, 'r') as f:
    results = yaml.safe_load(f)

star_held_out = results['star_held_out']    
anchor_held_out = results['anchor_held_out']
star_anchor = results['star_anchor']

for model_pair_type in model_pair_types:
    print(f'Model pair type: {model_pair_type}')
    tmp_result = results[model_pair_type]
    train_loss_barriers = [item['train_loss_barrier'] for item in tmp_result]
    print(f'Train loss barrier: ${mean(train_loss_barriers):.3f} \pm {stdev(train_loss_barriers):.3f}$')

star_train_losses = [item['train_losses'][0] for item in star_held_out]
regular_train_losses = [item['train_losses'][0] for item in anchor_held_out]

print(f'Star train loss: ${mean(star_train_losses):.3f} \pm {stdev(star_train_losses):.3f}$')
print(f'Regular train loss: ${mean(regular_train_losses):.3f} \pm {stdev(regular_train_losses):.3f}$')
# find mean of star_held_out
