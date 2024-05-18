import argparse
import yaml
from statistics import mean, stdev
import matplotlib.pyplot as plt

import numpy as np

# plt.tight_layout()
plt.rcParams.update({'font.size': 16})
plt.rcParams['svg.fonttype'] = 'none'


SPECIAL_STAR_MODELS = {
    'resnet18_cifar10_sgd': 'by2vpp9d',
    'resnet18_cifar100_sgd': 'rylbd95p',
}

model_pair_types = [
    'star_held_out',
    'anchor_held_out',
    'star_anchor',
]

COLORS = {  
    'star': 'red',
    'regular': 'blue',
}

parser = argparse.ArgumentParser(description='Print results')
parser.add_argument('--result_file', '-r', type=str, help='Path to the result file')
args = parser.parse_args()

config_name = args.result_file.split('/')[-1].split('.')[0]
with open(args.result_file, 'r') as f:
    results = yaml.safe_load(f)

star_held_out = results['star_held_out']    
anchor_held_out = results['anchor_held_out']
star_anchor = results['star_anchor']

if config_name in SPECIAL_STAR_MODELS.keys():
    star_held_out = [item for item in star_held_out if item['model1_id'] == SPECIAL_STAR_MODELS[config_name]]
    star_anchor = [item for item in star_anchor if item['model1_id'] == SPECIAL_STAR_MODELS[config_name]]

# print number of each pair

print(f'Number of star_held_out: {len(star_held_out)}')
print(f'Number of anchor_held_out: {len(anchor_held_out)}')
print(f'Number of star_anchor: {len(star_anchor)}')

for model_pair_type in model_pair_types:
    print(f'Model pair type: {model_pair_type}')
    tmp_result = results[model_pair_type]
    train_loss_barriers = [item['train_loss_barrier'] for item in tmp_result]
    print(f'Train loss barrier: ${mean(train_loss_barriers):.3f} \pm {stdev(train_loss_barriers):.3f}$')

star_train_losses = [item['train_losses'][0] for item in star_held_out]
regular_train_losses = [item['train_losses'][0] for item in anchor_held_out]

print(f'Star train loss: ${mean(star_train_losses):.3f} \pm {stdev(star_train_losses):.3f}$')
print(f'Regular train loss: ${mean(regular_train_losses):.3f} \pm {stdev(regular_train_losses):.3f}$')


def make_plot(
    data1, 
    data2, 
    label1,
    label2,
    ylabel,
    title,
):
    num_points = len(data1[0])
    t = np.linspace(0, 1, num_points)

    data1_means = [mean([item[i] for item in data1]) for i in range(num_points)]
    data1_stds = [stdev([item[i] for item in data1]) for i in range(num_points)]

    data2_means = [mean([item[i] for item in data2]) for i in range(num_points)]
    data2_stds = [stdev([item[i] for item in data2]) for i in range(num_points)]

    plt.figure(figsize=(10, 10))

    plt.fill_between(
        t,
        [mean - std for mean, std in zip(data1_means, data1_stds)],
        [mean + std for mean, std in zip(data1_means, data1_stds)],
        alpha=0.5,
        color=COLORS['star']
    )
    plt.plot(t, data1_means, color=COLORS['star'], label=label1)

    plt.fill_between(
        t,
        [mean - std for mean, std in zip(data2_means, data2_stds)],
        [mean + std for mean, std in zip(data2_means, data2_stds)],
        alpha=0.5,
        color=COLORS['regular']
    )
    plt.plot(t, data2_means, color=COLORS['regular'], label=label2)

    plt.xlabel('t')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()

    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    for n, label in enumerate(ax.yaxis.get_ticklabels()):
        if n % 2 != 0:
            label.set_visible(False)
            
    plt.savefig(f'figures/{title}.svg', format='svg', bbox_inches='tight', dpi=300)
    plt.clf()



star_anchor_train_loss_lists = [item['train_losses'] for item in star_anchor]
anchor_held_out_train_loss_lists = [item['train_losses'] for item in anchor_held_out]
star_held_out_train_loss_lists = [item['train_losses'] for item in star_held_out]

make_plot(
    star_anchor_train_loss_lists,
    anchor_held_out_train_loss_lists,
    'Star-regular',
    'regular-regular',
    'Train loss',
    f'{config_name}_star_anchor_regular_train_loss'
)

make_plot(
    star_held_out_train_loss_lists,
    anchor_held_out_train_loss_lists,
    'Star-regular',
    'regular-regular',
    'Train loss',
    f'{config_name}_star_held_out_regular_train_loss'
)

star_held_out_train_acc_lists = [item['train_accuracies'] for item in star_held_out]
anchor_held_out_train_acc_lists = [item['train_accuracies'] for item in anchor_held_out]
star_anchor_train_acc_lists = [item['train_accuracies'] for item in star_anchor]

make_plot(
    star_anchor_train_acc_lists,
    anchor_held_out_train_acc_lists,
    'Star-regular',
    'regular-regular',
    'Train accuracy',
    f'{config_name}_star_anchor_regular_train_acc'
)

make_plot(
    star_held_out_train_acc_lists,
    anchor_held_out_train_acc_lists,
    'Star-regular',
    'regular-regular',
    'Train accuracy',
    f'{config_name}_star_held_out_regular_train_acc'
)
