import argparse
import yaml
from statistics import mean, stdev
import matplotlib.pyplot as plt

import numpy as np

# plt.tight_layout()
plt.rcParams['font.size'] = 40
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
    'star': '#ff7f0e',
    'regular': '#1f77b4',
}

plot_barriers_vs_num_anchors = False

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
    plot_barriers_vs_num_anchors = True 
# print number of each pair

print(f'Number of star_held_out: {len(star_held_out)}')
print(f'Number of anchor_held_out: {len(anchor_held_out)}')
print(f'Number of star_anchor: {len(star_anchor)}')

for model_pair_type, tmp_result in zip(
    model_pair_types,
    [star_held_out, anchor_held_out, star_anchor]
):
    print(f'Model pair type: {model_pair_type}')
    train_loss_barriers = [item['train_loss_barrier'] for item in tmp_result]

    if len(train_loss_barriers) == 0:
        print(f'No train loss barriers found for {model_pair_type}')
        continue
    elif len(train_loss_barriers) == 1:
        print(f'Train loss barrier: ${train_loss_barriers[0]:.3f}$')
    else:
        print(f'Train loss barrier: ${mean(train_loss_barriers):.3f} \pm {stdev(train_loss_barriers):.3f}$')

star_train_losses = [item['train_losses'][0] for item in star_held_out]
regular_train_losses = [item['train_losses'][0] for item in anchor_held_out]

if len(star_train_losses) == 1:
    print(f'Star train loss: ${star_train_losses[0]:.3f}$')
else:
    print(f'Star train loss: ${mean(star_train_losses):.3f} \pm {stdev(star_train_losses):.3f}$')

if len(regular_train_losses) == 0:
    print(f'No regular train losses found')
elif len(regular_train_losses) == 1:
    print(f'Regular train loss: ${regular_train_losses[0]:.3f}$')
else:
    print(f'Regular train loss: ${mean(regular_train_losses):.3f} \pm {stdev(regular_train_losses):.3f}$')


def make_plot(
    data1, 
    data2, 
    label1,
    label2,
    ylabel,
    title,
    include_legend=True,
):
    num_points = len(data1[0])
    t = np.linspace(0, 1, num_points)

    data1_means = [mean([item[i] for item in data1]) for i in range(num_points)]

    if len(data1) == 1:
        data1_stds = [0 for i in range(num_points)]
    else:
        data1_stds = [stdev([item[i] for item in data1]) for i in range(num_points)]

    data2_means = [mean([item[i] for item in data2]) for i in range(num_points)]

    if len(data2) == 1: 
        data2_stds = [0 for i in range(num_points)]
    else:
        data2_stds = [stdev([item[i] for item in data2]) for i in range(num_points)]

    plt.figure(figsize=(10, 5))

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
    # legend in right middle
    if include_legend:
        plt.legend(loc='center right')
    plt.grid()

    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    for n, label in enumerate(ax.yaxis.get_ticklabels()):
        if n % 2 != 0:
            label.set_visible(False)
            
    plt.savefig(f'figures/{title}.svg', format='svg', bbox_inches='tight')
    plt.clf()


if len(anchor_held_out) == 0:   
    print('No anchor held out models found')
    exit()

star_anchor_train_loss_lists = [item['train_losses'] for item in star_anchor]
anchor_held_out_train_loss_lists = [item['train_losses'] for item in anchor_held_out]
star_held_out_train_loss_lists = [item['train_losses'] for item in star_held_out]

make_plot(
    star_anchor_train_loss_lists,
    anchor_held_out_train_loss_lists,
    'Star-regular',
    'regular-regular',
    'Train loss',
    f'{config_name}_star_anchor_regular_train_loss',
    include_legend=True
)

make_plot(
    star_held_out_train_loss_lists,
    anchor_held_out_train_loss_lists,
    'Star-regular',
    'regular-regular',
    'Train loss',
    f'{config_name}_star_held_out_regular_train_loss',
    include_legend=False
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
    f'{config_name}_star_anchor_regular_train_acc',
    include_legend=False
)

make_plot(
    star_held_out_train_acc_lists,
    anchor_held_out_train_acc_lists,
    'Star-regular',
    'regular-regular',
    'Train accuracy',
    f'{config_name}_star_held_out_regular_train_acc',
    include_legend=False
)





if plot_barriers_vs_num_anchors:

    import wandb
    from dotmap import DotMap
    api = wandb.Api()
    from tqdm import tqdm

    plt.rcParams['font.size'] = 24

    if 'cifar100' in config_name:
        plt.figure(figsize=(10, 3.85))
    else:
        plt.figure(figsize=(10, 4))

    def get_num_anchors(model_id):

        wandb_run = api.run(f'mode-connect/star-domain/{model_id}')
        config = DotMap(wandb_run.config)
        config.model.anchor_model_wandb_ids
        return len(config.model.anchor_model_wandb_ids)
    
    with open(args.result_file, 'r') as f:
        star_held_out = yaml.safe_load(f)['star_held_out']
    star_model_ids = set([item['model1_id'] for item in star_held_out])
    star_model_ids = list(star_model_ids)
    num_anchors_list = [get_num_anchors(model_id) for model_id in star_model_ids]

    star_train_loss_barriers = [
        [item['train_loss_barrier'] 
        for 
        item in star_held_out 
        if 
        item['model1_id'] == model_id] 
        for model_id in star_model_ids
    ]


    star_train_loss_barriers_means = [mean(item) for item in star_train_loss_barriers]
    star_train_loss_barriers_stds = [stdev(item) for item in star_train_loss_barriers]

    plt.scatter(
        num_anchors_list,
        star_train_loss_barriers_means,
        color='red',
        label='star-regular',
    )

    plt.errorbar(
        num_anchors_list,
        star_train_loss_barriers_means,
        yerr=star_train_loss_barriers_stds,
        fmt='o',
        color=COLORS['star'],
    )

    regular_train_loss_barriers = [item['train_loss_barrier'] for item in anchor_held_out]
    plt.axhline(
        y=mean(regular_train_loss_barriers),
        color=COLORS['regular'],
        label='regular-regular',
    )
    plt.fill_between(
        [0, max(num_anchors_list)],
        [mean(regular_train_loss_barriers) - stdev(regular_train_loss_barriers)],
        [mean(regular_train_loss_barriers) + stdev(regular_train_loss_barriers)],
        color=COLORS['regular'],
        alpha=0.5,
    )
    # plt.axhline(
    #     y=mean(regular_train_loss_barriers) + stdev(regular_train_loss_barriers),
    #     color=COLORS['regular'],
    #     linestyle='--',
    # )
    # plt.axhline(
    #     y=mean(regular_train_loss_barriers) - stdev(regular_train_loss_barriers),
    #     color=COLORS['regular'],
    #     linestyle='--',
    # )
    
    plt.xlabel('Number of source models')
    plt.ylabel('Train loss barrier')
    plt.grid()
    plt.legend()
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(f'figures/{config_name}_train_loss_barrier_vs_num_anchors.svg', format='svg', bbox_inches='tight')
    plt.clf()