from statistics import mean, stdev
import yaml 
import matplotlib.pyplot as plt

widths = [4, 8, 16, 32, 64, 128, 256]

regular_barrier_means = []
regular_barrier_stdevs = []
star_barrier_means = []
star_barrier_stdevs = []

for width in widths:

    result_file = f"results/out/mlp_mnist_depth_1_width_{width}.yaml"
    with open(result_file, 'r') as f:
        results = yaml.safe_load(f)
    
    regular_barriers = [item['train_loss_barrier'] for item in results['anchor_held_out']]
    star_barriers = [item['train_loss_barrier'] for item in results['star_held_out']]

    regular_barrier_means.append(mean(regular_barriers))
    regular_barrier_stdevs.append(stdev(regular_barriers))
    star_barrier_means.append(mean(star_barriers))
    star_barrier_stdevs.append(stdev(star_barriers))

print(regular_barrier_means)
print(regular_barrier_stdevs)
print(star_barrier_means)
print(star_barrier_stdevs)

# plot these barriers wrt width on x axis


plt.errorbar(widths, regular_barrier_means, yerr=regular_barrier_stdevs, label='regular-regular', marker='o')
plt.errorbar(widths, star_barrier_means, yerr=star_barrier_stdevs, label='star-regular', marker='o')
plt.xlabel('Width')
plt.ylabel('Train Loss Barrier')    
plt.legend()  
plt.savefig('results/figures/mlp_barriers_vs_width.png')    