from statistics import mean, stdev
import yaml 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

ANNOTATION_OFFSET = -3*1e-3

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

# plot star barrier vs regular barrier
# use grid lines

# scatter plot between regular and star barrier means, each point should be labeled with the width
plt.scatter(
    regular_barrier_means, 
    star_barrier_means,
    marker='o',
    label='Empirical barriers',
)

plt.xlim(-0.01, 0.36)
plt.ylim(-0.01, 0.36)

plt.grid(True)

for i, txt in enumerate(widths):
    plt.annotate(
        txt, 
        (regular_barrier_means[i]+ANNOTATION_OFFSET, star_barrier_means[i]), 
        fontsize=8
)
    
# fit a linear regression line and plot it

X = np.array(regular_barrier_means).reshape(-1, 1)
y = np.array(star_barrier_means)
reg = LinearRegression().fit(X, y)
X_plot = np.linspace(-0.01, 0.36, 100).reshape(-1, 1)
plt.plot(
    X_plot, 
    reg.predict(X_plot), 
    color='red',
    label='Fitted line'
)

# also annotate the line with the equation
plt.annotate(
    f"y = {reg.coef_[0]:.2f}x + {reg.intercept_:.2f}", 
    (0.1, 0.1), 
    fontsize=8,
)

plt.xlabel('regular-regular loss barrier')
plt.ylabel('star-regular loss barrier')

plt.legend()
plt.savefig('results/figures/mnist_mlp_star_vs_regular_barrier.svg')