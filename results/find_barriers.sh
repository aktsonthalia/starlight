# python find_barriers.py --dataset mnist --model mlp --setting depth_1_width_4 --input_file wandb_links.yaml --output_dir out/ > out/find_barriers_mnist_mlp_depth_1_width_4.out &
# python find_barriers.py --dataset mnist --model mlp --setting depth_1_width_8 --input_file wandb_links.yaml --output_dir out/ > out/find_barriers_mnist_mlp_depth_1_width_8.out &
# python find_barriers.py --dataset mnist --model mlp --setting depth_1_width_16 --input_file wandb_links.yaml --output_dir out/ > out/find_barriers_mnist_mlp_depth_1_width_16.out &
# python find_barriers.py --dataset mnist --model mlp --setting depth_1_width_32 --input_file wandb_links.yaml --output_dir out/ > out/find_barriers_mnist_mlp_depth_1_width_32.out &
# python find_barriers.py --dataset mnist --model mlp --setting depth_1_width_64 --input_file wandb_links.yaml --output_dir out/ > out/find_barriers_mnist_mlp_depth_1_width_64.out &
# python find_barriers.py --dataset mnist --model mlp --setting depth_1_width_128 --input_file wandb_links.yaml --output_dir out/ > out/find_barriers_mnist_mlp_depth_1_width_128.out &
# python find_barriers.py --dataset mnist --model mlp --setting depth_1_width_256 --input_file wandb_links.yaml --output_dir out/ > out/find_barriers_mnist_mlp_depth_1_width_256.out &
# python find_barriers.py --dataset cifar10 --model resnet18 --setting sgd --input_file wandb_links.yaml --output_dir out/ > out/find_barriers_cifar10_resnet18_sgd.out &
# VGG
python find_barriers.py --dataset cifar10 --model vgg11 --setting sgd --input_file wandb_links.yaml --output_dir out/ > out/find_barriers_cifar10_vgg11_sgd.out &
python find_barriers.py --dataset cifar10 --model vgg19 --setting sgd --input_file wandb_links.yaml --output_dir out/ > out/find_barriers_cifar10_vgg19_sgd.out &
# python find_barriers.py --dataset cifar100 --model resnet18 --setting sgd --input_file wandb_links.yaml --output_dir out/ > out/find_barriers_cifar100_resnet18_sgd.out &