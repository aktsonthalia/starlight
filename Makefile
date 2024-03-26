cifar10_wrn_16_1:
	rm results/wandb_links/cifar10_wrn_16_1.yaml
	python train_many.py -c configs/cifar10_wrn_16_1.yaml -w results/wandb_links/cifar10_wrn_16_1.yaml -o 5 -a 10