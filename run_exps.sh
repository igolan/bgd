#!/bin/bash
# Examples for running different types of experiments:
# If you are not using CUDA, remove the "CUDA_VISIBLE_DEVICES=X" prefix.
# Hyper-parameters are not necessarily optimal.


seeds=( 2019 2020 2021 2022 2023 2024 2025 2026 2027 2028 )

for seed in "${seeds[@]}"
do
    ######
    ## Discrete task agnostic on Permuted MNIST (10 epochs per task) domain learning
    ######
    CUDA_VISIBLE_DEVICES=3 python main.py --logname discrete_domain_permuted_mnist_10tasks_10epochs_seed${seed} --nn_arch mnist_simple_net_1000width_domainlearning_1024input_10cls_1ds --test_freq 50 --seed ${seed} --permute_seed ${seed} --dataset ds_padded_permuted_mnist --num_epochs $(( 10 * 10 )) --optimizer bgd --std_init 0.045 --batch_size 128 --results_dir permuted_domain_1000width_10epc --train_mc_iters 10 --inference_map --desc desc

    ######
    ## Discrete task agnostic on Permuted MNIST (10 epochs per task) class learning
    ######
    CUDA_VISIBLE_DEVICES=3 python main.py --separate_labels_space --logname discrete_class_permuted_mnist_10tasks_10epochs_seed${seed} --nn_arch mnist_simple_net_1000width_classlearning_1024input_100cls_1ds --test_freq 50 --seed ${seed} --permute_seed ${seed} --dataset ds_padded_permuted_mnist --num_epochs $(( 10 * 10 )) --optimizer bgd --std_init 0.02 --batch_size 128 --results_dir permuted_class_1000width_10epc --train_mc_iters 10 --inference_map --desc desc

    ######
    ## Discrete task agnostic on Permuted MNIST (10 epochs per task) class learning, +Labels trick
    ######
    CUDA_VISIBLE_DEVICES=3 python main.py --labels_trick --separate_labels_space --logname discrete_class_permuted_mnist_lblstrck_10tasks_10epochs_seed${seed} --nn_arch mnist_simple_net_1000width_classlearning_1024input_100cls_1ds --test_freq 50 --seed ${seed} --permute_seed ${seed} --dataset ds_padded_permuted_mnist --num_epochs $(( 10 * 10 )) --optimizer bgd --std_init 0.02 --batch_size 128 --results_dir permuted_class_1000width_10epc --train_mc_iters 10 --inference_map --desc desc

    ######
    ## Discrete task agnostic on Permuted MNIST (10 epochs per task) task learning
    ######
    CUDA_VISIBLE_DEVICES=3 python main.py --logname discrete_task_permuted_mnist_10tasks_10epochs_seed${seed} --nn_arch mnist_simple_net_1000width_tasklearning_1024input_10cls_10ds --test_freq 50 --seed ${seed} --permute_seed ${seed} --dataset ds_padded_permuted_mnist --num_epochs $(( 10 * 10 )) --optimizer bgd --std_init 0.02 --batch_size 128 --results_dir permuted_task_1000width_10epc --train_mc_iters 10 --inference_map --desc desc


    ######
    ## Discrete task agnostic on Split MNIST (10 epochs per task) domain learning
    ######
    CUDA_VISIBLE_DEVICES=3 python main.py --logname discrete_domain_split_mnist_5tasks_4epochs_seed${seed} --nn_arch mnist_simple_net_400width_domainlearning_1024input_2cls_1ds --test_freq 10 --seed ${seed} --permute_seed ${seed} --dataset ds_padded_split_mnist --num_epochs $(( 5 * 4 )) --optimizer bgd --std_init 0.02 --batch_size 128 --results_dir split_domain_1000width_10epc --train_mc_iters 10 --inference_mc --test_mc_iters 10 --inference_map --inference_committee --committee_size 10 --inference_aggsoftmax --inference_initstd --desc desc

    ######
    ## Discrete task agnostic on Split MNIST (10 epochs per task) class learning
    ######
    CUDA_VISIBLE_DEVICES=3 python main.py --separate_labels_space --logname discrete_class_split_mnist_5tasks_4epochs_seed${seed} --nn_arch mnist_simple_net_400width_domainlearning_1024input_10cls_1ds --test_freq 10 --seed ${seed} --permute_seed ${seed} --dataset ds_padded_split_mnist --num_epochs $(( 5 * 4 )) --optimizer bgd --std_init 0.017 --batch_size 128 --results_dir split_class_1000width_10epc --train_mc_iters 10 --inference_mc --test_mc_iters 10 --inference_map --inference_committee --committee_size 10 --inference_aggsoftmax --inference_initstd --desc desc

    ######
    ## Discrete task agnostic on Split MNIST (10 epochs per task) class learning, +Labels trick
    ######
    CUDA_VISIBLE_DEVICES=3 python main.py --labels_trick --separate_labels_space --logname discrete_class_split_mnist_lblstrck_5tasks_4epochs_seed${seed} --nn_arch mnist_simple_net_400width_domainlearning_1024input_10cls_1ds --test_freq 10 --seed ${seed} --permute_seed ${seed} --dataset ds_padded_split_mnist --num_epochs $(( 5 * 4 )) --optimizer bgd --std_init 0.017 --batch_size 128 --results_dir split_class_1000width_10epc --train_mc_iters 10 --inference_mc --test_mc_iters 10 --inference_map --inference_committee --committee_size 10 --inference_aggsoftmax --inference_initstd --desc desc

    ######
    ## Discrete task agnostic on Split MNIST (10 epochs per task) task learning
    ######
    CUDA_VISIBLE_DEVICES=3 python main.py --logname discrete_task_split_mnist_5tasks_4epochs_seed${seed} --nn_arch mnist_simple_net_400width_tasklearning_1024input_2cls_5ds --test_freq 10 --seed ${seed} --permute_seed ${seed} --dataset ds_padded_split_mnist --num_epochs $(( 5 * 4 )) --optimizer bgd --std_init 0.02 --batch_size 128 --results_dir split_task_1000width_10epc --train_mc_iters 10 --inference_mc --test_mc_iters 10 --inference_map --inference_committee --committee_size 10 --inference_aggsoftmax --inference_initstd --desc desc

done

#####
# Continuous task agnostic on Permuted MNIST
#####
# 10 tasks (which is 9 permutations in addition to the original MNIST)
num_tasks=10
CUDA_VISIBLE_DEVICES=1 python main.py --logname continuous_permuted_mnist_10tasks --num_workers 1 --test_freq 10 --permute_seed 2019 --seed 2019 --iterations_per_virtual_epc 469 --contpermuted_beta 4 --num_of_permutations $(( num_tasks - 1 )) --optimizer bgd --nn_arch mnist_simple_net_200width_domainlearning_1024input_10cls_1ds --dataset ds_padded_cont_permuted_mnist --num_epochs $(( 20 * num_tasks )) --std_init 0.06 --batch_size 128 --results_dir res --train_mc_iters 10 --inference_mc --test_mc_iters 10 --inference_map --inference_committee --committee_size 10 --inference_aggsoftmax --inference_initstd --desc BGD on continuous permuted mnist
# 5 tasks (which is 4 permutations in addition to the original MNIST)
num_tasks=5
CUDA_VISIBLE_DEVICES=1 python main.py --logname continuous_permuted_mnist_5tasks --num_workers 1 --test_freq 10 --permute_seed 2019 --seed 2019 --iterations_per_virtual_epc 469 --contpermuted_beta 4 --num_of_permutations $(( num_tasks - 1 )) --optimizer bgd --nn_arch mnist_simple_net_200width_domainlearning_1024input_10cls_1ds --dataset ds_padded_cont_permuted_mnist --num_epochs $(( 20 * num_tasks )) --std_init 0.06 --batch_size 128 --results_dir res --train_mc_iters 10 --inference_mc --test_mc_iters 10 --inference_map --inference_committee --committee_size 10 --inference_aggsoftmax --inference_initstd --desc BGD on continuous permuted mnist

#####
# Discrete task agnostic on Permuted MNIST (10 epochs per task) - domain learning
#####
CUDA_VISIBLE_DEVICES=1 python main.py --logname discrete_permuted_mnist_10tasks_10epochs --nn_arch mnist_simple_net_200width_domainlearning_1024input_10cls_1ds --test_freq 5 --seed 2019 --permute_seed 2019 --dataset ds_padded_permuted_mnist --num_epochs $(( 10 * 10 )) --optimizer bgd --std_init 0.02 --batch_size 128 --results_dir res --train_mc_iters 10 --inference_mc --test_mc_iters 10 --inference_map --inference_committee --committee_size 10 --inference_aggsoftmax --inference_initstd --desc desc

#####
# Discrete task agnostic on Permuted MNIST (300 epochs per task) - domain learning (not padded MNIST)
#####
CUDA_VISIBLE_DEVICES=1 python main.py --logname discrete_permuted_mnist_10tasks_300epochs --nn_arch mnist_simple_net_200width_domainlearning_784input_10cls_1ds --test_freq 5 --seed 2019 --permute_seed 2019 --dataset ds_permuted_mnist --num_epochs $(( 300 * 10 )) --optimizer bgd --std_init 0.06 --batch_size 128 --results_dir res --train_mc_iters 10 --inference_mc --test_mc_iters 10 --inference_map --inference_committee --committee_size 10 --inference_aggsoftmax --inference_initstd --desc desc


#####
# Task learning on Vision mix (MNIST, notMNIST, FashionMNIST, SVHN and CIFAR10), 20 epochs per task.
#####
CUDA_VISIBLE_DEVICES=1 python main.py --logname task_learning_visionmix --nn_arch lenet --test_freq 5 --seed 2019 --dataset ds_visionmix --bw_to_rgb --num_epochs $(( 20 * 5 )) --optimizer bgd --std_init 0.02 --batch_size 64 --results_dir res --train_mc_iters 10 --inference_mc --test_mc_iters 10 --inference_map --inference_committee --committee_size 10 --inference_aggsoftmax --inference_initstd --desc desc

#####
# Task learning on CIFAR10/CIFAR100, 20 epochs per task.
#####
CUDA_VISIBLE_DEVICES=1 python main.py --logname task_learning_cifar10and100 --nn_arch zenke_net --test_freq 5 --seed 2019 --dataset ds_cifar10and100 --num_epochs $(( 150 * 6 )) --optimizer bgd --std_init 0.019 --mean_eta 2 --batch_size 256 --results_dir res --train_mc_iters 10 --inference_mc --test_mc_iters 10 --inference_map --inference_committee --committee_size 10 --inference_aggsoftmax --inference_initstd --desc desc

#####
# CIFAR10 classification
#####
CUDA_VISIBLE_DEVICES=1 python main.py --logname cifar10_classification --nn_arch vgg16zhang_with_bn --test_freq 5 --seed 2019 --dataset ds_cifar10 --num_epochs $(( 400 )) --optimizer bgd --std_init 0.011 --mean_eta 8 --batch_size 128 --results_dir res --train_mc_iters 10 --inference_mc --test_mc_iters 10 --inference_map --inference_committee --committee_size 10 --inference_aggsoftmax --inference_initstd --desc desc
