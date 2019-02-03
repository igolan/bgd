# Bayesian Gradient Descent (BGD) - Task Agnostic Continual Learning

This is an implementation of Bayesian Gradient Descent (BGD), an algorithm for continual learning which is applicable to scenarios where task identity or boundaries are unknown during both training and testing — task-agnostic continual learning.  
It is based on the online version of variational Bayes, and learns a posterior distribution on a deep neural network weights.

Please see our paper, [Task Agnostic Continual Learning Using Online Variational Bayes](https://arxiv.org/abs/1803.10123), for further details.

## Classic continual learning experiments
BGD works on any continual learning problem.

For example, to run Continuous task-agnostic Permuted MNIST experiment, use the command:
```
python main.py --logname continuous_permuted_mnist_10tasks --num_workers 1 --test_freq 10 --permute_seed 2019 --seed 2019 --iterations_per_virtual_epc 469 --contpermuted_beta 4 --num_of_permutations $(( 10 - 1 )) --optimizer bgd --nn_arch mnist_simple_net_200width_domainlearning_1024input_10cls_1ds --dataset ds_padded_cont_permuted_mnist --num_epochs $(( 20 * 10)) --std_init 0.06 --batch_size 128 --results_dir res --train_mc_iters 10 --inference_mc --test_mc_iters 10 --inference_map --inference_committee --committee_size 10 --inference_aggsoftmax --inference_initstd --desc BGD on continuous permuted mnist
```

See more examples on different experiments in [./run_exps.sh](./run_exps.sh).


## Using BGD PyTorch optimizer

This code includes implementation of BGD as a PyTorch optimizer, allowing easy integration to existing code.

Using BGD in your code requires few adaptations:
* Use BGD as the optimizer. ([./optimizers_lib/bgd_optimizer.py](./optimizers_lib/bgd_optimizer.py) )
* Add Monte-Carlo loop inside the batch loop.
* Randomize weights in the beginning of the Monte-Carlo loop.
* Use bgd_optimizer.aggregate_grads(batch_size) after every .backward().
* Use bgd_optimizer.step() on the end of the Monte-Carlo iterations.

A pseudo-code example of using BGD:
```
For samples, labels in data:
    for mc_iter in range(mc_iters):
        bgd_optimizer.randomize_weights()
        output = model.forward(samples)
        loss = cirterion(output, labels)
        bgd_optimizer.zero_grad()
        loss.backward()
        bgd_optimizer.aggregate_grads(batch_size)
    optimizer.step()
```

## Continuous task-agnostic Permuted MNIST experiment

In the paper, we present the Continuous task-agnostic Permuted MNIST experiment, where task-switch is performed slowly over time.
To create similar experiment, you can use the included sampler - ContinuousMultinomialSampler.
The sampler, and the function of creating data for the continuous experiment are at [./utils/datasets.py](./utils/datasets.py) (relevant function: ds_padded_cont_permuted_mnist()).

This will produce the following distribution of tasks over iterations:
![Distribution of samples from each task as a function of iteration. The tasks are not changed abruptly, but slowly over time --- tasks boundaries are undefined. Moreover, the algorithm has no access to this distribution. Here, number of samples from each task in each batch is a random variable drawn from a distribution over tasks, and this distribution changes over time (iterations).](/images/tasks_distribution.png)

## Labels trick
The labels trick is used for "class learning", where different tasks do not share the same label space (and do not share output heads).
In such case, we train only the heads of labels appear in the current mini-batch.

It is implemented in [./nn_utils/NNTrainer.py](./nn_utils/NNTrainer.py) (search for the term "labels_trick").
Integrating the labels trick in your code is easy, replace the loss calculation with this code:
```
if labels_trick and training:
    # Use labels trick
    # Get current batch labels (and sort them for reassignment)
    unq_lbls = labels.unique().sort()[0]
    # Assign new labels (0,1 ...)
    for lbl_idx, lbl in enumerate(unq_lbls):
        labels[labels == lbl] = lbl_idx
    # Calcualte loss only over the heads appear in the batch:
    loss = criterion(outputs[:, unq_lbls], labels)
else:
    loss = criterion(outputs, labels)

```


## Hyper-parameters
When using BGD, the main hyper-parameters which need adjustment are:
* STD init, usually in the range [0.01, 0.06].
* Eta, usually in the range [1,10].
* Initialization scheme (usually He/Xavier initialization) - BGD intialize the mean parameter using the initialization weights. Different initialization schemes might affect performance.  
Examples for hyper-parameters:

| Neytwork                                        | STD init | eta | 
| -------------                                   | -------- | --- |
| Fully-connected, 2 hidden layers of width 200   | 0.06     | 1   |
| Fully-connected, 2 hidden layers of width 400   | 0.05     | 1   |
| Fully-connected, 2 hidden layers of width 800   | 0.05     | 1   |
| Fully-connected, 2 hidden layers of width 1200  | 0.04     | 1   |
| Fully-connected, 2 hidden layers of width 2000  | 0.015    | 1   |
| VGG11-like, with Batch-Norm                     | 0.011    | 8   |
| VGG11-like, without Batch-Norm                  | 0.015    | 10  |




## Requirements
See [./requirements.txt](./requirements.txt)

