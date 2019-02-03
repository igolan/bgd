from utils.utils import *
from utils.logging_utils import Logger
import utils.datasets
from nn_utils.NNTrainer import NNTrainer
from probes_lib.top import *
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch
import argparse
import models
from time import time
import pickle
import os
import socket
import sys
from ast import literal_eval
from nn_utils.init_utils import init_model
import optimizers_lib


###########################################################################
# Script's arguments
###########################################################################
archs_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
optimizers_names = sorted(name for name in optimizers_lib.__dict__
                          if name.islower() and not name.startswith("__")
                          and callable(optimizers_lib.__dict__[name]))
datasets_names = sorted(name for name in utils.datasets.__dict__
                        if name.islower() and name.startswith("ds_")
                        and callable(utils.datasets.__dict__[name]))


parser = argparse.ArgumentParser(description='Train and record statistics of a Neural Network with BGD')
parser.add_argument('--dataset', default="ds_mnist", type=str, choices=datasets_names,
                    help='The name of the dataset to train. [Default: ds_mnist]')
parser.add_argument('--nn_arch', type=str, required=True, choices=archs_names,
                    help='Neural network architecture')
parser.add_argument('--logname', type=str, required=True,
                    help='Prefix of logfile name')
parser.add_argument('--results_dir', type=str, default="TMP",
                    help='Results dir name')
parser.add_argument('--seed', type=int,
                    help='Seed for randomization.')
parser.add_argument('--num_workers', default=8, type=int,
                    help='Num of workers for data loader')
parser.add_argument('--num_epochs', default=400, type=int,
                    help='Maximum number of training epochs. [Default: 400]')
parser.add_argument('--batch_size', default=128, type=int,
                    help='Batch size [Default: 128]')
parser.add_argument('--pruning_percents', default=[], type=int, nargs='*',
                    help='A list of percents to check pruning with [Default: []]')

# BGD
parser.add_argument('--train_mc_iters', default=10, type=int,
                    help='Number of MonteCarlo samples during training(default 10)')
parser.add_argument('--std_init', default=5e-2, type=float,
                    help='STD init value (default 5e-2)')
parser.add_argument('--mean_eta', default=1, type=float,
                    help='Eta for mean step (default 1)')

parser.add_argument('--permanent_prune_on_epoch', default=-1, type=int,
                    help='Permanent prune on epoch')
parser.add_argument('--permanent_prune_on_epoch_percent', default=90, type=float,
                    help='Permanent prune percent of weights')

parser.add_argument('--test_freq', default=1, type=int,
                    help='Run test set every test_freq epochs [default: 1]')
parser.add_argument('--contpermuted_beta', default=3, type=int,
                    help='Beta value for continuous permuted mnist. [default: 3]')

parser.add_argument('--optimizer', type=str, default="bgd", choices=optimizers_names,
                    help='Optimizer.')
parser.add_argument('--optimizer_params', default="{}", type=str, nargs='*',
                    help='Optimizer parameters')


parser.add_argument('--inference_mc', default=False, action='store_true',
                    help='Use MonteCarlo samples as inference method.')
parser.add_argument('--inference_map', default=False, action='store_true',
                    help='Use MAP as inference method.')
parser.add_argument('--inference_committee', default=False, action='store_true',
                    help='Use committee as inference method.')
parser.add_argument('--inference_aggsoftmax', default=False, action='store_true',
                    help='Use aggsoftmax as inference method.')
parser.add_argument('--inference_initstd', default=False, action='store_true',
                    help='Use initstd as inference method.')


parser.add_argument('--committee_size', default=0, type=int,
                    help='Size of committee (when using committee inference)')
parser.add_argument('--test_mc_iters', default=0, type=int,
                    help='Number of MC iters when testing (when using MC inference)')

parser.add_argument('--init_params',
                    default=["{\"bias_type\":", "\"xavier\",", "\"conv_type\":", "\"xavier\",",
                             "\"bn_init\":", "\"01\"}"], type=str, nargs='*', help='Initialization parameters')

parser.add_argument('--desc', default="", type=str, nargs='*',
                    help='Desc file content')

parser.add_argument('--bw_to_rgb', default=False, action='store_true',
                    help='Convert black&white (e.g. MNIST) images to RGB format')

parser.add_argument('--permuted_offset', default=False, action='store_true',
                    help='Use offset for permuted mnist experiment')
parser.add_argument('--labels_trick', default=False, action='store_true',
                    help='Use labels trick (train only the heads of current batch labels)')

parser.add_argument('--num_of_permutations', default=9, type=int,
                    help='Number of permutations (in addition to the original MNIST) ' 
                         'when using Permuted MNIST dataset [default: 9]')
parser.add_argument('--iterations_per_virtual_epc', default=468, type=int,
                    help='When using continuous dataset, number of iterations per epoch (in continuous mode, '
                         'epoch is not defined)')

parser.add_argument('--separate_labels_space', default=False, action='store_true',
                    help='Use separate label space for each task')

parser.add_argument('--permute_seed', type=int,
                    help='Seed for creating the permutations.')



args = parser.parse_args()

###########################################################################
# Verify arguments
###########################################################################
inference_methods = set()
if args.inference_committee:
    assert (args.committee_size > 0)
    inference_methods.add("committee")
if args.inference_mc:
    assert (args.test_mc_iters > 0)
    inference_methods.add("test_mc")
if args.inference_map:
    inference_methods.add("map")
if args.inference_aggsoftmax:
    inference_methods.add("agg_softmax")
if args.inference_initstd:
    inference_methods.add("init_std")

assert(len(inference_methods) > 0)

if args.optimizer != "bgd":
    assert args.train_mc_iters == 1, "Monte Carlo iterations are for BGD optimizer only"
    assert len(inference_methods) == 1 and "map" in inference_methods, "When not using BGD, must use MAP for inference"

###########################################################################
# CUDA and seeds
###########################################################################

# Create permuations
all_permutation = []
if not args.permute_seed:
    args.permute_seed = int(time()*10000) % (2**31)

set_seed(args.permute_seed, fully_deterministic=False)
for p_idx in range(args.num_of_permutations):
    input_size = 32 * 32
    if "padded" not in args.dataset:
        input_size = 28 * 28
    permutation = list(range(input_size))
    random.shuffle(permutation)
    all_permutation.append(permutation)

# Set seed
if not args.seed:
    args.seed = int(time()*10000) % (2**31)

set_seed(args.seed, fully_deterministic=False)

if torch.cuda.is_available():
    cudnn.benchmark = True


###########################################################################
# Logging
###########################################################################
# Create logger
save_path = os.path.join("./logs", str(args.results_dir) + "/")
if not os.path.exists(save_path):
    os.makedirs(save_path)

logger = Logger(True, save_path + args.logname, True, True)

logger.info("Script args: " + str(args))

if args.desc != "":
    logger.create_desc_file(" ".join(args.desc))
else:
    logger.create_desc_file(str(args))


logger.info("Computer name: " + str(socket.gethostname()) + " with pytorch version: " + str(torch.__version__))

lastlogs_logger = Logger(add_timestamp=False, logfile_name="last_logs.txt", logfile_name_time_suffix=False,
                         print_to_screen=False)
lastlogs_logger.info(logger.get_log_basename() + " ")
lastlogs_logger = None


###########################################################################
# Model and training
###########################################################################

# Dataset
train_loaders, test_loaders = utils.datasets.__dict__[args.dataset](batch_size=args.batch_size,
                                                                    num_workers=args.num_workers,
                                                                    permutations=all_permutation,
                                                                    separate_labels_space=args.separate_labels_space,
                                                                    num_epochs=args.num_epochs,
                                                                    iterations_per_virtual_epc=
                                                                    args.iterations_per_virtual_epc,
                                                                    contpermuted_beta=args.contpermuted_beta,
                                                                    logger=logger)

# Probes manager
probes_manager = ProbesManager()

# Model
model = models.__dict__[args.nn_arch](probes_manager=probes_manager)
if torch.cuda.is_available():
    model = torch.nn.DataParallel(model).cuda()
    logger.info("Transformed model to CUDA")

criterion = nn.CrossEntropyLoss()

init_params = {"logger": logger}
if args.init_params != "":
    init_params = dict(init_params, **literal_eval(" ".join(args.init_params)))

init_model(get_model(model), **init_params)

# optimizer model
optimizer_model = optimizers_lib.__dict__[args.optimizer]
optimizer_params = dict({"logger": logger,
                         "mean_eta": args.mean_eta,
                         "std_init": args.std_init,
                         "mc_iters": args.train_mc_iters}, **literal_eval(" ".join(args.optimizer_params)))
optimizer = optimizer_model(model, probes_manager=probes_manager, **optimizer_params)

trainer = NNTrainer(train_loader=train_loaders, test_loader=test_loaders,
                    criterion=criterion, net=model, logger=logger, probes_manager=probes_manager,
                    std_init=args.std_init, mean_eta=args.mean_eta, train_mc_iters=args.train_mc_iters,
                    test_mc_iters=args.test_mc_iters, committee_size=args.committee_size, batch_size=args.batch_size,
                    inference_methods=inference_methods,
                    pruning_percents=args.pruning_percents,
                    bw_to_rgb=args.bw_to_rgb,
                    labels_trick=args.labels_trick,
                    test_freq=args.test_freq,
                    optimizer=optimizer)


trainer.train_epochs(verbose_freq=100, max_epoch=args.num_epochs,
                     permanent_prune_on_epoch=args.permanent_prune_on_epoch,
                     permanent_prune_on_epoch_percent=args.permanent_prune_on_epoch_percent)


print("Done")
