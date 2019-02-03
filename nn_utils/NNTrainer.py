from utils.logging_utils import Logger
from utils.utils import *
from probes_lib.basic import *
from utils.utils import AverageTracker
from optimizers_lib.bgd_optimizer import BGD
import torch


class NNTrainer:
    def __init__(self, train_loader, test_loader, criterion, net, logger, **kwargs):
        # NN Configurations variables
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion       # type: torch.nn.modules.loss
        self.logger = logger             # type: Logger
        self.net = net                   # type: torch.nn.Module
        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')

        self.desc = kwargs.get("desc", None)

        self.std_init = kwargs.get("std_init", 5e-2)
        self.batch_size = kwargs.get("batch_size", None)
        self.mean_eta = kwargs.get("mean_eta", 1)

        self.test_freq = kwargs.get("test_freq", 1)

        self.bw_to_rgb = kwargs.get("bw_to_rgb", False)

        self.labels_trick = kwargs.get("labels_trick", False)

        # Statistics variables
        self.epochs_trained = 0
        self.probes_manager = kwargs.get("probes_manager", None)

        self.inference_methods = kwargs.get("inference_methods", {"test_mc"})
        self.logger.info("Inference method: " + str(self.inference_methods))

        self.test_mc_iters = kwargs.get("test_mc_iters", 0)
        self.committee_size = kwargs.get("committee_size", 0)

        # Create probes
        self.probes_manager.add_probe(probe=AccLossProbe(type="train"), probe_name="train_acc_loss",
                                      probe_locs=["post_train_forward"])
        for inference_method in self.inference_methods:
            self.probes_manager.add_probe(probe=AccLossProbe(type="test"),
                                          probe_name="test_" + inference_method + "_acc_loss",
                                          probe_locs=["post_" + inference_method + "_test_forward"])
            for ds_idx in range(0, self.test_loader.__len__()):
                self.probes_manager.add_probe(probe=AccLossProbe(type="test"),
                                              probe_name="test_" + inference_method + "_acc_loss" + str(ds_idx),
                                              probe_locs=["post_" + inference_method + "_test_forward_dataset" +
                                                          str(ds_idx)])
                self.probes_manager.add_probe(probe=AvgAccOverTasksProbe(type="test"),
                                              probe_name="test_" + inference_method + "_acc_avg_acc_tasks" +
                                                         str(ds_idx),
                                              probe_locs=["post_" + inference_method + "_test_forward_dataset" +
                                                          str(ds_idx)])

        self.probes_manager.add_probe(probe=TimingProbe(), probe_name="total_epoch_time",
                                      probe_locs=["epoch_end"])
        self.probes_manager.add_probe(probe=EpochNumProbe(),
                                      probe_name="epoch", probe_locs=["post_train_forward"])

        # Print parameters
        self.print_num_of_net_params()
        self.print_criterion_params()
        self.layers_names = {l: name for l, (name, _) in enumerate(self.net.named_parameters())}

        self.net_num_of_params = 0
        for parameter in self.net.parameters():
            self.net_num_of_params += parameter.numel()

        # If BatchNorm layers exists in the network we will save the running stats on the first MC iteration, and
        #   restore after the last MC iteration.
        self.save_bn_runnings = bn_exists_in_net(self.net)

        self.optimizer = kwargs.get("optimizer", None)

        # Time recorders:
        self.single_mc_iter_time = None  # TimeRecorder()
        self.single_mc_iter_time_avg = {"test": AverageTracker(),
                                        "train": AverageTracker()}

        self.all_mc_iters_time = None  # TimeRecorder()
        self.all_mc_iters_time_avg = {"test": AverageTracker(),
                                      "train": AverageTracker()}

        self.total_minibatch_time = None  # TimeRecorder()
        self.total_minibatch_time_avg = {"test": AverageTracker(),
                                         "train": AverageTracker()}

        self.total_pass_time = None  # TimeRecorder()
        self.total_pass_time_avg = {"test": AverageTracker(),
                                    "train": AverageTracker()}

        self.tasks_avg_output = {}

    # Training and testing
    def train_epochs(self, verbose_freq=2000, max_epoch=1, **kwargs):
        if self.epochs_trained >= max_epoch:
            return False

        self.logger.info("Running training from epoch " + str(self.epochs_trained + 1) + " to epoch " + str(max_epoch))

        for epoch_number in range(self.epochs_trained, max_epoch):
            # Time recorders
            epoch_time = TimeRecorder()

            #####
            # Train
            #####
            if self.save_stats_on_epoch(epoch_number, max_epoch):
                self.probes_manager.epoch_prologue()

            self.logger.info("Training epoch number " + str(epoch_number + 1) + " with dataset number " +
                             str(self.get_dataset_idx(max_epoch=max_epoch)))
            self.train_mode()

            # Set current dataset
            if hasattr(get_model(self.net), "set_dataset"):
                get_model(self.net).set_dataset(self.get_dataset_idx(max_epoch=max_epoch))
            train_loss, train_acc = self.forward(
                data_loader=self.train_loader[self.get_dataset_idx(max_epoch=max_epoch)], training=True,
                verbose_freq=verbose_freq)
            if self.save_stats_on_epoch(epoch_number, max_epoch):
                probe_data = {"train_loss": train_loss,
                              "train_acc": train_acc,
                              "net": self.net,
                              "weights": self.weights_lst(),
                              "epochs_trained": self.epochs_trained}
                self.probes_manager.add_data(probe_loc="post_train_forward", **probe_data)

            #####
            # Test
            #####
            # Run test set for every inference method
            if self.save_stats_on_epoch(epoch_number, max_epoch):
                for inference_method in self.inference_methods:
                    avg_acc_over_tasks = AverageTracker()
                    # Run test set for every dataset
                    ds_idx = 0
                    for ds_idx in range(0, self.test_loader.__len__()):
                        self.logger.info("Running test set for epoch number " + str(epoch_number + 1) +
                                         " for dataset idx " + str(ds_idx) + " using " + str(inference_method))
                        self.eval_mode()
                        if hasattr(get_model(self.net), "set_dataset"):
                            get_model(self.net).set_dataset(ds_idx)
                        test_loss, test_acc = self.forward(data_loader=self.test_loader[ds_idx], training=False,
                                                           verbose_freq=0, inference_method=inference_method)
                        avg_acc_over_tasks.add(test_acc)
                        probe_data = {"test_loss": test_loss,
                                      "test_acc": test_acc,
                                      "net": self.net,
                                      "weights": self.weights_lst(),
                                      "avg_acc_over_tasks": avg_acc_over_tasks.avg,
                                      "epochs_trained": self.epochs_trained}
                        if ds_idx == 0:
                            self.probes_manager.add_data(probe_loc="post_" + inference_method + "_test_forward",
                                                         **probe_data)
                        self.probes_manager.add_data(probe_loc="post_" + inference_method + "_test_forward_dataset" +
                                                               str(ds_idx), **probe_data)
                    self.logger.info("Average accuracy over all tasks for epoch number " + str(epoch_number + 1) +
                                     " for dataset idx " + str(ds_idx) + " using " + str(inference_method) + " is " +
                                     str(avg_acc_over_tasks.avg))

            if hasattr(get_model(self.net), "set_dataset"):
                get_model(self.net).set_dataset(0)

            if self.save_stats_on_epoch(epoch_number, max_epoch):
                probe_data = {"net": self.net,
                              "avg_time": epoch_time.get_time(),
                              "weights": self.weights_lst(),
                              "epochs_trained": self.epochs_trained}
                self.probes_manager.add_data(probe_loc="epoch_end", **probe_data)
            #####
            # Save statistics and model
            #####
            if self.save_stats_on_epoch(epoch_number, max_epoch):
                self.probes_manager.epoch_epilogue()
                self.probes_manager.calc_epoch_stats()
                self.save_current_stats()
            self.logger.info("Finished epoch number " + str(epoch_number + 1) +
                             ", Took " + str(int(epoch_time.get_time())) + " seconds")
            for fwd_type in ["train", "test"]:
                if self.all_mc_iters_time is not None:
                    self.logger.info(fwd_type + ": All MC iters takes " +
                                     str(self.all_mc_iters_time_avg[fwd_type].avg) + " seconds on average")
                if self.single_mc_iter_time is not None:
                    self.logger.info(fwd_type + ": Single MC iter takes " +
                                     str(self.single_mc_iter_time_avg[fwd_type].avg) + " seconds on average")
                if self.total_minibatch_time is not None:
                    self.logger.info(fwd_type + ": A single minibatch " +
                                     str(self.total_minibatch_time_avg[fwd_type].avg) + " seconds on average")
                if self.total_pass_time is not None:
                    self.logger.info(fwd_type + ": Total pass takes " +
                                     str(self.total_pass_time_avg[fwd_type].avg) + " seconds on average")

        return True

    def get_dataset_idx(self, max_epoch, epoch_num=None):
        # Datasets (tasks) are splitted evenly over the training
        epoch_num = epoch_num or self.epochs_trained
        return min(epoch_num // (max_epoch // self.train_loader.__len__()), self.train_loader.__len__())

    def save_stats_on_epoch(self, epoch_number, max_epoch):
        # Returns whether or not to save statistics on current epoch.
        # Save every self.test_freq epochs, or when dataset (task) is switched.
        dataset_idx_prev = self.get_dataset_idx(max_epoch=max_epoch, epoch_num=self.epochs_trained - 1)
        dataset_idx_cur = self.get_dataset_idx(max_epoch=max_epoch, epoch_num=self.epochs_trained)
        dataset_idx_next = self.get_dataset_idx(max_epoch=max_epoch, epoch_num=self.epochs_trained + 1)
        return (epoch_number <= 1 or epoch_number % self.test_freq == 0 or epoch_number >= (max_epoch - 1) or
                dataset_idx_cur != dataset_idx_prev or dataset_idx_cur != dataset_idx_next)

    def save_current_stats(self):
        # Saves the statistics to a file.
        samples_distribution_over_time = None
        if hasattr(self.train_loader[0].sampler, "samples_distribution_over_time"):
            samples_distribution_over_time = self.train_loader[0].sampler.samples_distribution_over_time
        vars_dict = {'per_epoch_stats': self.probes_manager.per_epoch_stats,
                     'epochs_trained': self.epochs_trained,
                     'layers_names': self.layers_names,
                     'desc': self.desc,
                     "samples_distribution_over_time": samples_distribution_over_time}
        self.logger.save_variables(var=vars_dict, var_name="stats")

    def weights_lst(self):
        return [w.data for w in list(self.net.parameters())]

    def weights_grad_lst(self):
        return [w.grad for w in list(self.net.parameters())]

    def train_mode(self):
        self.net.train()
        self.probes_manager.train()

    def eval_mode(self):
        self.net.eval()
        self.probes_manager.eval()

    def forward(self, data_loader=None, verbose_freq=2000, training=True, inference_method="test_mc"):
        if training:
            self.train_mode()
            fwd_name = "train"
        else:
            self.eval_mode()
            fwd_name = "test"

        loss_avg = AverageTracker()
        acc_avg = AverageTracker()
        set_size = 0

        self.total_pass_time.reset() if self.total_pass_time is not None else None

        for i, data in enumerate(data_loader, 0):
            self.total_minibatch_time.reset() if self.total_minibatch_time is not None else None

            # Get data
            inputs, labels = data
            inputs, labels = (inputs.to(self.device), labels.to(self.device))

            if self.bw_to_rgb and inputs.shape[1] == 1:
                inputs = inputs.repeat(1, 3, 1, 1)

            # Reset per-minibatch variables
            # Initialize runnings, used to save batch-norm running stats.
            runnings = None
            # Initialize votes, used to for committee inference method.
            votes = None
            # Initialize outputs_aggregated, used to for aggregated softmax inference method.
            outputs_aggregated = None

            # MonteCarlo samples
            self.all_mc_iters_time.reset() if self.all_mc_iters_time is not None else None
            if not training:
                if inference_method == "map":
                    # If using map, we don't need MonteCarlo samples for inference
                    num_of_mc_iters = 1
                elif inference_method == "committee":
                    num_of_mc_iters = self.committee_size
                elif inference_method == "test_mc" or inference_method == "agg_softmax":
                    num_of_mc_iters = self.test_mc_iters
                elif inference_method == "init_std":
                    num_of_mc_iters = self.test_mc_iters
                else:
                    assert False
            else:
                num_of_mc_iters = 1
                if hasattr(self.optimizer, "get_mc_iters"):
                    num_of_mc_iters = self.optimizer.get_mc_iters()

            for k in range(0, num_of_mc_iters):
                torch.autograd.set_grad_enabled(training)
                self.single_mc_iter_time.reset() if self.single_mc_iter_time is not None else None
                if type(self.optimizer) == BGD:
                    # Randomize weights
                    if not training and inference_method == "map":
                        # If using map, weights are deterministic given their mean. Generate weights using std=0
                        self.optimizer.randomize_weights(force_std=0)
                    elif not training and inference_method == "init_std":
                        self.optimizer.randomize_weights(force_std=self.optimizer.std_init)
                    else:
                        self.optimizer.randomize_weights()

                # Forward:
                outputs = self.net(inputs)

                # Save running mean and var for BatchNorm layers. We don't want each minibatch to go through the
                #      running statistics train_mc_iters times, thus we save the running stats after the first
                #      MonteCarlo iteration and restore it after the last iteration.
                if k == 0 and self.save_bn_runnings and training:
                    runnings = save_bn_running(self.net)

                # Calc loss
                if self.labels_trick and training:
                    # Use labels trick
                    if k == 0:
                        # We do so only when k==0 (first MC iteration) because we don't want to check the unique labels
                        #  twice (we assign the labels inplace)
                        # Get current batch labels (and sort them for reassignment)
                        unq_lbls = labels.unique().sort()[0]
                        # Assign new labels (0,1 ...)
                        for lbl_idx, lbl in enumerate(unq_lbls):
                            labels[labels == lbl] = lbl_idx
                    # Calcualte loss only over the heads appear in the batch:
                    loss = self.criterion(outputs[:, unq_lbls], labels)
                else:
                    loss = self.criterion(outputs, labels)

                loss_avg.add(loss.item(), inputs.size(0))
                if loss.item() != loss.item():
                    self.logger.info("Loss is NaN!!!")
                assert loss.item() == loss.item()  # Assert loss is not NaN

                # Backprop
                if training:
                    # Zero the gradient
                    self.optimizer.zero_grad()
                    loss.backward()
                    # Accumulate gradients
                    self.optimizer.aggregate_grads(batch_size=inputs.size(0))
                    self.probes_manager.add_data(probe_loc="post_backward_pre_optim_step",
                                                 weights_grad=self.weights_grad_lst(),
                                                 weights=self.weights_lst())
                # Make prediction on testing
                if not training:
                    outputs_softmax = torch.nn.functional.softmax(outputs, dim=1)
                    max_vals, predicted = torch.max(outputs_softmax.data, dim=1)
                    if inference_method == "committee":
                        # For committee, vote for current prediction
                        if votes is None:
                            votes = (max_vals.view(max_vals.shape[0], 1).repeat(1, outputs.data.shape[1]) ==
                                     outputs_softmax.data)
                        else:
                            votes.add_(max_vals.view(max_vals.shape[0], 1).repeat(1, outputs.data.shape[1]) ==
                                       outputs_softmax.data)
                    elif inference_method == "agg_softmax":
                        if outputs_aggregated is None:
                            outputs_aggregated = torch.nn.functional.softmax(outputs, dim=1).data
                        else:
                            outputs_aggregated.add_(torch.nn.functional.softmax(outputs, dim=1).data)
                    else:
                        # If not using committee, add current prediction to the average
                        correct_tags = (predicted == labels.data).sum()
                        acc_avg.add((float(correct_tags) / labels.size(0)) * 100, labels.size(0))
                # Record MonteCarlo single iteration time
                self.single_mc_iter_time_avg[fwd_name].add(self.single_mc_iter_time.get_time()
                                                           ) if self.single_mc_iter_time is not None else None
            torch.autograd.set_grad_enabled(False)
            # Record Total MonteCarlo iterations time
            self.all_mc_iters_time_avg[fwd_name].add(self.all_mc_iters_time.get_time()
                                                     ) if self.all_mc_iters_time is not None else None

            # If using committee for inference, calculate the majority vote and add to average
            if not training:
                if inference_method == "committee":
                    _, predicted = torch.max(votes, dim=1)
                    correct_tags = (predicted == labels.data).sum()
                    acc_avg.add((float(correct_tags) / labels.size(0)) * 100, labels.size(0))
                elif inference_method == "agg_softmax":
                    outputs_aggregated.div_(num_of_mc_iters)
                    max_vals, predicted = torch.max(outputs_aggregated, dim=1)
                    # For each value range in max_vals, calculate accuracy
                    correct_tags = (predicted == labels.data).sum()
                    acc_avg.add((float(correct_tags) / labels.size(0)) * 100, labels.size(0))

            # Restore running statistics for BatchNorm layers
            if self.save_bn_runnings and training:
                load_bn_running(self.net, runnings)

            set_size += labels.size(0)
            # Take a step
            if training:
                self.optimizer.step()

            # Print statistics
            if verbose_freq and verbose_freq > 0 and (i % verbose_freq) == (verbose_freq - 1):
                self.logger.info("Epoch " + str(self.epochs_trained + 1) + ", " + fwd_name + " set, " +
                                 "Iter " + str(i + 1) +
                                 " current average loss " + str(loss_avg.avg) +
                                 " current average acc " + str(acc_avg.avg) + "%")
            # Record total minibatch time
            self.total_minibatch_time_avg[fwd_name].add(self.total_minibatch_time.get_time()
                                                        ) if self.total_minibatch_time is not None else None

        self.total_pass_time_avg[fwd_name].add(self.total_pass_time.get_time()
                                               ) if self.total_pass_time is not None else None
        if training:
            self.epochs_trained += 1

        if verbose_freq is not None:
            fwd_desc = fwd_name
            if not training:
                fwd_desc += " [" + inference_method + "]"
            self.logger.info("Stats for " + fwd_desc + " set of size " + str(set_size) + ", " +
                             "loss is " + str(loss_avg.avg) + ", " +
                             "acc is " + str(acc_avg.avg) + "%")

        return loss_avg.avg, acc_avg.avg

    # Print functions
    def print_num_of_params(self):
        num_of_params = 0
        for parameter in self.net.parameters():
            num_of_params += parameter.numel()
        self.logger.info("Number of parameters in the model is " + str("{:,}".format(num_of_params)))

    def print_layers(self):
        for l, (name, parameter) in enumerate(self.net.named_parameters()):
            self.logger.info("layer " + str(l) + " is " + name + " with number of parameters " +
                             str("{:,}".format(parameter.numel())) + " and norm " + str(parameter.data.cpu().norm()))

    def print_criterion_params(self):
        self.logger.info("Criterion parameters: type=" + str(type(self.criterion)))

    def print_num_of_net_params(self):
        num_of_params = 0
        for parameter in self.net.parameters():
            num_of_params += parameter.numel()
        self.logger.info("Number of parameters in the model is " + str("{:,}".format(num_of_params)))
