import time

import numpy as np
import torch
from torch.nn import functional as F
from utils import count_acc


class Logger:
    def __init__(self, log_interval, args, verbose=True):

        self.log_interval = log_interval
        self.args = args
        self.verbose = verbose
        self.scale = 16 if args.k_shot == 1 else 2

        # highest accuracy observed on training and validation set, and on average
        self.best_train_accuracy = 0
        self.best_valid_accuracy = 0
        self.best_train_valid_accuracy = 0

        # accuracies (train/valid) we see for the best models
        self.best_model_train_stats = [0, 0]
        self.best_model_valid_stats = [0, 0]
        self.best_model_train_valid_stats = [0, 0]

        # print some infos
        if verbose:
            print(
                'n-way: ', args.n_way, ', k-shot: ', args.k_shot,
                '\nseed:', args.seed,
                '\nnum_context_params:', args.num_context_params,
                '\ncontext_in:', args.context_in,
                '\ncontext_in_type:', args.context_in_type,
                '\nlr[in]:', args.lr_inner,
                ', [out]: (embed){}-(critic){}'.format(args.lr_meta_embed, args.lr_meta_critic),
                '\ngrad[in]:', args.num_grad_steps_inner, ', [out]:', args.num_grad_steps_eval,
                '\nbatchs:', args.tasks_per_metaupdate
            )

        # initialise dictionary to keep track of accuracies/losses
        # - for training tasks
        self.training_stats = {
            #
            'test_accuracy_pre_update': [],
            'test_accuracy_post_update': [],
            #
            'test_loss_pre_update': [],
            'test_loss_post_update': []
        }
        # - for validation tasks
        self.validation_stats = {
            #
            'test_accuracy_pre_update': [],
            'test_accuracy_post_update': [],
            #
            'test_loss_pre_update': [],
            'test_loss_post_update': []
        }

        # keep track of how long the experiment takes
        self.start_time = time.time()

    def prepare_inner_loop(self, iter_counter, mode='train'):
        """
        Called before iterating over tasks in the inner loop
        :return:
        """
        if iter_counter % self.log_interval == 0:

            if mode == 'train':
                for key in self.training_stats.keys():
                    self.training_stats[key].append([])
            elif mode == 'valid':
                for key in self.validation_stats.keys():
                    self.validation_stats[key].append([])
            else:
                raise NotImplementedError()

    def log_pre_update(self, iter_counter, support_x, query_x, query_y, model, mode='train'):

        if iter_counter % self.log_interval == 0:
            if mode == 'train':
                loss, acc = self.get_loss_accuracy(support_x, query_x, query_y, model)
                self.training_stats['test_accuracy_pre_update'][-1].append(acc)
                self.training_stats['test_loss_pre_update'][-1].append(loss)
            elif mode == 'valid':
                loss, acc = self.get_loss_accuracy(support_x, query_x, query_y, model)
                self.validation_stats['test_accuracy_pre_update'][-1].append(acc)
                self.validation_stats['test_loss_pre_update'][-1].append(loss)
            else:
                raise NotImplementedError()

    def log_post_update(self, iter_counter, support_x, query_x, query_y, model, mode='train'):

        if iter_counter % self.log_interval == 0:
            if mode == 'train':
                loss, acc = self.get_loss_accuracy(support_x, query_x, query_y, model)
                self.training_stats['test_accuracy_post_update'][-1].append(acc)
                self.training_stats['test_loss_post_update'][-1].append(loss)
            elif mode == 'valid':
                loss, acc = self.get_loss_accuracy(support_x, query_x, query_y, model)
                self.validation_stats['test_accuracy_post_update'][-1].append(acc)
                self.validation_stats['test_loss_post_update'][-1].append(loss)
            else:
                raise NotImplementedError()

    def summarise_inner_loop(self, mode):
        if mode == 'train':
            for key in self.training_stats.keys():
                self.training_stats[key][-1] = np.mean(self.training_stats[key][-1])
        if mode == 'valid':
            for key in self.validation_stats.keys():
                self.validation_stats[key][-1] = np.mean(self.validation_stats[key][-1])

    def update_best_model(self, model, critic_net, save_path):

        # get the current training and validation accuracy
        train_acc = self.training_stats['test_accuracy_post_update'][-1]
        valid_acc = self.validation_stats['test_accuracy_post_update'][-1]
        train_valid_acc = 0.5 * (train_acc + valid_acc)

        if train_acc > self.best_train_accuracy:

            self.best_train_accuracy = train_acc
            # log what the corresponding accuracy on training and validation set are
            self.best_model_train_stats = [train_acc, valid_acc]

            if save_path is not None:
                np.save(save_path + '_best_train', self.best_model_train_stats)
                torch.save(model.state_dict(), save_path + '_best_train_model.pth')
                torch.save(critic_net.state_dict(), save_path + '_best_train_critic_model.pth')

        if valid_acc > self.best_valid_accuracy:

            self.best_valid_accuracy = valid_acc
            self.best_model_valid_stats = [train_acc, valid_acc]

            if save_path is not None:
                np.save(save_path + '_best_valid', self.best_model_valid_stats)
                torch.save(model.state_dict(), save_path + '_best_valid_model.pth')
                torch.save(critic_net.state_dict(), save_path + '_best_valid_critic_model.pth')

        if train_valid_acc > self.best_train_valid_accuracy:

            self.best_train_valid_accuracy = train_valid_acc
            self.best_model_train_valid_stats = [train_acc, valid_acc]

            if save_path is not None:
                np.save(save_path + '_best_train_valid', self.best_model_train_valid_stats)
                torch.save(model.state_dict(), save_path + '_best_train_valid_model.pth')
                torch.save(critic_net.state_dict(), save_path + '_best_train_valid_critic_model.pth')

    def print(self, iter_counter, grad_inner, grad_meta_e, grad_meta_c):
        if self.verbose and (iter_counter % self.log_interval == 0):
            self.print_logs(iter_counter, grad_inner, grad_meta_e, grad_meta_c)

    def print_header(self):
        if self.verbose:
            print('||------||----------- TRAINING ------------||----------- VALIDATION ----------||-----------------------------||---------||')
            print('||------||------ LOSS ------|----- ACC ----||------ LOSS ------|----- ACC ----||----------- GRAD ------------||---------||')
            print('|| iter ||    query set     |   query set  ||    query set     |   query set  ||  inner  |  meta_E |  meta_C ||   time  ||')
            print('||------||------------------|--------------||------------------|--------------||---------|---------|---------||---------||')

    def print_logs(self, iter_counter, grad_inner, grad_meta_e, grad_meta_c):
        if self.verbose:
            print('||{:5} || {:<7}->{:<7} | {:<5}->{:<5} || {:<7}->{:<7} | {:<5}->{:<5} || {:<7} | {:<7} | {:<7} || {:<7} ||'.format(
                    iter_counter,
                    # meta-train, task-test
                    np.round(self.training_stats['test_loss_pre_update'][-1], 3),
                    np.round(self.training_stats['test_loss_post_update'][-1], 3),
                    np.round(self.training_stats['test_accuracy_pre_update'][-1], 3),
                    np.round(self.training_stats['test_accuracy_post_update'][-1], 3),
                    # meta-valid, task-test
                    np.round(self.validation_stats['test_loss_pre_update'][-1], 3),
                    np.round(self.validation_stats['test_loss_post_update'][-1], 3),
                    np.round(self.validation_stats['test_accuracy_pre_update'][-1], 3),
                    np.round(self.validation_stats['test_accuracy_post_update'][-1], 3),
                    # gradients
                    np.round(grad_inner[0].abs().mean().item(), 4),
                    np.round(grad_meta_e[0].abs().mean().item(), 4),
                    np.round((grad_meta_c[0]).abs().mean().item(), 5),
                    # time
                    np.round((time.time() - self.start_time) / 60, 2)
                ))

    def get_loss_accuracy(self, support, query, label, model):
        logits = model.forward_pred(support, query, self.args.n_way, self.args.k_shot)
        loss = F.cross_entropy(logits/self.scale, label).item()
        acc = count_acc(logits, label)
        return loss, acc

