"""
This script evaluates a saved model (will crash if nothing is saved).
"""
import os
import time

import arguments
import numpy as np
import scipy.stats as st
import torch
import utils
from dataloader.dataset_miniimagenet import MiniImagenet
from torch.utils.data import DataLoader
from model.CriticNetwork import CriticNetwork
from model.model import ProtoNet


def evaluate(args, model, critic_model, logger, dataloader, mode, num_grad_steps):
    for c, batch in enumerate(dataloader):

        support_x = batch[0].to(args.device)
        query_x = batch[1].to(args.device)
        query_y = batch[2].to(args.device)

        for inner_batch_idx in range(support_x.shape[0]):

            # reset context parameters
            model.reset_context_params()

            logger.log_pre_update(support_x[inner_batch_idx],
                                  query_x[inner_batch_idx], query_y[inner_batch_idx],
                                  model, mode)

            for _ in range(num_grad_steps):
                # forward train data through net
                support_encoded = model(support_x[inner_batch_idx])

                # compute loss
                loss_inner = critic_model(support_encoded, args.n_way, args.k_shot)

                # compute gradient for context parameters
                grad_inner = torch.autograd.grad(loss_inner, model.context_params)[0]

                # set context parameters to their updated values
                model.context_params = model.context_params - args.lr_inner * grad_inner

            logger.log_post_update(support_x[inner_batch_idx],
                                   query_x[inner_batch_idx], query_y[inner_batch_idx],
                                   model, mode)

    # reset context parameters
    model.reset_context_params()


class Logger:
    def __init__(self, args):

        self.args = args

        # initialise dictionary to keep track of accuracies/losses
        self.train_stats = {
            'test_accuracy_pre_update': [],
            'test_accuracy_post_update': [],
        }
        self.valid_stats = {
            'test_accuracy_pre_update': [],
            'test_accuracy_post_update': [],
        }
        self.test_stats = {
            'test_accuracy_pre_update': [],
            'test_accuracy_post_update': [],
        }

        # keep track of how long the experiment takes
        self.start_time = time.time()

    def log_pre_update(self, support_x, query_x, query_y, model, mode):
        if mode == 'train':
            self.train_stats['test_accuracy_pre_update'].append(self.get_accuracy(support_x, query_x, query_y, model))
        elif mode == 'val':
            self.valid_stats['test_accuracy_pre_update'].append(self.get_accuracy(support_x, query_x, query_y, model))
        elif mode == 'test':
            self.test_stats['test_accuracy_pre_update'].append(self.get_accuracy(support_x, query_x, query_y, model))

    def log_post_update(self, support_x, query_x, query_y, model, mode):
        if mode == 'train':
            self.train_stats['test_accuracy_post_update'].append(self.get_accuracy(support_x, query_x, query_y, model))
        elif mode == 'val':
            self.valid_stats['test_accuracy_post_update'].append(self.get_accuracy(support_x, query_x, query_y, model))
        elif mode == 'test':
            self.test_stats['test_accuracy_post_update'].append(self.get_accuracy(support_x, query_x, query_y, model))

    @staticmethod
    def print_header():
        print('||--------------------------------------------------------------------------------------------------------||')
        print('||------------- TRAINING ------------------------|---------------- EVALUATION ----------------------------||')
        print('||-----------------------------------------------|--------------------------------------------------------||')
        print('||-----------------|     observed performance    |  META_TRAIN  |  META_VALID  |       META_TEST          ||')
        print('||    selection    |-----------------------------|--------------|--------------|--------------------------||')
        print('||    criterion    |    train     |     valid    |     test     |     test     |          test            ||')
        print('||-----------------|--------------|--------------|--------------|--------------|--------------------------||')

    def print_logs(self, selection_criterion, logged_perf=None):
        if logged_perf is None:
            logged_perf = [' ', ' ']
        else:
            logged_perf = [np.round(logged_perf[0], 3), np.round(logged_perf[1], 3)]

        avg_acc = np.mean(self.test_stats['test_accuracy_post_update'])
        conf_interval = st.t.interval(0.95,
                                      len(self.test_stats['test_accuracy_post_update']) - 1,
                                      loc=avg_acc,
                                      scale=st.sem(self.test_stats['test_accuracy_post_update']))

        print('||   {:<11}   |    {:<5}     |     {:<5}    | {:<5}->{:<5} | {:<5}->{:<5} | {:<5}->{:<5} (+/- {}) ||'.format(
                selection_criterion,
                # performance we observed during training
                logged_perf[0],
                logged_perf[1],
                # meta-valid, task-test
                np.round(np.mean(self.train_stats['test_accuracy_pre_update']), 3),
                np.round(np.mean(self.train_stats['test_accuracy_post_update']), 3),
                #
                np.round(np.mean(self.valid_stats['test_accuracy_pre_update']), 3),
                np.round(np.mean(self.valid_stats['test_accuracy_post_update']), 3),
                #
                np.round(np.mean(self.test_stats['test_accuracy_pre_update']), 3),
                np.round(100 * avg_acc, 3),
                #
                np.round(100 * np.mean(np.abs(avg_acc - conf_interval)), 2),
            ))

    def get_accuracy(self, support, query, label, model):
        # 5-way 5-shot test
        logits = model.forward_pred(support, query, args.n_way, args.k_shot)
        acc = utils.count_acc(logits, label)
        return acc


if __name__ == '__main__':

    args = arguments.parse_args()

    # --- settings ---
    args.seed = 39
    # utils.set_seed(args.seed)
    args.context_in_type = ['mix', 'mix', 'mix', 'mix']
    args.k_shot = 5
    args.lr_inner = 0.1
    args.num_grad_steps_inner = 2
    args.num_grad_steps_eval = 2
    args.num_context_params = 100

    if args.k_shot == 1:
        args.tasks_per_metaupdate = 2
        in_channel = 5
    else:
        args.tasks_per_metaupdate = 2
        in_channel = 8

    path = os.path.join(utils.get_base_path(), 'result_files', utils.get_path_from_args(args))

    try:
        training_stats, validation_stats = np.load(path + '.npy', allow_pickle=True)
        print('load path:[{}]'.format(path))
    except FileNotFoundError:
        print('You need to run the experiments first and make sure the results are saved at {}'.format(path))
        raise FileNotFoundError

    Logger.print_header()

    for num_grad_steps in [2]:

        print('\n --- ', num_grad_steps, '--- \n')

        # initialise logger
        logger = Logger(args)

        for selection_criterion in ['valid', 'train', 'train_valid']:

            logger = Logger(args)

            for dataset in ['train', 'val', 'test']:
                # load model and its performance during training
                model = ProtoNet(args.num_filters,
                                 args.out_dim,
                                 args.num_context_params,
                                 args.context_in,
                                 args.context_in_type,
                                 args.num_film_hidden_layers,
                                 args.device).to(args.device)
                model.load_state_dict(torch.load(path + '_best_{}_model.pth'.format(selection_criterion),
                                                 map_location='cpu' if not torch.cuda.is_available() else 'cuda:0'))

                critic_model = CriticNetwork(in_channel).to(args.device)
                critic_model.load_state_dict(torch.load(path + '_best_{}_critic_model.pth'.format(selection_criterion),
                                                        map_location='cpu' if not torch.cuda.is_available() else 'cuda:0'))

                best_performances = np.load(path + '_best_{}.npy'.format(selection_criterion))

                # initialise dataloader
                mini_test = MiniImagenet(mode=dataset, n_way=args.n_way,
                                         k_shot=args.k_shot, k_query=args.k_query,
                                         batchsz=1000, verbose=False, imsize=args.imsize,
                                         data_path=args.data_path)
                db_test = DataLoader(mini_test, batch_size=16, shuffle=True, num_workers=8, pin_memory=True)

                # evaluate the model
                evaluate(args, model, critic_model, logger, db_test, mode=dataset, num_grad_steps=num_grad_steps)

            logger.print_logs(selection_criterion, best_performances)
