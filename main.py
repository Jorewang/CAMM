import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import utils
from model.CriticNetwork import CriticNetwork
from arguments import parse_args
from dataloader.dataset_miniimagenet import MiniImagenet
from logger import Logger
from model.model import ProtoNet
from torch.utils.data import DataLoader


def evaluate(iter_counter, args, model, logger, dataloader, lossC, save_path):
    logger.prepare_inner_loop(iter_counter, mode='valid')
    for i, batch in enumerate(dataloader):
        support_x = batch[0].to(args.device)
        query_x = batch[1].to(args.device)
        query_y = batch[2].to(args.device)

        for inner_batch_idx in range(support_x.shape[0]):

            # reset context parameters
            model.reset_context_params()

            logger.log_pre_update(iter_counter,
                                  support_x[inner_batch_idx],
                                  query_x[inner_batch_idx],
                                  query_y[inner_batch_idx],
                                  model, mode='valid')

            # -------------- inner update --------------
            for _ in range(args.num_grad_steps_eval):
                # forward train data through net
                support_encoded = model(support_x[inner_batch_idx])

                # compute loss
                loss_inner = lossC(support_encoded, args.n_way, args.k_shot)
                # compute gradient for context parameters
                grad_inner = torch.autograd.grad(loss_inner,
                                                 model.context_params)[0]
                # set context parameters to their updated values
                model.context_params = model.context_params - args.lr_inner * grad_inner

            # compute val acc after inner update
            logger.log_post_update(iter_counter,
                                   support_x[inner_batch_idx],
                                   query_x[inner_batch_idx], query_y[inner_batch_idx],
                                   model, mode='valid')

    # reset context parameters
    model.reset_context_params()

    # this will take the mean over the batches
    logger.summarise_inner_loop(mode='valid')

    # keep track of best models
    logger.update_best_model(model, lossC, save_path)


def run(args, log_interval=100, pre_train_path=None, save_path=None):
    args = args
    utils.set_seed(args.seed)

    scale = 16 if args.k_shot == 1 else 2
    in_channel = 5 if args.k_shot == 1 else 8

    model = ProtoNet(args.num_filters,
                     args.out_dim,
                     args.num_context_params,
                     args.context_in,
                     args.context_in_type,
                     args.num_film_hidden_layers,
                     args.device).to(args.device)
    if pre_train_path:
        # load pre_train model
        print('----------start-----------')
        print('Loading pre trained model!')
        model.load_state_dict(torch.load(pre_train_path, map_location=args.device), strict=False)

    # Critic model(need train)
    inner_loss_model = CriticNetwork(in_channel).to(args.device)
    # Cross Entropy Loss
    outer_loss = nn.CrossEntropyLoss().to(args.device)

    model.train()
    inner_loss_model.train()

    if args.optimizer == 'Adam':
        meta_critic_optimiser = torch.optim.Adam(inner_loss_model.parameters(), args.lr_meta_critic)
    else:
        meta_critic_optimiser = torch.optim.SGD(inner_loss_model.parameters(), args.lr_meta_critic,
                                                momentum=0.9, nesterov=True)
    if not pre_train_path:
        if args.optimizer == 'Adam':
            meta_optimiser = torch.optim.Adam(model.parameters(), args.lr_meta_embed)
        else:
            meta_optimiser = torch.optim.SGD(model.parameters(), lr=args.lr_meta_embed,
                                             momentum=0.9, nesterov=True)
    else:
        if args.optimizer == 'Adam':
            meta_optimiser = torch.optim.Adam([{'params': model.embedding_layer_dict.parameters(),
                                                'lr': args.lr_meta_embed * 0.1},
                                               {'params': model.film_layer_dict.parameters()},
                                               {'params': model.premul_dict.parameters()}], lr=args.lr_meta_embed)
        else:
            meta_optimiser = torch.optim.SGD([{'params': model.embedding_layer_dict.parameters(),
                                               'lr': args.lr_meta_embed * 0.1},
                                              {'params': model.film_layer_dict.parameters()},
                                              {'params': model.premul_dict.parameters()}], lr=args.lr_meta_embed,
                                             momentum=0.9, nesterov=True)

    critic_scheduler = torch.optim.lr_scheduler.StepLR(meta_critic_optimiser, 1000, args.lr_meta_critic_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(meta_optimiser, 1000, args.lr_meta_embed_decay)

    # init logger
    logger = Logger(log_interval, args)

    meta_grad_init = [0 for _ in range(len(model.state_dict()))]
    critic_meta_grad_init = [0 for _ in range(len(inner_loss_model.state_dict()))]

    iter_counter = 0
    while iter_counter <= args.n_iter:

        data_train = MiniImagenet(mode='train', n_way=args.n_way, k_shot=args.k_shot, k_query=args.k_query,
                                  batchsz=10000, imsize=args.imsize, data_path=args.data_path)
        loader_train = DataLoader(data_train, batch_size=args.tasks_per_metaupdate,
                                  shuffle=True, num_workers=2, pin_memory=True)

        # fixed 5-way, 5-shot when eval
        data_valid = MiniImagenet(mode='val', n_way=args.n_way, k_shot=args.k_shot, k_query=args.k_query,
                                  batchsz=500, imsize=args.imsize, data_path=args.data_path)
        loader_valid = DataLoader(data_valid, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)

        logger.print_header()

        for step, batch in enumerate(loader_train):
            support_x = batch[0].to(args.device)
            query_x = batch[1].to(args.device)
            query_y = batch[2].to(args.device)

            # skip batch if we don't have enough tasks in the current batch (might happen in last batch)
            if support_x.shape[0] != args.tasks_per_metaupdate:
                continue

            # initialise meta-gradient
            meta_grad = copy.deepcopy(meta_grad_init)
            critic_meta_grad = copy.deepcopy(critic_meta_grad_init)

            logger.prepare_inner_loop(iter_counter)

            for inner_batch_idx in range(args.tasks_per_metaupdate):
                # reset context parameters
                model.reset_context_params()

                # -------------- inner update --------------
                logger.log_pre_update(iter_counter,
                                      support_x[inner_batch_idx],
                                      query_x[inner_batch_idx],
                                      query_y[inner_batch_idx], model)

                for _ in range(args.num_grad_steps_inner):
                    # forward train data through net
                    support_encoded = model(support_x[inner_batch_idx])

                    # compute loss
                    task_loss_train = inner_loss_model(support_encoded,
                                                       args.n_way, args.k_shot)

                    # compute gradient for context parameters
                    task_grad_train = torch.autograd.grad(task_loss_train,
                                                          model.context_params,
                                                          create_graph=True)[0]

                    # set context parameters to their updated values
                    model.context_params = model.context_params - args.lr_inner * task_grad_train

                # -------------- get meta gradient --------------
                logits = model.forward_pred(support_x[inner_batch_idx],
                                            query_x[inner_batch_idx],
                                            args.n_way,
                                            args.k_shot)

                # compute loss on test data
                task_loss_test = outer_loss(logits/scale, query_y[inner_batch_idx])

                # compute gradient for shared parameters
                task_grad_test = torch.autograd.grad(task_loss_test, model.parameters(), retain_graph=True)
                critic_task_grad_test = torch.autograd.grad(task_loss_test, inner_loss_model.parameters())

                # add to meta-gradient
                for g in range(len(task_grad_test)):
                    meta_grad[g] += task_grad_test[g].detach()

                for i in range(len(critic_task_grad_test)):
                    critic_meta_grad[i] += critic_task_grad_test[i].detach()
                # -------------------------------------------------

                logger.log_post_update(iter_counter,
                                       support_x[inner_batch_idx],
                                       query_x[inner_batch_idx],
                                       query_y[inner_batch_idx], model)

            # reset context parameters
            model.reset_context_params()

            # summarise inner loop and get validation performance
            logger.summarise_inner_loop(mode='train')

            if iter_counter % log_interval == 0:
                # evaluate how good the current model is (*before* updating so we can compare better):
                evaluate(iter_counter, args, model, logger, loader_valid, inner_loss_model, save_path)
                if save_path is not None:
                    np.save(save_path, [logger.training_stats, logger.validation_stats])

            logger.print(iter_counter, task_grad_train, meta_grad, critic_meta_grad)
            iter_counter += 1
            if iter_counter > args.n_iter:
                break

            # --------------meta update-------------
            meta_optimiser.zero_grad()
            meta_critic_optimiser.zero_grad()

            # set gradients of parameters manually
            for c, param in enumerate(model.parameters()):
                param.grad = meta_grad[c] / float(args.tasks_per_metaupdate)
                param.grad.data.clamp_(-10, 10)

            for k, v in enumerate(inner_loss_model.parameters()):
                v.grad = critic_meta_grad[k] / float(args.tasks_per_metaupdate)
                v.grad.data.clamp_(-10, 10)

            # the meta-optimiser only operates on the shared parameters, not the context parameters
            meta_optimiser.step()
            meta_critic_optimiser.step()
            critic_scheduler.step()
            scheduler.step()
    model.reset_context_params()


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists(os.path.join(utils.get_base_path(), 'result_files')):
        os.mkdir(os.path.join(utils.get_base_path(), 'result_files'))
    if not os.path.exists(os.path.join(utils.get_base_path(), 'result_plots')):
        os.mkdir(os.path.join(utils.get_base_path(), 'result_plots'))

    if os.path.exists(os.path.join(utils.get_base_path(), 'pre_train_file', 'pre_train.pth')) and args.load_pre:
        pre_train_path = os.path.join(utils.get_base_path(), 'pre_train_file', 'pre_train.pth')
    else:
        pre_train_path = None

    path = os.path.join(utils.get_base_path(), 'result_files', utils.get_path_from_args(args))
    log_interval = 500
    if (not os.path.exists(path + '.npy')) or args.rerun:
        print('Starting experiment. Logging under filename {}'.format(path + '.npy'))
        run(args, log_interval=log_interval, pre_train_path=pre_train_path, save_path=path)
    else:
        print('Found results in {}. If you want to re-run, use the argument --rerun'.format(path))

    # ----------------plot-------------------
    plt.switch_backend('agg')
    training_stats, validation_stats = np.load(path + '.npy', allow_pickle=True)

    plt.figure(figsize=(12, 6))
    x_ticks = np.arange(1, log_interval * len(training_stats['test_accuracy_pre_update']), log_interval)

    # training set
    plt.subplot(1, 2, 1)
    p = plt.plot(x_ticks, training_stats['test_accuracy_pre_update'], '--', label='[test] pre-update')
    plt.plot(x_ticks, training_stats['test_accuracy_post_update'], linewidth=1,
             label='[test] post-update')
    plt.ylim([0, 1.01])
    plt.yticks(np.linspace(0.0, 1.0, 11))
    plt.xlim([0, args.n_iter])

    # validation set
    plt.subplot(1, 2, 2)
    p = plt.plot(x_ticks, validation_stats['test_accuracy_pre_update'], '--', label='[test] pre-update')
    plt.plot(x_ticks, validation_stats['test_accuracy_post_update'], linewidth=1,
             label='[test] post-update')
    plt.ylim([0, 1.01])
    plt.yticks(np.linspace(0.0, 1.0, 11))
    plt.xlim([0, args.n_iter])

    # Title
    # [train_n_way, train_k_shot],
    # [hid_dim, out_dim],
    # [lr_inner, lr_meta_embed, lr_meta_critic, lr_meta_decay_embed, lr_meta_decay_critic],
    # [num_context_params, hidden_layers, context_in, context_in_type],
    # [num_grad_inner, num_grad_eval]
    # [batch_tasks, iter, seed]
    title = '[{}-{}],[{}-{}],[{}-{}-{}-{}-{}],[{}-{}-{}-{}],' \
            '[{}-{}],[{}-{}-{}]'.format(args.n_way, args.k_shot,
                                        args.num_filters, args.out_dim,
                                        args.lr_inner, args.lr_meta_embed, args.lr_meta_critic,
                                        args.lr_meta_embed_decay, args.lr_meta_critic_decay,
                                        args.num_context_params, args.num_film_hidden_layers,
                                        args.context_in, args.context_in_type,
                                        args.num_grad_steps_inner, args.num_grad_steps_eval,
                                        args.tasks_per_metaupdate, args.n_iter, args.seed
                                        )
    plt.suptitle(title)
    plt.title(' ')
    plt.xlabel('num iter', fontsize=10)
    plt.ylabel('accuracy', fontsize=10)
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(utils.get_base_path(), 'result_plots', '{}'.format(title.replace('.', ''))))
    plt.close()
