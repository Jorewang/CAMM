import argparse

import torch


def parse_args():
    parser = argparse.ArgumentParser(description='CAEM')

    parser.add_argument('--n_iter', type=int, default=30000, help='number of meta-iterations')
    parser.add_argument('--seed', type=int, default=95)

    parser.add_argument('--tasks_per_metaupdate', type=int, default=2,
                        help='number of tasks in each batch per meta-update')

    parser.add_argument('--n_way', type=int, default=5, help='number of object classes to learn')
    parser.add_argument('--k_shot', type=int, default=5, help='number of examples per class to learn from')
    parser.add_argument('--k_query', type=int, default=15, help='number of examples to evaluate on (in outer loop)')

    parser.add_argument('--lr_inner', type=float, default=0.1, help='inner-loop learning rate (per task)')
    parser.add_argument('--lr_meta_embed', type=float, default=0.001,
                        help='outer-loop Embedding model learning rate (used with Adam optimiser)')
    parser.add_argument('--lr_meta_critic', type=float, default=0.001,
                        help='outer-loop Critic model learning rate (used with Adam optimiser)')
    parser.add_argument('--lr_meta_embed_decay', type=float, default=0.9,
                        help='decay factor for meta learning rate of Embedding model')
    parser.add_argument('--lr_meta_critic_decay', type=float, default=0.9,
                        help='decay factor for meta learning rate of Critic model')

    parser.add_argument('--num_grad_steps_inner', type=int, default=2,
                        help='number of gradient steps in inner loop (during training)')
    parser.add_argument('--num_grad_steps_eval', type=int, default=2,
                        help='number of gradient updates at test time (for evaluation)')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'Sgd'])

    # network architecture
    parser.add_argument('--num_context_params', type=int, default=100, help='number of context parameters')
    parser.add_argument('--context_in', nargs='+', default=[True, True, True, True],
                        help='per layer, indicate if context params are added')
    parser.add_argument('--context_in_type', nargs='+', default=['mix', 'mix', 'mix', 'mix'],
                        help='only relation to context params or mixed')

    parser.add_argument('--imsize', type=int, default=84, help='downscale images to this size')
    parser.add_argument('--num_filters', type=int, default=64, help='number of filters per conv-layer')
    parser.add_argument('--out_dim', type=int, default=64, help='number of filters last conv-layer')
    parser.add_argument('--num_film_hidden_layers', type=int, default=1, help='mumber of hidden layers used for FiLM')

    parser.add_argument('--data_path', type=str, default='../data', help='folder which contains image data')
    parser.add_argument('--rerun', action='store_true', default=False,
                        help='Re-run experiment (will override previously saved results)')
    parser.add_argument('--load_pre', default=True, help='whether to load pre_trained model')
    # choice
    # parser.add_argument('--model_type', type=str, default='ConvNet', choices=['ConvNet', 'ResNet'])
    args = parser.parse_args()

    # use the GPU if available
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return args
