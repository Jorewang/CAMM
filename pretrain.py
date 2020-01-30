import os
import torch
import torch.nn.functional as F
from model.classifier import ClassifierNet
from dataloader.dataset_miniimagenet import MiniImagenet
from dataloader.pretrain_dataset import Dataset64
from torch.utils.data import DataLoader
from utils import count_acc, Averager, get_base_path
import argparse


def print_header():
    print('||------||-------TRAINING-----||--------VALIDATION---------||')
    print('|| iter ||---LOSS---|---ACC---||---CUR ACC---|---MAX ACC---||')
    print('||------||----------|---------||-------------|-------------||')


def print_logs(iter, loss, acc, cur_acc, max_acc, train=True):
    if train:
        print('||{:^6}||{:^10.3f}|{:^9.3f}||{:^13}|{:^13.3f}||'.format(iter, loss, acc, 'None', max_acc))
    else:
        print('||{:^6}||{:^10}|{:^9}||{:^13.3f}|{:^13.3f}||'.format(iter, 'None', 'None', cur_acc, max_acc))


def pre_train(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ClassifierNet(hid_dim=args.num_filters, z_dim=args.out_dim).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, 5000, args.lr_decay)

    iter_counter = 0
    max_val_acc = 0
    model.train()
    while iter_counter <= args.n_iter:
        data_train = Dataset64(imsize=84, data_path=args.data_path)
        loader_train = DataLoader(dataset=data_train, batch_size=args.batch_size,
                                  shuffle=True, num_workers=8, pin_memory=True)

        data_valid = MiniImagenet(mode='val', n_way=args.n_way, k_shot=args.k_shot, k_query=args.k_query,
                                  batchsz=200, imsize=84, data_path=args.data_path, verbose=True)

        loader_valid = DataLoader(data_valid, batch_size=4, shuffle=True, num_workers=8, pin_memory=True)
        print_header()
        for step, batch in enumerate(loader_train):
            iter_counter += 1
            if iter_counter > args.n_iter:
                break

            data = batch[0].to(device)
            label = batch[1].to(device)

            logits = model(data)
            loss = criterion(logits, label)
            acc = count_acc(logits, label)

            if iter_counter % 100 == 0:
                print_logs(iter_counter, loss.item(), acc, 0, max_val_acc)

            optim.zero_grad()
            loss.backward()
            optim.step()
            scheduler.step()

            if iter_counter % 300 == 0:
                model.eval()
                va = Averager()
                with torch.no_grad():
                    for i, task_batch in enumerate(loader_valid):
                        support_x = task_batch[0].to(device)
                        query_x = task_batch[1].to(device)
                        query_y = task_batch[2].to(device)
                        for inner_batch_idx in range(support_x.shape[0]):
                            predictions = model.forward_proto(support_x[inner_batch_idx], query_x[inner_batch_idx],
                                                              args.n_way, args.k_shot)
                            acc = count_acc(predictions, query_y[inner_batch_idx])
                            va.add(acc)
                    val_acc = va.item()
                    if val_acc > max_val_acc:
                        torch.save(model.state_dict(),
                                   './pre_train_file/pre_train-{}_{}.pth'.format(args.num_filters, args.out_dim))
                        max_val_acc = val_acc
                print_logs(iter_counter, 0, 0, val_acc, max_val_acc, False)
                model.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_iter', type=int, default=60000, help='number of meta-iterations')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_decay', type=float, default=0.5)
    parser.add_argument('--data_path', type=str, default='../data', help='folder which contains image data')
    parser.add_argument('--n_way', type=int, default=16, help='test on val')
    parser.add_argument('--k_shot', type=int, default=1, help='test on val')
    parser.add_argument('--k_query', type=int, default=15, help='test on val')
    parser.add_argument('--num_filters', type=int, default=128, help='number of filters per conv-layer')
    parser.add_argument('--out_dim', type=int, default=64, help='number of filters last conv-layer')
    args = parser.parse_args()

    if not os.path.exists(os.path.join(get_base_path(), 'pre_train_file')):
        os.mkdir(os.path.join(get_base_path(), 'pre_train_file'))

    pre_train(args)
