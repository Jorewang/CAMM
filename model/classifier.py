import sys
sys.path.append('../')
import torch.nn as nn
import torch.nn.functional as F
from utils import euclidean_dist


class ClassifierNet(nn.Module):
    def __init__(self,
                 hid_dim,
                 z_dim):
        super(ClassifierNet, self).__init__()
        self.embedding_layer_dict = nn.ModuleDict()
        self.cnn_layer_size = 4

        in_channels = 3
        out_channels = hid_dim
        for i in range(self.cnn_layer_size):
            if i == self.cnn_layer_size - 1:
                out_channels = z_dim
            self.embedding_layer_dict['conv{}'.format(i)] = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.embedding_layer_dict['bn{}'.format(i)] = nn.BatchNorm2d(out_channels, track_running_stats=False)
            in_channels = out_channels

        # last classifier
        self.fc = nn.Linear(z_dim * 25, 64)

    def forward(self, x, is_emb=False):
        out = x
        for i in range(self.cnn_layer_size):
            out = self.embedding_layer_dict['conv{}'.format(i)](out)
            out = self.embedding_layer_dict['bn{}'.format(i)](out)
            out = F.relu(out)
            out = F.max_pool2d(out, 2)

        # flatting
        out = out.view(out.size(0), -1)
        if not is_emb:
            out = self.fc(out)
        return out

    def forward_proto(self, data_shot, data_query, n_way, k_shot):
        proto = self.forward(data_shot, is_emb=True)
        proto = proto.view(n_way, k_shot, -1).mean(dim=1)

        query = self.forward(data_query, is_emb=True)
        logits = euclidean_dist(query, proto)
        return logits


if __name__ == '__main__':
    import torch
    x = torch.rand(3, 3, 84, 84)
    m = ClassifierNet(64, 64)
    print(m(x).size())
