import sys
sys.path.append('../')
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import euclidean_dist


class ProtoNet(nn.Module):
    def __init__(self,
                 hid_dim,
                 z_dim,
                 num_context_params,
                 context_in,
                 context_in_type,
                 num_film_hidden_layers,
                 device):
        super(ProtoNet, self).__init__()
        self.context_in = context_in
        self.context_in_type = context_in_type
        self.num_film_hidden_layers = num_film_hidden_layers
        self.hid_dim = hid_dim
        self.z_dim = z_dim
        self.cnn_layer_size = 4

        self.embedding_layer_dict = nn.ModuleDict()
        self.film_layer_dict = nn.ModuleDict()
        self.premul_dict = nn.ParameterDict()

        # context_params
        self.context_params = torch.zeros(size=[num_context_params], requires_grad=True).to(device)

        # conv bn
        in_channels = 3
        out_channels = hid_dim
        for i in range(self.cnn_layer_size):
            if i == self.cnn_layer_size - 1:
                out_channels = z_dim
            self.embedding_layer_dict['conv{}'.format(i)] = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.embedding_layer_dict['bn{}'.format(i)] = nn.BatchNorm2d(out_channels, track_running_stats=False)
            in_channels = out_channels

        # film layer
        channels = hid_dim
        for i in range(self.cnn_layer_size):
            if i == self.cnn_layer_size - 1:
                channels = z_dim
            if context_in[i]:
                if num_film_hidden_layers == 1:
                    if self.context_in_type[i] == 'only':
                        self.film_layer_dict['film{}'.format(i)] = nn.Linear(num_context_params, channels * 2 // 8)
                    else:
                        self.film_layer_dict['film{}'.format(i)] = nn.Linear(num_context_params + channels, channels*2//8)
                    self.film_layer_dict['film{}{}'.format(i, i)] = nn.Linear(channels * 2 // 8, channels * 2)
                else:
                    if self.context_in_type[i] == 'only':
                        self.film_layer_dict['film{}'.format(i)] = nn.Linear(num_context_params, channels * 2)
                    else:
                        self.film_layer_dict['film{}'.format(i)] = nn.Linear(num_context_params + channels, channels * 2)

        for i in range(self.cnn_layer_size):
            if context_in[i]:
                self.premul_dict['gamma{}'.format(i)] = nn.Parameter(torch.FloatTensor(1).fill_(0.0),
                                                                     requires_grad=True)
                self.premul_dict['beta{}'.format(i)] = nn.Parameter(torch.FloatTensor(1).fill_(0.0),
                                                                    requires_grad=True)

    def reset_context_params(self):
        self.context_params = self.context_params.detach() * 0
        self.context_params.requires_grad = True

    def forward(self, x):
        out = x
        channels = self.hid_dim
        for i in range(self.cnn_layer_size):
            if i == self.cnn_layer_size - 1:
                channels = self.z_dim
            out = self.embedding_layer_dict['conv{}'.format(i)](out)
            out = self.embedding_layer_dict['bn{}'.format(i)](out)
            if self.context_in[i]:
                # FiLM it: forward through film layer to get scale and shift parameter
                if self.context_in_type[i] == 'only':
                    film = self.film_layer_dict['film{}'.format(i)](self.context_params)
                    if self.num_film_hidden_layers == 1:
                        film = self.film_layer_dict['film{}{}'.format(i, i)](F.relu(film))
                    gamma = film[:channels].view(1, -1, 1, 1)
                    beta = film[channels:].view(1, -1, 1, 1)
                else:
                    film = F.avg_pool2d(out, out.size(2))
                    film = film.view(film.size(0), -1)
                    film = self.film_layer_dict['film{}'.format(i)](torch.cat((film, self.context_params.expand(film.size(0), -1)), dim=1))
                    if self.num_film_hidden_layers == 1:
                        film = self.film_layer_dict['film{}{}'.format(i, i)](F.relu(film))
                    gamma = film[:, :channels].view(film.size(0), -1, 1, 1)
                    beta = film[:, channels:].view(film.size(0), -1, 1, 1)
                # transform feature map
                out = (self.premul_dict['gamma{}'.format(i)]*gamma+1.0)*out + self.premul_dict['beta{}'.format(i)]*beta
                # out = (gamma + 1.0) * out + beta
            out = F.relu(out)
            out = F.max_pool2d(out, 2)

        return out.view(out.size(0), -1)

    def forward_pred(self, data_shot, data_query, n_way, k_shot):
        proto = self.forward(data_shot)
        proto = proto.view(n_way, k_shot, -1).mean(dim=1)

        query = self.forward(data_query)
        logits = euclidean_dist(query, proto)
        return logits


if __name__ == '__main__':
    pro = ProtoNet(64, 64, 100,
                   [True, True, True, True],
                   ['only', 'only', 'only', 'only'], 0, torch.device('cpu'))
    x = torch.rand(5, 3, 84, 84)
    print(pro)
