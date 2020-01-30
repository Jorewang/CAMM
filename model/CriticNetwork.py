import torch
import torch.nn as nn


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels, track_running_stats=False),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2)
    )


class CriticNetwork(nn.Module):
    def __init__(self, in_channels, hid_dim=64):
        super(CriticNetwork, self).__init__()
        self.block = nn.Sequential(
            conv_block(in_channels, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
        )

        self.loss = nn.Linear(hid_dim * 25, 1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, support_embedding, n_way, k_shot):
        h = w = 40
        b, _ = support_embedding.size()
        proto = support_embedding.view(n_way, k_shot, -1).mean(dim=1)
        if k_shot == 1:
            support_emb_features = support_embedding.view(b, -1, h, w)
            emb_features_square = support_emb_features ** 2
            proto_mean_features = proto.mean(dim=0).view(-1, h, w).repeat(b, 1, 1, 1)
            mean_features_square = proto_mean_features ** 2
            var_features = proto.var(dim=0).view(-1, h, w).repeat(b, 1, 1, 1)
            x = torch.cat([support_emb_features,
                           emb_features_square,
                           proto_mean_features,
                           mean_features_square,
                           var_features], dim=1)
        else:
            support_emb_features = support_embedding.view(b, -1, h, w)
            emb_features_square = support_emb_features ** 2
            proto_emb_features = proto.unsqueeze(dim=1).repeat(1, k_shot, 1).view(b, -1, h, w)
            proto_emb_square = proto_emb_features ** 2
            proto_emb_var = support_embedding.view(n_way, k_shot, -1).var(dim=1).unsqueeze(dim=1).repeat(1, k_shot, 1).view(b, -1, h, w)
            proto_mean_features = proto.mean(dim=0).view(-1, h, w).repeat(b, 1, 1, 1)
            proto_mean_square = proto_mean_features ** 2
            proto_mean_var = proto.var(dim=0).view(-1, h, w).repeat(b, 1, 1, 1)
            x = torch.cat([support_emb_features,
                           emb_features_square,
                           proto_emb_features,
                           proto_emb_square,
                           proto_emb_var,
                           proto_mean_features,
                           proto_mean_square,
                           proto_mean_var], dim=1)

        x = self.block(x)
        x = x.view(x.shape[0], -1)
        x = self.loss(x)
        return x.mean()


if __name__ == '__main__':
    ipt = torch.randn(5, 8)
    print(ipt)
    a = ipt.unsqueeze(dim=1).repeat(1, 3, 1).view(15, -1, 2, 2)
    b = ipt.view(5, -1, 2, 2).unsqueeze(dim=1).repeat(1, 3, 1, 1, 1).view(15, -1, 2, 2)
    print(a == b)
