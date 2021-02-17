import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_glu(module, input_dim, output_dim):
    gain_value = np.sqrt((input_dim + output_dim) / np.sqrt(input_dim))
    torch.nn.init.xavier_normal_(module.weight, gain=gain_value)
    return


class CDFLayer(nn.Module):
    def __init__(self, device="cpu"):
        super(CDFLayer, self).__init__()
        self.loc_scale = Parameter(torch.FloatTensor([0.0, 1.0]).to(device))

    def forward(self, x, dim=1):
        m = torch.distributions.Cauchy(self.loc_scale[0], self.loc_scale[1])
        return m.cdf(torch.cumsum(x, dim))


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device="cpu"):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        h = self.net(x)  # [B, D]
        return h


class GLULayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GLULayer, self).__init__()

        self.output_dim = output_dim
        self.fc = nn.Linear(input_dim, 2 * output_dim, bias=False)
        initialize_glu(self.fc, input_dim, 2 * output_dim)
        self.bn = nn.BatchNorm1d(2 * output_dim)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        out = torch.mul(x[:, : self.output_dim], torch.sigmoid(x[:, self.output_dim :]))
        return out
