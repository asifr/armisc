import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def linear(input, weight, bias=None):
    if input.dim() == 2 and bias is not None:
        # fused op is marginally faster
        ret = torch.addmm(bias, input, weight.t())
    else:
        output = input.matmul(weight.t())
        if bias is not None:
            output += bias
        ret = output
    return ret


class FeatureRegression(nn.Module):
    """Feature regression: sum(w[i] * x[i])"""

    def __init__(self, input_size, output_size=1):
        super(FeatureRegression, self).__init__()
        self.weight = Parameter(torch.Tensor(output_size, input_size))
        nn.init.xavier_normal_(self.weight)

    def forward(self, inputs):
        return linear(inputs, self.weight)


class TemporalDecayRegression(nn.Module):
    """Temporal decay regression exp(-relu(sum(w[i] * x[i])))"""

    def __init__(self, input_size, output_size=1, interactions=False):
        super(TemporalDecayRegression, self).__init__()
        self.interactions = interactions
        if interactions:
            self.linear = nn.Linear(input_size, output_size)
        else:
            self.weight = Parameter(torch.Tensor(output_size, input_size))
            nn.init.xavier_normal_(self.weight)

    def forward(self, inputs):
        if self.interactions:
            w = self.linear(inputs)
        else:
            w = linear(inputs, self.weight)
        gamma = torch.exp(-F.relu(w))
        return gamma


class ImputerRegression(nn.Module):
    """Estimate variable from other features"""

    def __init__(self, input_size, bias=True):
        super(ImputerRegression, self).__init__()
        self.bias = bias
        self.W = Parameter(torch.Tensor(input_size, input_size))
        nn.init.xavier_uniform_(self.W)
        stdv = 1.0 / math.sqrt(self.W.size(0))
        if self.bias:
            self.b = Parameter(torch.Tensor(input_size))
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        torch.diagonal(self.W).fill_(0)
        if self.bias:
            return F.linear(x, self.W, self.b)
        else:
            return F.linear(x, self.W)


class FeatureEmbedding(nn.Module):
    """Regression layer with temporal decay."""

    def __init__(self, input_size, output_size=1, interactions=False):
        super(FeatureEmbedding, self).__init__()
        if interactions:
            self.feature_reg = nn.Linear(input_size, output_size)
        else:
            self.feature_reg = FeatureRegression(input_size, output_size)
        self.temporal_decay = TemporalDecayRegression(
            input_size, output_size, interactions=interactions
        )

    def forward(self, inputs, deltas):
        """input size: [batch_size,features] or [batch_size,timesteps,features]"""
        # feature transformation
        x = self.feature_reg(inputs)
        # decay rate
        gamma = self.temporal_decay(deltas)
        # complement
        xc = x * gamma  # [B, T, D]
        return xc


class FeatureTransformer(nn.Module):
    """Regression layer with temporal decay and imputation."""

    def __init__(self, input_size, interactions=True):
        super(FeatureTransformer, self).__init__()
        if interactions:
            self.feature_reg = nn.Linear(input_size, input_size)
        else:
            self.feature_reg = FeatureRegression(input_size, input_size)
        self.temporal_decay = TemporalDecayRegression(
            input_size, input_size, interactions=interactions
        )
        self.beta_reg = nn.Linear(input_size * 2, input_size)
        self.impute_reg = ImputerRegression(input_size)

    def forward(self, inputs, deltas, masks):
        """input size: [batch_size,features] or [batch_size,timesteps,features]"""
        # decay rate
        gamma = self.temporal_decay(deltas)
        # feature transformation
        x = self.feature_reg(inputs)
        # weight transformed features by decay
        x = x * gamma
        # calculate the weight of imputed and transformed variables
        beta = torch.sigmoid(self.beta_reg(torch.cat([gamma, masks], 1)))
        # impute variables from other variables
        imp = self.impute_reg(x)
        z = beta * x + (1 - beta) * imp
        # complement
        c = masks * x + (1 - masks) * z
        return c
