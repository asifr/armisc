"""
PyTorch model trainer. The trainer class is generic and only handles the 
training and validation loop. The `training_step` and `validation_step`
should be defined as functions.

Example implementation:
=======================

from dataclasses import dataclass
from torch.utils.data import TensorDataset, DataLoader
from armisc import trainer

# options
@dataclass
class TrainerOpts(trainer.TrainerOpts):
    name: str = 'Model'
    input_size: int = None

# return a loss
def training_step(model, batch, opts, epoch, batch_idx):
    x,y = batch
    probs, m_loss = model(x)
    loss = F.binary_cross_entropy(probs,y)
    return loss

# return a metric, like AUC -- this metric is a loss so it should 
# be decreasing as the accuracy improves
def validation_step(model, batch, opts, epoch, batch_idx):
    loss = training_step(model, batch, opts, epoch, batch_idx)
    return loss

# data loaders
train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
val_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
train_loader = DataLoader(train_dataset, batch_size=1024)
val_loader = DataLoader(val_dataset, batch_size=1024)

# instantiate model and trainer
input_size = X_train.shape[-1]
opts = TrainerOpts(class_weight=class_weight, input_size=input_size)
model = Model(input_size)
train = trainer.Trainer(opts, model, training_step, validation_step)

# fit model
train.fit(train_loader, val_loader, 100)
"""

import os
import math
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter


to_floats = lambda *args: [torch.FloatTensor(a).to(device) for a in args]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.max_epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


@dataclass
class TrainerOpts:
    class_weight: torch.tensor = None
    lr: float = 0.01
    beta: list = (0.9, 0.95)
    grad_norm_clip: float = 1.0
    device: str = 'cpu'
    name: str = 'Untitled'


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Trainer():
    def __init__(self, opts, model, training_step, validation_step=None):
        super().__init__()
        self.opts = opts
        self.training_step = training_step
        self.validation_step = validation_step
        self.model = model.to(opts.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr, betas=opts.beta)

    def fit(self, train_loader, val_loader=None, n_epochs=10, patience=5):
        writer = SummaryWriter()
        patience_counter = 0
        best_loss = np.inf
        epoch = 0

        while (epoch < n_epochs and patience_counter < patience):

            losses = AverageMeter()
            metrics = AverageMeter()

            # train
            self.model.train()
            pbar = tqdm(enumerate(train_loader), total=len(train_loader), position=0, leave=True, desc='Training')
            for batch_idx, batch in pbar:
                # forward pass
                loss = self.training_step(self.model, batch, self.opts, epoch, batch_idx)
                # optimize step
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opts.grad_norm_clip)
                self.optimizer.step()
                losses.update(loss.cpu().item(), batch[0].size(0))

            # eval
            if val_loader is not None:
                self.model.eval()
                pbar = tqdm(enumerate(val_loader), total=len(val_loader), position=0, leave=True, desc='Validation')
                for batch_idx, batch in pbar:
                    metric = self.validation_step(self.model, batch, self.opts, epoch, batch_idx)
                    metrics.update(metric.cpu().item(), batch[0].size(0))

            writer.add_scalars(f'data/{self.opts.name}',{
                'train_loss': losses.avg, 
                'val_loss': metrics.avg
            }, epoch)

            print(f"Epoch: {epoch} | Train loss: {losses.avg:.4f} | Val metric: {metrics.avg: .4f} | Patience: {patience_counter}")

            # check patience
            if metrics.avg < best_loss:
                best_loss = metrics.avg
                patience_counter = 0
                self.save()
            else:
                patience_counter += 1
            if patience_counter == patience:
                break

            epoch += 1

        writer.export_scalars_to_json(os.path.join('./', self.opts.name + '_scalars.json'))
        writer.close()

    def save(self):
        torch.save({'state_dict': self.model.state_dict()}, os.path.join('./', self.opts.name + '.pt'))
