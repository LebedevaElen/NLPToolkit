from tqdm import tqdm
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from iterator import BatchIterator


def _do_epoch(model: nn.Module,
              loss: nn.Module,
              sampler: Iterable,
              optimizer: optim.Optimizer = None,
              name: str = None,
              metric: Callable = None,
              metric_name: str = '',
              metric_param: Dict = None,
              metric_threshold: float = None) -> float:
    """
    Train model for one epoch

    :param model: pytorch model
    :param loss: pytorch loss
    :param sampler: batch iterator
    :param optimizer: pytorch optimizer
    :param name: tqdm prefix
    :param metric: metric for evaluating results
    :param metric_name: metric name
    :param metric_param: additional metric parameters
    :return: metric value
    """
    epoch_loss = 0
    is_train = not optimizer is None

    pred = []
    gt = []

    with torch.autograd.set_grad_enabled(is_train):
        with tqdm(total=sampler.n_batches) as progress_bar:
            for i, (X, y, F) in enumerate(sampler):
                X = torch.LongTensor(X).cuda()
                y = torch.FloatTensor(y).cuda()
                if F is not None:
                    F = torch.FloatTensor(F).cuda()
                output = model(X, F)
                # print(output)
                loss_value = loss(output, y.view(-1, 1))

                epoch_loss += loss_value

                if optimizer is not None:
                    optimizer.zero_grad()
                    loss_value.backward()
                    optimizer.step()

                progress_bar.update()
                progress_bar.set_description(f'{name:>5s} '
                                             f'Loss = {loss_value:.5f}'
                                             )

                output = output.view(-1).cpu().detach().numpy()
                if metric_threshold is not None:
                    output = (output > metric_threshold).astype(np.uint8)
                pred.append(output)
                gt.append(y.view(-1).cpu().detach().numpy().astype(np.uint8))

            pred = np.concatenate(pred)
            gt = np.concatenate(gt)

            if metric is not None:
                if metric_param is not None:
                    metric_value = metric(gt, pred, **metric_param)
                else:
                    metric_value = metric(gt, pred)

                progress_bar.set_description(
                    f'{name:>5s} Loss = {epoch_loss / sampler.n_batches:.5f},'
                    f' {metric_name} = {metric_value:.3f}'
                )

        return metric_value


def fit(model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        train_iterator: BatchIterator,
        epochs_count: int = 1,
        val_iterator: BatchIterator = None,
        metric: Callable = None,
        metric_name: str = '',
        metric_param: Dict = None,
        metric_threshold: float = None):
    """
    Train and evaluate model for epochs_count epochs.
    :param model: model to train
    :param criterion: pytorch loss function
    :param optimizer: pytorch optimizer
    :param train_iterator: train batch iterator
    :param epochs_count: number of epochs
    :param val_iterator: validation batch iterator
    :param metric: evaluation metric
    :param metric_name: metric name
    :param metric_param: extra metric parameters
    :param metric_threshold: metric threshold
    """

    best_metric_value = 0
    best_epoch = -1

    for epoch in range(epochs_count):
        name_prefix = f'[{epoch + 1} / {epochs_count}] '
        _do_epoch(model, criterion, train_iterator, optimizer,
                  name_prefix + 'Train:', metric, metric_name, metric_param,
                  metric_threshold)

        if val_iterator is not None:
            val_metric_value = _do_epoch(model, criterion, val_iterator, None,
                                         name_prefix + '  Val:', metric,
                                         metric_name, metric_param,
                                         metric_threshold)

        torch.save(model.state_dict(), f'ckpt{epoch:05}.pth')
        if val_metric_value > best_metric_value:
            best_metric_value = val_metric_value
            best_epoch = epoch
            torch.save(model.state_dict(), f'best_model.pth')

        if epoch - best_epoch > 5:
            break
