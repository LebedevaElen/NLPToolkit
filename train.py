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
             metric_param: Dict = None):
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
            for i, (X, F, y) in enumerate(sampler):
                X, F, y = torch.LongTensor(X).cuda(), torch.FloatTensor(
                    F).cuda(), torch.LongTensor(y).cuda()
                output = model(X, F)
                loss_value = loss(output, y)

                epoch_loss += loss_value

                if optimizer is not None:
                    optimizer.zero_grad()
                    loss_value.backward()
                    optimizer.step()

                progress_bar.update()
                progress_bar.set_description(f'{name:>5s} '
                                             f'Loss = {loss_value:.5f}'
                )

                output = torch.argmax(output, dim=1)
                pred.append(output.view(-1).cpu().detach().numpy())
                gt.append(y.view(-1).cpu().detach().numpy().astype(np.uint8))

            pred = np.concatenate(pred)
            gt = np.concatenate(gt)

            if metric is not None:
                metric_value = metric(gt, pred, **metric_param \
                    if metric_param is not None else None)
                progress_bar.set_description(
                    f'{name:>5s} Loss = {epoch_loss / sampler.n_batches:.5f},'
                    f' {metric_name} = {metric_value:.3f}'
            )

        return metric_value


def fit(model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        train_data: Tuple[List[np.array], np.array, np.array],
        iterator: BatchIterator,
        epochs_count: int = 1,
        batch_size: int = 64,
        val_data: Tuple[List[np.array], np.array, np.array] = None,
        val_batch_size: int = None,
        metric: Callable = None,
        metric_name: str = '',
        metric_param: Dict = None):
    """
    Train and evaluate model for epochs_count epochs.
    :param model: model to train
    :param criterion: pytorch loss function
    :param optimizer: pytorch optimizer
    :param train_data: tuple (text_data, target, extra_features)
    :param iterator: batch iterator
    :param epochs_count: number of epochs
    :param batch_size: train batch size
    :param val_data: tuple (text_data, target, extra_features)
    :param val_batch_size: validation batch size
    :param metric: evaluation metric
    :param metric_name: metric name
    :param metric_param: extra metric parameters
    """

    train_text_data, train_target, train_extra_features = train_data
    train_sampler = iterator(train_text_data, train_target, batch_size,
                             extra_features=train_extra_features)

    if val_data is not None:
        if val_batch_size is None:
            val_batch_size = batch_size
        val_text_data, val_target, val_extra_features = val_data
        val_sampler = iterator(val_text_data, val_target, val_batch_size,
                               extra_features=val_extra_features)

    best_metric_value = 0
    best_epoch = -1

    for epoch in range(epochs_count):
        name_prefix = f'[{epoch + 1} / {epochs_count}] '
        _do_epoch(model, criterion, train_sampler, optimizer,
                 name_prefix + 'Train:', metric, metric_name, metric_param)

        if not val_data is None:
            val_metric_value = _do_epoch(model, criterion, val_sampler, None,
                                        name_prefix + '  Val:', metric,
                                        metric_name, metric_param)

        torch.save(model.state_dict(), f'ckpt{epoch:05}.pth')
        if val_metric_value > best_metric_value:
            best_metric_value = val_metric_value
            best_epoch = epoch
            torch.save(model.state_dict(), f'best_model.pth')

        if epoch - best_epoch > 5:
            break