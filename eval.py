from typing import Optional

import torch
from torch.optim import Adam
import torch.nn as nn

from model import LogReg


def get_idx_split(data, split, preload_split):
    if split[:4] == 'rand':
        train_ratio = float(split.split(':')[1])
        num_nodes = data.x.size(0)
        train_size = int(num_nodes * train_ratio)
        indices = torch.randperm(num_nodes)
        return {
            'train': indices[:train_size],
            'val': indices[train_size:2 * train_size],
            'test': indices[2 * train_size:]
        }
    elif split.startswith('wikics'):
        split_idx = int(split.split(':')[1])
        return {
            'train': data.train_mask[:, split_idx],
            'test': data.test_mask,
            'val': data.val_mask[:, split_idx]
        }
    elif split == 'preloaded':
        assert preload_split is not None, 'use preloaded split, but preloaded_split is None'
        train_mask, test_mask, val_mask = preload_split
        return {
            'train': train_mask,
            'test': test_mask,
            'val': val_mask
        }
    else:
        raise RuntimeError(f'Unknown split type {split}')


def linear_regression(z,
                      data,
                      y,
                      evaluator,
                      num_epochs: int = 5000,
                      test_device: Optional[str] = None,
                      split: str = 'rand:0.1',
                      preload_split=None):
    test_device = z.device if test_device is None else test_device
    z = z.detach().to(test_device)
    num_hidden = z.size(1)
    # y = dataset[0].degree_y.view(-1).to(test_device)

    # Linear regression model
    class LinearReg(nn.Module):
        def __init__(self, num_features):
            super(LinearReg, self).__init__()
            self.linear = nn.Linear(num_features, 1)

        def forward(self, x):
            return self.linear(x)

    predictor = LinearReg(num_hidden).to(test_device)

    # Optimizer and loss function
    optimizer = Adam(predictor.parameters(), lr=0.01, weight_decay=0.0)
    mse_loss = nn.MSELoss()

    split = get_idx_split(data, split, preload_split)
    split = {k: v.to(test_device) for k, v in split.items()}
    best_test_r2 = 0
    best_test_mae = float('inf')

    for epoch in range(num_epochs):
        predictor.train()
        optimizer.zero_grad()

        output = predictor(z[split['train']])
        loss = mse_loss(output.view(-1), y[split['train']])
        # if epoch == num_epochs - 1:
        #     final_mse = loss.item()
            
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:

            res = evaluator.eval({
                'y_true': y[split['test']].view(-1, 1),
                'y_pred': predictor(z[split['test']]).view(-1, 1)
            })
            if best_test_r2 < res['r2']:
                best_test_r2 = res['r2']
            if best_test_mae > res['mae']:
                best_test_mae = res['mae']
    

    return {'r2': best_test_r2, 'mae': best_test_mae}
from sklearn.metrics import r2_score
class LinearRegressionEvaluator:
    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def _eval(y_true, y_pred):
        y_true = y_true.float()
        y_pred = y_pred.float()
        y_mean = torch.mean(y_true)
        squared_error = torch.sum((y_true - y_pred) ** 2)
        total_variation = torch.sum((y_true - y_mean) ** 2)
        r2 = 1.0 - (squared_error / total_variation)

        mae = torch.mean(torch.abs(y_true - y_pred))

        return r2.item(), mae.item()

    def eval(self, res):
        r2, mae = self._eval(**res)
        return {'r2': r2, 'mae': mae}
