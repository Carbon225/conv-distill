from typing import NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvModelOutput(NamedTuple):
    y: torch.Tensor
    loss: Optional[torch.Tensor]


class LeftPadConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size):
        super().__init__()
        self.norm1 = nn.BatchNorm1d(channels)
        self.conv1 = LeftPadConv1d(channels, channels, kernel_size, padding=kernel_size - 1)
        self.norm2 = nn.BatchNorm1d(channels)
        self.conv2 = LeftPadConv1d(channels, channels, kernel_size, padding=kernel_size - 1)

    def forward(self, x):
        shortcut = x
        x = self.conv1(F.relu(self.norm1(x)))
        x = self.conv2(F.relu(self.norm2(x)))
        return x + shortcut


class ConvModel(nn.Module):
    def __init__(self, *, dim=768, kernel_size=2, num_layers=4):
        super().__init__()
        self.encoder = nn.Sequential(
            LeftPadConv1d(dim, dim, kernel_size, padding=kernel_size - 1),
            *[
                ResBlock(dim, kernel_size)
                for _ in range(num_layers)
            ],
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Conv1d(dim, dim, kernel_size=1)
        )
        self.loss_fn = nn.MSELoss(reduction='none')

    def forward(self, x, y=None, attention_mask=None, **kwargs):
        # (B, L, D)
        x = self.encoder(x.transpose(1, 2)).transpose(1, 2)
        if y is not None:
            loss = self.loss_fn(x, y)
            if attention_mask is not None:
                loss = loss.mean(axis=2) * attention_mask
                loss = loss.sum() / attention_mask.sum()
            else:
                loss = loss.mean()
        else:
            loss = None
        return ConvModelOutput(y=x, loss=loss)


class AttentionSpy(nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.layer_idx = attn.layer_idx
        self.inputs = None
        self.outputs = None

    def forward(self, *args, **kwargs):
        # print('Attention in block', self.layer_idx)
        # print('Positional arguments:')
        # for i, arg in enumerate(args):
        #     print(f'{i}: {arg.shape}')
        # print('Keyword arguments:')
        # for key, value in kwargs.items():
        #     print(f'{key}: {value}')
        # print()

        inputs = args[0]
        outputs = self.attn(*args, **kwargs)

        self.inputs = inputs
        self.outputs = outputs[0]

        return outputs
