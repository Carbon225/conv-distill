import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import wandb
import os
from typing import NamedTuple, Optional
from tqdm import tqdm


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def main():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    for block in model.transformer.h:
        block.attn = AttentionSpy(block.attn)

    conv_blocks = [
        ConvModel(dim=768, kernel_size=2, num_layers=4)
        for _ in range(len(model.transformer.h))
    ]

    for i, conv_block in enumerate(conv_blocks):
        if os.path.exists(f'conv_block_{i}.pth'):
            conv_block.load_state_dict(torch.load(f'conv_block_{i}.pth', map_location=torch.device('cpu')))

    model.to(DEVICE)
    model.eval()

    for conv_block in conv_blocks:
        conv_block.to(DEVICE)
        conv_block.train()

    conv_optimizers = [
        torch.optim.Adam(conv_block.parameters(), lr=1e-3)
        for conv_block in conv_blocks
    ]

    dataset = load_dataset("imdb", split='unsupervised')

    def tokenize(examples):
        return tokenizer(examples['text'], return_tensors='pt', padding=True, truncation=True)

    dataset_tokenized = dataset.map(tokenize, batched=True, remove_columns=['text'])

    loader = DataLoader(dataset_tokenized, batch_size=8, shuffle=True)

    wandb.init(project='conv-distill')

    with torch.no_grad():
        with tqdm(total=len(loader)) as pbar:
            for batch in loader:
                input_ids = torch.stack(batch['input_ids']).to(DEVICE)
                attention_mask = torch.stack(batch['attention_mask']).to(DEVICE)

                model(input_ids=input_ids, attention_mask=attention_mask)

                total_loss = 0
                block_losses = []

                for i, conv_block in enumerate(conv_blocks):
                    optimizer = conv_optimizers[i]
                    attn_block = model.transformer.h[i].attn

                    attn_inputs = attn_block.inputs
                    attn_outputs = attn_block.outputs

                    with torch.enable_grad():
                        optimizer.zero_grad()
                        conv_outputs = conv_block(attn_inputs, attn_outputs, attention_mask)
                        loss = conv_outputs.loss
                        total_loss += loss.item()
                        block_losses.append(loss.item())
                        loss.backward()
                        optimizer.step()

                logs = {}
                logs['train/loss'] = total_loss / len(conv_blocks)
                for i, loss in enumerate(block_losses):
                    logs[f'train/block_{i}_loss'] = loss
                wandb.log(logs)

                pbar.update(1)
                pbar.set_postfix(loss=total_loss / len(conv_blocks))

    for i, conv_block in enumerate(conv_blocks):
        conv_block.cpu()
        conv_block.eval()
        torch.save(conv_block.state_dict(), f'conv_block_{i}.pth')


if __name__ == '__main__':
    main()
