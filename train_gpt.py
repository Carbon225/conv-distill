import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import wandb
import os
from tqdm import tqdm

from modeling import ConvModel, AttentionSpy


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
