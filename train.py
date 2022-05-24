import time
import torch
import numpy as np
from torch import nn, optim
from collections import defaultdict

class TrainRunner:
    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        epochs,
        learning_rate,
        lr_dc_step,     
        weight_decay,
        patience,
        device,
    ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.lr_dc_step = lr_dc_step
        self.weight_decay = weight_decay
        self.patience = patience
        self.device = device
    
    def train(self):
        model = self.model.to(self.device)
        params = model.parameters()
        optimizer = optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_dc_step)
        max_results = defaultdict(float)
        max_epochs = defaultdict(int)
        mean_loss = 0
        t = time.time()
        cnt = 0
        for epoch in range(self.epochs):
            log_interval = 100
            model.train()
            for batch in self.train_loader:
                batch = move_to_device(batch, self.device)
                optimizer.zero_grad()
                logits = model(batch)
                labels = batch['labels'] - 2
                loss = nn.functional.cross_entropy(logits, labels)
                loss.backward()
                optimizer.step()
                mean_loss += loss.item() / log_interval
                if cnt > 0 and cnt % log_interval == 0:
                    print(
                        f'Batch {cnt}: Loss = {mean_loss:.4f}, Time Elapsed = {time.time() - t:.2f}s'
                    )
                    t = time.time()
                    mean_loss = 0
                cnt += 1
            scheduler.step()
            curr_results = evaluate(
                self.model, self.test_loader, self.device,
            )

            print(f'\nEpoch {epoch}:')
            print_results(curr_results)

            any_better_result = False
            for metric in curr_results:
                if curr_results[metric] > max_results[metric]:
                    max_results[metric] = curr_results[metric]
                    max_epochs[metric] = epoch
                    any_better_result = True

            if any_better_result:
                bad_counter = 0
            else:
                bad_counter += 1
                if bad_counter == self.patience:
                    break

        print('\nBest results')
        print_results(max_results, max_epochs)
        return max_results

def print_results(results, epochs=None):
    print('Metric\t' + '\t'.join(results.keys()))
    print(
        'Value\t' +
        '\t'.join([f'{round(val * 100, 2):.2f}' for val in results.values()])
    )
    if epochs is not None:
        print('Epoch\t' + '\t'.join([str(epochs[metric]) for metric in results]))

def evaluate(model, data_loader, device, Ks=[10, 20]):
    model.eval()
    num_samples = 0
    max_K = max(Ks)
    results = defaultdict(float)
    with torch.no_grad():
        for batch in data_loader:
            batch = move_to_device(batch, device)
            logits = model(batch)
            labels = batch['labels'] - 2
            batch_size = logits.size(0)
            num_samples += batch_size
            topk = torch.topk(logits, k=max_K, sorted=True)[1]
            labels = labels.unsqueeze(-1)
            for K in Ks:
                hit_ranks = torch.where(topk[:, :K] == labels)[1] + 1
                hit_ranks = hit_ranks.float().cpu()
                results[f'HR@{K}'] += hit_ranks.numel()
                results[f'MRR@{K}'] += hit_ranks.reciprocal().sum().item()
                results[f'NDCG@{K}'] += torch.log2(1 + hit_ranks).reciprocal().sum().item()
    for metric in results:
        results[metric] /= num_samples
    return results

def move_to_device(maybe_tensor, device):
    if torch.is_tensor(maybe_tensor):
        return maybe_tensor.to(device)
    elif isinstance(maybe_tensor, np.ndarray):
        return torch.from_numpy(maybe_tensor).to(device).contiguous()
    elif isinstance(maybe_tensor, dict):
        return {
            key: move_to_device(value, device)
            for key, value in maybe_tensor.items()
        }
    elif isinstance(maybe_tensor, list):
        return [move_to_device(x, device) for x in maybe_tensor]
    else:
        return maybe_tensor
