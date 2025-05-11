# =====================
# File: src/train_eval.py
# =====================
import torch
import numpy as np
from torch import nn
import dgl
from dgl.dataloading import DataLoader
from copy import deepcopy
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)


def infer(model_cp, hg, mask, batch_size, sampler_type, n_neighbors, best_model_fp=None):
    model = deepcopy(model_cp)
    if best_model_fp:
        model.load_state_dict(torch.load(best_model_fp))
    model.to(hg.device)
    model.eval()
    features = hg.ndata['feat']
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(len(model.convs)) if sampler_type == 1 \
              else dgl.dataloading.MultiLayerNeighborSampler([n_neighbors]*len(model.convs))
    dataloader = DataLoader(
        hg, {'bizIdx': torch.where(mask)[0]}, sampler,
        batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
    y_preds = []
    for input_nodes, output_nodes, blocks in dataloader:
        h = {k: features[k][input_nodes[k]].to(hg.device) for k in input_nodes}
        blocks = [b.to(hg.device) for b in blocks]
        logits = model(blocks, h)
        y_preds.append(logits.softmax(dim=1))
    return torch.cat(y_preds).cpu()


def test(model, hg, mask, batch_size, sampler_type, n_neighbors):
    model.eval()
    features = hg.ndata['feat']
    labels = hg.ndata['label']['bizIdx'].long()
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(len(model.convs)) if sampler_type == 1 \
              else dgl.dataloading.MultiLayerNeighborSampler([n_neighbors]*len(model.convs))
    dataloader = DataLoader(
        hg, {'bizIdx': torch.where(mask)[0]}, sampler,
        batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
    correct = 0
    for input_nodes, output_nodes, blocks in dataloader:
        h = {k: features[k][input_nodes[k]].to(hg.device) for k in input_nodes}
        blocks = [b.to(hg.device) for b in blocks]
        logits = model(blocks, h)
        y_preds = logits.argmax(1)
        correct += (labels[output_nodes['bizIdx']] == y_preds.cpu()).sum().item()
    return correct / torch.where(mask)[0].shape[0]


def evaluate(model_cp, hg, mask, batch_size, sampler_type, n_neighbors, best_model_fp=None):
    model = deepcopy(model_cp)
    if best_model_fp:
        model.load_state_dict(torch.load(best_model_fp))
    model.to(hg.device)
    model.eval()
    features = hg.ndata['feat']
    labels = hg.ndata['label']['bizIdx'].long()
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(len(model.convs)) if sampler_type == 1 \
              else dgl.dataloading.MultiLayerNeighborSampler([n_neighbors]*len(model.convs))
    dataloader = DataLoader(
        hg, {'bizIdx': torch.where(mask)[0]}, sampler,
        batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
    y_true, y_pred_probs, logits_list = [], [], []
    for input_nodes, output_nodes, blocks in dataloader:
        h = {k: features[k][input_nodes[k]].to(hg.device) for k in input_nodes}
        blocks = [b.to(hg.device) for b in blocks]
        logits = model(blocks, h)
        y_pred_probs.append(logits.softmax(dim=1).cpu().numpy())
        logits_list.append(logits.cpu().detach().numpy())
        y_true.append(labels[output_nodes['bizIdx']].cpu().numpy())
    y_true = np.concatenate(y_true)
    y_pred_probs = np.concatenate(y_pred_probs)
    acc = accuracy_score(y_true, y_pred_probs.argmax(axis=1))
    precision = precision_score(y_true, y_pred_probs.argmax(axis=1), zero_division=0)
    recall = recall_score(y_true, y_pred_probs.argmax(axis=1), zero_division=0)
    f1 = f1_score(y_true, y_pred_probs.argmax(axis=1), zero_division=0)
    cm = confusion_matrix(y_true, y_pred_probs.argmax(axis=1))
    auc = roc_auc_score(y_true, y_pred_probs[:, 1])
    return acc, precision, recall, f1, cm, auc

def extract_embeddings(model, hg, mask, batch_size, sampler_type, n_neighbors, best_model_fp=None):
    if best_model_fp:
        model.load_state_dict(torch.load(best_model_fp))
    model.to(hg.device)
    model.eval()
    features = hg.ndata['feat']
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(len(model.convs)) if sampler_type == 1 \
              else dgl.dataloading.MultiLayerNeighborSampler([n_neighbors]*len(model.convs))
    dataloader = DataLoader(
        hg, {'bizIdx': torch.where(mask)[0]}, sampler,
        batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
    embeddings_list = []
    for input_nodes, output_nodes, blocks in dataloader:
        h = {k: features[k][input_nodes[k]].to(hg.device) for k in input_nodes}
        blocks = [b.to(hg.device) for b in blocks]
        embeddings = model(blocks, h, return_embeddings=True)
        embeddings_list.append(embeddings.cpu().numpy())
    return np.concatenate(embeddings_list, axis=0)

# =====================
# Training functions
# =====================
def train_step(model, hg, features, train_mask, val_mask, optimizer, loss_fn, batch_size,
               sampler_type, n_neighbors, weight_values, weight=True):
    model.train()
    optimizer.zero_grad()
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(len(model.convs)) if sampler_type == 1 \
              else dgl.dataloading.MultiLayerNeighborSampler([n_neighbors]*len(model.convs))
    dataloader = DataLoader(
        hg, {'bizIdx': torch.where(train_mask)[0]}, sampler,
        batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
    val_loader = DataLoader(
        hg, {'bizIdx': torch.where(val_mask)[0]}, sampler,
        batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)

    train_losses, val_losses = [], []
    if weight:
        weight = torch.tensor(weight_values).float().to(hg.device)
    else:
        weight = None
    labels = hg.ndata['label']['bizIdx'].long()

    for input_nodes, output_nodes, blocks in dataloader:
        h = {k: features[k][input_nodes[k]].to(hg.device) for k in input_nodes}
        blocks = [b.to(hg.device) for b in blocks]
        logits = model(blocks, h)
        loss = loss_fn(logits, labels[output_nodes['bizIdx']], weight=weight)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_losses.append(loss.item())

    with torch.no_grad():
        for input_nodes, output_nodes, blocks in val_loader:
            h = {k: features[k][input_nodes[k]].to(hg.device) for k in input_nodes}
            blocks = [b.to(hg.device) for b in blocks]
            logits = model(blocks, h)
            val_loss = loss_fn(logits, labels[output_nodes['bizIdx']])
            val_losses.append(val_loss.item())

    return np.mean(train_losses), np.mean(val_losses)


def train(hg, split_idx, model, epochs, batch_size, sampler_type, n_neighbors, weight_values,
          device, save_path, loss_fn=nn.CrossEntropyLoss(), lr=0.001, es_criteria=5, weight=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best_val_metric = 1e10
    train_losses, val_losses = [], []
    features = hg.ndata['feat']
    train_mask = split_idx['train'].to(device)
    val_mask = split_idx['valid'].to(device)
    es_iters = 0

    for e in range(1, epochs + 1):
        train_loss, val_loss = train_step(
            model, hg, features, train_mask, val_mask, optimizer,
            loss_fn, batch_size, sampler_type, n_neighbors, weight_values, weight)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_metric = val_loss

        if val_metric > 1e5:
            break
        if val_metric < best_val_metric:
            best_val_metric = val_metric
            torch.save(model.state_dict(), save_path)
            es_iters = 0
        else:
            es_iters += 1
        if es_iters >= es_criteria:
            break

    return np.array(train_losses), np.array(val_losses)
