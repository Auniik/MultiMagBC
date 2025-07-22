
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, roc_curve
import torch
from tqdm import tqdm
import torch.nn as nn


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    losses = []
    all_preds, all_labels = [], []
    for images_dict, labels in tqdm(dataloader, desc='Train', leave=False):
        # move to device
        images = {k: v.to(device) for k, v in images_dict.items()}
        labels = labels.to(device)
        optimizer.zero_grad()
        logits, _ = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        losses.append(loss.item())
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    return np.mean(losses), acc

def find_optimal_threshold(y_true, y_probs):
    """Find optimal threshold that maximizes F1 score"""
    _, _, thresholds = roc_curve(y_true, y_probs)
    best_threshold = 0.5
    best_f1 = 0
    
    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)
        try:
            current_f1 = f1_score(y_true, y_pred)
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_threshold = threshold
        except:
            continue
    
    return best_threshold

def eval_model(model, dataloader, criterion, device):
    model.eval()
    losses = []
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images_dict, labels in tqdm(dataloader, desc='Eval ', leave=False):
            images = {k: v.to(device) for k, v in images_dict.items()}
            labels = labels.to(device)
            logits, _ = model(images)
            loss = criterion(logits, labels)

            losses.append(loss.item())
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.cpu().numpy())
    
    # Find optimal threshold
    optimal_threshold = find_optimal_threshold(all_labels, all_probs)
    
    # Calculate predictions with optimal threshold
    all_preds = (np.array(all_probs) >= optimal_threshold).astype(int)
    
    acc = accuracy_score(all_labels, all_preds)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    
    return np.mean(losses), acc, bal_acc, f1, auc