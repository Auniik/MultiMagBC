
import numpy as np
import time
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_curve, roc_auc_score, roc_curve, precision_score, recall_score, confusion_matrix
import torch
from tqdm import tqdm
import torch.nn as nn
from config import mixup_data, mixup_criterion
from utils.helpers import safe_autocast


def train_one_epoch(model, dataloader, criterion, optimizer, device, use_mixup=True, mixup_alpha=0.2, accumulation_steps=1):
    model.train()
    scaler = torch.GradScaler(enabled=(device.type == "cuda"))
    losses = []
    all_preds, all_labels = [], []
    optimizer.zero_grad()

    for batch_idx, (images_dict, mask, labels) in enumerate(tqdm(dataloader, desc='Train', leave=False)):
        images = {k: v.to(device, non_blocking=True) for k, v in images_dict.items()}
        labels = labels.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        with safe_autocast(device):  # AMP for forward pass
            if use_mixup and mixup_alpha > 0:
                mixed_images, y_a, y_b, lam = mixup_data(images, labels, mixup_alpha, device)
                logits = model(mixed_images, mask)
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam)  # must return tensor
                lam_tensor = torch.full_like(y_a, lam)
                dominant_labels = torch.where(lam_tensor >= 0.5, y_a, y_b)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_labels.extend(dominant_labels.cpu().numpy())
            else:
                logits = model(images, mask)
                loss = criterion(logits, labels)  # keep as tensor
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_labels.extend(labels.cpu().numpy())

        all_preds.extend(preds)

        # Scaled backward pass
        scaler.scale(loss / accumulation_steps).backward()

        # Gradient accumulation & step
        if (batch_idx + 1) % accumulation_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Only convert to float AFTER backward
        losses.append(loss.item())

    # Handle leftover gradients if dataloader length not divisible by accumulation_steps
    if len(dataloader) % accumulation_steps != 0:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

    acc = accuracy_score(all_labels, all_preds)
    return np.mean(losses), acc

def find_optimal_threshold(y_true, y_probs):
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    f1_scores = f1_scores[:-1]  # trim to match thresholds length
    valid = (thresholds >= 0.3) & (thresholds <= 0.7)
    if not np.any(valid):
        return 0.5
    best_idx = np.argmax(f1_scores[valid])
    return thresholds[valid][best_idx]

def eval_model(model, dataloader, criterion, device, optimal_threshold=0.5):
    model.eval()
    losses = []
    all_preds, all_labels, all_probs = [], [], []
    start_time = time.time()

    with torch.no_grad():
        for images_dict, mask, labels in tqdm(dataloader, desc='Eval ', leave=False):
            images = {k: v.to(device, non_blocking=True) for k, v in images_dict.items()}
            labels = labels.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            
            # Use mixed precision for faster inference
            with safe_autocast(device):
                logits = model(images, mask)
                loss = criterion(logits, labels)

            losses.append(float(loss))
            probs = torch.softmax(logits, dim=1)[:, 1].float().cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.cpu().numpy())
    
    total_time = time.time() - start_time
    avg_inference_time = total_time / len(dataloader.dataset)
    
    # Apply threshold
    all_preds = (np.array(all_probs) >= optimal_threshold).astype(int)
    
    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    
    cm = confusion_matrix(all_labels, all_preds)
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    
    return {
        'loss': np.mean(losses),
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'f1_score': f1,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'all_probs': all_probs,
        'confusion_matrix': cm.tolist(),
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'thresholds': thresholds.tolist(),
        'avg_inference_time': avg_inference_time
    }

def set_dropout_train_only(model):
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
        elif isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
            module.eval()
    

def eval_model_with_threshold_optimization(model, dataloader, criterion, device, use_dropout=True):
    """Evaluate model with safe dropout + NaN handling + clamped logits."""
    model.eval()
    torch.set_grad_enabled(False)
    if use_dropout:
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()

    losses, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images_dict, mask, labels in tqdm(dataloader, desc='Val ', leave=False):
            images = {k: v.to(device, non_blocking=True) for k, v in images_dict.items()}
            labels = labels.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            with safe_autocast(device):
                logits = model(images, mask)

            # Clamp + check for NaNs
            logits = torch.clamp(logits, -20, 20)
            if torch.isnan(logits).any():
                print("⚠️ NaN detected in logits! Skipping batch.")
                continue

            loss = criterion(logits, labels)
            losses.append(loss.item())

            probs = torch.softmax(logits, dim=1)[:, 1].float().cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.cpu().numpy())

    # Threshold optimization
    optimal_threshold = find_optimal_threshold(all_labels, all_probs)
    all_preds = (np.array(all_probs) >= optimal_threshold).astype(int)

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)

    return np.mean(losses), acc, bal_acc, f1, auc, precision, recall, optimal_threshold