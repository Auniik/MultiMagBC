
import numpy as np
import time
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_curve, roc_auc_score, roc_curve, precision_score, recall_score, confusion_matrix
import torch
from tqdm import tqdm
import torch.nn as nn
from config import mixup_data, mixup_criterion
from utils.helpers import safe_autocast


def train_one_epoch(model, dataloader, criterion, optimizer, device, scheduler=None, use_mixup=True, mixup_alpha=0.2, accumulation_steps=1, epoch=0):
    model.train()
    scaler = torch.GradScaler(enabled=(device.type == "cuda"))
    losses = []
    all_preds, all_labels = [], []
    optimizer.zero_grad()

    for batch_idx, (images_dict, mask, labels) in enumerate(tqdm(dataloader, desc=f'Train Epoch {epoch}', leave=False)):
        images = {k: v.to(device, non_blocking=True) for k, v in images_dict.items()}
        labels = labels.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        with safe_autocast(device):  # AMP for forward pass
            if use_mixup and mixup_alpha > 0:
                mixed_images, y_a, y_b, lam = mixup_data(images, labels, mixup_alpha, device)
                logits = model(mixed_images, mask)
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
                dominant_labels = torch.where(torch.full_like(y_a, lam) >= 0.5, y_a, y_b)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_labels.extend(dominant_labels.cpu().numpy())
            else:
                logits = model(images, mask)
                loss = criterion(logits, labels)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_labels.extend(labels.cpu().numpy())

        # Skip invalid losses
        if not torch.isfinite(loss):
            print(f"⚠️ Skipping batch {batch_idx} due to non-finite loss.")
            continue

        all_preds.extend(preds)
        losses.append(float(loss))

        # Backpropagation
        scaler.scale(loss / accumulation_steps).backward()

        # Gradient accumulation & step
        if (batch_idx + 1) % accumulation_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # **CosineAnnealingWarmRestarts: step per batch**
        if scheduler:
            scheduler.step(epoch + batch_idx / len(dataloader))

    # Handle leftover gradients if dataloader length not divisible by accumulation_steps
    if len(dataloader) % accumulation_steps != 0:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

    acc = accuracy_score(all_labels, all_preds)
    return np.mean(losses), acc

def find_optimal_threshold(y_true, y_probs, optimize_for='accuracy'):
    """Find optimal threshold optimizing for different metrics"""
    # Replace NaNs/Infs to avoid crash
    y_probs = np.nan_to_num(y_probs, nan=0.5, posinf=1.0, neginf=0.0)
    
    if optimize_for == 'accuracy':
        # Optimize for overall accuracy
        thresholds = np.linspace(0.1, 0.9, 100)
        best_acc = 0
        best_thresh = 0.5
        
        for thresh in thresholds:
            preds = (y_probs >= thresh).astype(int)
            acc = accuracy_score(y_true, preds)
            if acc > best_acc:
                best_acc = acc
                best_thresh = thresh
        
        return best_thresh
    
    else:  # Default F1 optimization
        precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        # Clamp threshold range
        best_thresh = thresholds[best_idx] if 0.3 <= thresholds[best_idx] <= 0.7 else 0.5
        return best_thresh

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
    

def eval_model_with_threshold_optimization(
    model, dataloader, criterion, device,
    mc_dropout=True, optimize_for='accuracy',
    return_probs=False
):
    """Evaluate model with threshold optimization and optional MC-Dropout."""
    was_training = model.training
    
    if mc_dropout:
        model.train()
        set_dropout_train_only(model)
        torch.set_grad_enabled(False)
    else:
        model.eval()

    losses, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch_idx, (images_dict, mask, labels) in enumerate(tqdm(dataloader, desc='Val ', leave=False)):
            images = {k: v.to(device, non_blocking=True) for k, v in images_dict.items()}
            labels = labels.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            with safe_autocast(device):
                logits = model(images, mask)
                logits = torch.clamp(logits, -20, 20)  # prevent overflow
                loss = criterion(logits, labels)

            losses.append(float(loss))
            probs = torch.softmax(logits, dim=1)[:, 1].float().cpu().numpy()

            if np.isnan(probs).any():
                print(f"⚠️ NaN detected in batch {batch_idx}. Replacing with 0.5.")
                probs = np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)

            all_probs.extend(probs.tolist())
            all_labels.extend(labels.cpu().numpy())

    # Restore model mode
    if mc_dropout:
        torch.set_grad_enabled(True)
    if was_training:
        model.train()
    else:
        model.eval()

    # Optimize threshold
    optimal_threshold = find_optimal_threshold(all_labels, all_probs, optimize_for=optimize_for)
    all_preds = (np.array(all_probs) >= optimal_threshold).astype(int)

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    result = {
        'val_loss': np.mean(losses),
        'val_acc': acc,
        'val_bal': bal_acc,
        'val_f1': f1,
        'val_auc': auc,
        'val_prec': precision,
        'val_rec': recall,
        'optimal_threshold': optimal_threshold,
        'confusion_matrix': cm.tolist()
    }
    if return_probs:
        result['probabilities'] = all_probs
        result['labels'] = all_labels

    return result