
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
    
    # Initialize gradient clipping parameters
    max_grad_norm = 1.0
    scaler_updated = False  # Track scaler state

    for batch_idx, (images_dict, mask, labels) in enumerate(tqdm(dataloader, desc='Train', leave=False)):
        images = {k: v.to(device, non_blocking=True) for k, v in images_dict.items()}
        labels = labels.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        with safe_autocast(device):  # AMP for forward pass
            if use_mixup and mixup_alpha > 0:
                mixed_images, y_a, y_b, lam = mixup_data(images, labels, mixup_alpha, device)
                logits = model(mixed_images, mask)
                # Clamp logits for numerical stability
                logits = torch.clamp(logits, -20, 20)
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
                lam_tensor = torch.full_like(y_a, lam)
                dominant_labels = torch.where(lam_tensor >= 0.5, y_a, y_b)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_labels.extend(dominant_labels.cpu().numpy())
            else:
                logits = model(images, mask)
                # Clamp logits for numerical stability
                logits = torch.clamp(logits, -20, 20)
                loss = criterion(logits, labels)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_labels.extend(labels.cpu().numpy())
                
            # Check for NaN/Inf in loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️ NaN/Inf loss detected in batch {batch_idx}. Skipping batch.")
                optimizer.zero_grad()  # Clear gradients
                continue

        all_preds.extend(preds)

        # Scaled backward pass
        scaler.scale(loss / accumulation_steps).backward()

        # Gradient accumulation & step
        if (batch_idx + 1) % accumulation_steps == 0:
            # Unscale gradients before clipping (only once per accumulation step)
            try:
                scaler.unscale_(optimizer)
                scaler_updated = True
                
                # Apply gradient clipping
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                
                # Check for exploding gradients
                if torch.isfinite(grad_norm) and grad_norm <= max_grad_norm * 10:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    print(f"⚠️ Large gradient norm detected: {grad_norm:.4f}. Skipping step.")
                    scaler.update()  # Still need to update scaler
                    
            except RuntimeError as e:
                if "unscale_() has already been called" in str(e):
                    # Scaler already unscaled, just proceed with step
                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                    if torch.isfinite(grad_norm) and grad_norm <= max_grad_norm * 10:
                        scaler.step(optimizer)
                    scaler.update()
                else:
                    raise e
                    
            optimizer.zero_grad()
            scaler_updated = False

        losses.append(float(loss))

    # Handle leftover gradients if dataloader length not divisible by accumulation_steps
    if len(dataloader) % accumulation_steps != 0 and not scaler_updated:
        try:
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            if torch.isfinite(grad_norm) and grad_norm <= max_grad_norm * 10:
                scaler.step(optimizer)
            scaler.update()
        except RuntimeError as e:
            if "unscale_() has already been called" not in str(e):
                raise e

    acc = accuracy_score(all_labels, all_preds)
    return np.mean(losses), acc

def find_optimal_threshold(y_true, y_probs):
    # Replace NaNs/Infs to avoid crash
    y_probs = np.nan_to_num(y_probs, nan=0.5, posinf=1.0, neginf=0.0)
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
    

def eval_model_with_threshold_optimization(model, dataloader, criterion, device, mc_dropout=True):
    """Evaluate model with mixed precision (AMP) and safe threshold finding."""
    if mc_dropout:
        model.train()
        set_dropout_train_only(model)
        torch.set_grad_enabled(False)
    else:
        model.eval()

    losses = []
    all_labels, all_probs = [], []

    with torch.no_grad():
        for batch_idx, (images_dict, mask, labels) in enumerate(tqdm(dataloader, desc='Val ', leave=False)):
            images = {k: v.to(device, non_blocking=True) for k, v in images_dict.items()}
            labels = labels.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            with safe_autocast(device):
                logits = model(images, mask)
                # Clamp logits to avoid softmax overflow
                logits = torch.clamp(logits, -20, 20)
                loss = criterion(logits, labels)

            losses.append(float(loss))
            probs = torch.softmax(logits, dim=1)[:, 1].float().cpu().numpy()

            # Replace NaNs immediately if any
            if np.isnan(probs).any():
                print(f"⚠️ NaN detected in batch {batch_idx} probs. Replacing with 0.5.")
                probs = np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)

            all_probs.extend(probs.tolist())
            all_labels.extend(labels.cpu().numpy())

    if mc_dropout:
        torch.set_grad_enabled(True)

    # Find optimal threshold (now safe)
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