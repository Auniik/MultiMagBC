
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, roc_curve
import torch
from tqdm import tqdm
import torch.nn as nn


def train_one_epoch(model, dataloader, criterion, optimizer, device, use_mixup=True, mixup_alpha=0.2, accumulation_steps=1):
    """Enhanced training with gradient accumulation for better GPU utilization"""
    from config import mixup_data, mixup_criterion
    
    model.train()
    losses = []
    all_preds, all_labels = []
    
    optimizer.zero_grad()
    
    for batch_idx, (images_dict, labels) in enumerate(tqdm(dataloader, desc='Train', leave=False)):
        # Move to device
        images = {k: v.to(device) for k, v in images_dict.items()}
        labels = labels.to(device)
        
        # Apply moderate mixup augmentation
        if use_mixup and mixup_alpha > 0:
            mixed_images, y_a, y_b, lam = mixup_data(images, labels, mixup_alpha, device)
            logits, _ = model(mixed_images)
            loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            
            # For accuracy calculation, use original labels
            with torch.no_grad():
                orig_logits, _ = model(images)
                preds = torch.argmax(orig_logits, dim=1).cpu().numpy()
        else:
            logits, _ = model(images)
            loss = criterion(logits, labels)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        
        # Scale loss by accumulation steps
        loss = loss / accumulation_steps
        loss.backward()
        
        # Update weights every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            # Moderate gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        losses.append(loss.item() * accumulation_steps)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
    # Handle remaining gradients
    if len(dataloader) % accumulation_steps != 0:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    acc = accuracy_score(all_labels, all_preds)
    return np.mean(losses), acc

def find_optimal_threshold(y_true, y_probs):
    """Find optimal threshold that maximizes F1 score (restored for better calibration)"""
    _, _, thresholds = roc_curve(y_true, y_probs)
    best_threshold = 0.5
    best_f1 = 0
    
    # Filter out extreme thresholds
    valid_thresholds = [t for t in thresholds if 0.1 <= t <= 0.9]
    
    for threshold in valid_thresholds:
        y_pred = (y_probs >= threshold).astype(int)
        try:
            current_f1 = f1_score(y_true, y_pred, zero_division=0)
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_threshold = threshold
        except:
            continue
    
    return best_threshold

def eval_model(model, dataloader, criterion, device, optimal_threshold=0.5):
    """Evaluate model with a given threshold (do not optimize threshold on test set)"""
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
    
    # Use provided threshold (not optimized on this set)
    all_preds = (np.array(all_probs) >= optimal_threshold).astype(int)
    
    acc = accuracy_score(all_labels, all_preds)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    
    return np.mean(losses), acc, bal_acc, f1, auc, all_probs

def eval_model_with_threshold_optimization(model, dataloader, criterion, device, use_dropout=True):
    """Evaluate model and find optimal threshold with enhanced overfitting detection"""
    if use_dropout:
        # Keep model in training mode for validation dropout regularization
        model.train()
        # But disable gradient computation
        torch.set_grad_enabled(False)
    else:
        model.eval()
        
    losses = []
    all_labels, all_probs = [], []
    
    with torch.no_grad():
        for images_dict, labels in tqdm(dataloader, desc='Val ', leave=False):
            images = {k: v.to(device) for k, v in images_dict.items()}
            labels = labels.to(device)
            logits, _ = model(images)
            loss = criterion(logits, labels)

            losses.append(loss.item())
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.cpu().numpy())
    
    # Re-enable gradients
    if use_dropout:
        torch.set_grad_enabled(True)
    
    # Find optimal threshold on validation set
    optimal_threshold = find_optimal_threshold(all_labels, all_probs)
    
    # Calculate predictions with optimal threshold
    all_preds = (np.array(all_probs) >= optimal_threshold).astype(int)
    
    acc = accuracy_score(all_labels, all_preds)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    
    return np.mean(losses), acc, bal_acc, f1, auc, optimal_threshold