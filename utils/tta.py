#!/usr/bin/env python3
"""
Test-Time Augmentation (TTA) utilities for improved inference
"""

import torch
import torch.nn.functional as F
from typing import Dict, List
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, roc_curve
import time


def apply_tta_transforms(images_dict: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
    """
    Apply test-time augmentation transforms to input images.
    
    Returns list of augmented image dictionaries for ensemble prediction.
    """
    augmented_inputs = []
    
    # Original (no augmentation)
    augmented_inputs.append(images_dict)
    
    # Horizontal flip
    flipped = {}
    for mag_key, img in images_dict.items():
        flipped[mag_key] = torch.flip(img, dims=[-1])  # Flip width dimension
    augmented_inputs.append(flipped)
    
    # Vertical flip
    v_flipped = {}
    for mag_key, img in images_dict.items():
        v_flipped[mag_key] = torch.flip(img, dims=[-2])  # Flip height dimension
    augmented_inputs.append(v_flipped)
    
    # Both flips
    both_flipped = {}
    for mag_key, img in images_dict.items():
        both_flipped[mag_key] = torch.flip(img, dims=[-2, -1])  # Flip both dimensions
    augmented_inputs.append(both_flipped)
    
    return augmented_inputs


def tta_predict(model, images_dict: Dict[str, torch.Tensor], mask=None) -> torch.Tensor:
    """
    Perform test-time augmentation prediction by averaging predictions
    across multiple augmented versions of the input.
    
    Args:
        model: The trained model
        images_dict: Dictionary of input images
        mask: Optional mask tensor
        
    Returns:
        Averaged prediction logits
    """
    model.eval()
    
    # Get augmented inputs
    augmented_inputs = apply_tta_transforms(images_dict)
    
    predictions = []
    
    with torch.no_grad():
        for aug_input in augmented_inputs:
            # Forward pass
            if mask is not None:
                logits = model(aug_input, mask)
            else:
                logits = model(aug_input)
                
            # Handle tuple output (if model returns multiple outputs)
            if isinstance(logits, tuple):
                logits = logits[0]
                
            predictions.append(logits)
    
    # Average predictions
    avg_prediction = torch.stack(predictions).mean(dim=0)
    
    return avg_prediction


def evaluate_with_tta(model, dataloader, device, optimal_threshold=0.5, return_raw=False):
    """
    Evaluate model with test-time augmentation. 
    If return_raw=True, also return raw probabilities and labels for external analysis.
    """
    
    
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    start_time = time.time()
    
    with torch.no_grad():
        for images_dict, mask, labels in dataloader:
            images_dict = {k: v.to(device) for k, v in images_dict.items()}
            mask = mask.to(device)
            labels = labels.to(device)
            logits = tta_predict(model, images_dict, mask)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = (np.array(all_probs) >= optimal_threshold).astype(int)
    
    accuracy = accuracy_score(all_labels, all_preds)
    balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    
    end_time = time.time()
    avg_inference_time = (end_time - start_time) / len(all_labels)
    
    results = {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy, 
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'auc': auc,
        'confusion_matrix': cm,
        'avg_inference_time': avg_inference_time,
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(), 
        'thresholds': thresholds.tolist()
    }
    if return_raw:
        results.update({'all_labels': all_labels, 'all_probs': all_probs})
    return results