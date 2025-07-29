#!/usr/bin/env python3
"""
Ensemble utilities for training multiple models and averaging predictions
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict
from backbones.our.lightweight import MultiMagLightweightCNN


class LightweightEnsemble(nn.Module):
    """
    Ensemble of multiple lightweight models with different initializations
    """
    
    def __init__(self, num_models: int = 3, num_classes: int = 2, dropout: float = 0.4):
        super().__init__()
        
        self.models = nn.ModuleList([
            MultiMagLightweightCNN(num_classes=num_classes, dropout=dropout)
            for _ in range(num_models)
        ])
        self.num_models = num_models
        
    def forward(self, images_dict: Dict[str, torch.Tensor], mask=None):
        """Forward pass through all models and average predictions"""
        predictions = []
        
        for model in self.models:
            logits = model(images_dict, mask)
            predictions.append(logits)
            
        # Average predictions
        avg_prediction = torch.stack(predictions).mean(dim=0)
        return avg_prediction
    
    def get_individual_predictions(self, images_dict: Dict[str, torch.Tensor], mask=None):
        """Get predictions from each individual model"""
        predictions = []
        
        for i, model in enumerate(self.models):
            with torch.no_grad():
                logits = model(images_dict, mask)
                predictions.append(logits)
                
        return predictions
    
    def train_individual_model(self, model_idx: int):
        """Set specific model to training mode, others to eval"""
        for i, model in enumerate(self.models):
            if i == model_idx:
                model.train()
            else:
                model.eval()
                
    def eval_all(self):
        """Set all models to evaluation mode"""
        for model in self.models:
            model.eval()


def train_ensemble_fold(
    train_loader, 
    val_loader, 
    device, 
    config,
    num_models: int = 3,
    epochs_per_model: int = 25
):
    """
    Train an ensemble of lightweight models with different seeds
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader  
        device: Device to train on
        config: Training configuration
        num_models: Number of models in ensemble
        epochs_per_model: Epochs to train each model
        
    Returns:
        Trained ensemble model and metrics
    """
    import torch.optim as optim
    from config import FocalLoss, calculate_class_weights
    from training.train_mm_k_fold import train_one_epoch, eval_model_with_threshold_optimization
    from utils.helpers import seed_everything
    
    ensemble = LightweightEnsemble(num_models=num_models).to(device)
    ensemble_metrics = []
    
    # Get class weights (assuming train_loader has labels available)
    train_labels = []
    for _, _, labels in train_loader:
        train_labels.extend(labels.numpy())
    class_weights = calculate_class_weights(train_labels).to(device)
    
    # Train each model individually with different seeds
    for model_idx in range(num_models):
        print(f"\n🎯 Training ensemble model {model_idx + 1}/{num_models}")
        
        # Set different seed for each model
        seed_everything(42 + model_idx * 100)
        
        # Get individual model
        model = ensemble.models[model_idx]
        
        # Setup training components
        criterion = FocalLoss(alpha=0.25, gamma=4.0, weight=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=3e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
        
        best_val_bal_acc = 0
        epochs_no_improve = 0
        best_model_state = None
        
        # Train individual model
        for epoch in range(1, epochs_per_model + 1):
            # Set only this model to training mode
            ensemble.train_individual_model(model_idx)
            
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device,
                use_mixup=True, mixup_alpha=0.2
            )
            
            val_loss, val_acc, val_bal, val_f1, val_auc, val_prec, val_rec, threshold = eval_model_with_threshold_optimization(
                model, val_loader, criterion, device, mc_dropout=True
            )
            
            scheduler.step(val_bal)
            
            print(f"  Epoch {epoch:02d}: Train: Loss {train_loss:.4f}, Acc {train_acc:.3f} | "
                  f"Val: BalAcc {val_bal:.3f}, F1 {val_f1:.3f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            if val_bal > best_val_bal_acc:
                best_val_bal_acc = val_bal
                best_model_state = model.state_dict().copy()
                epochs_no_improve = 0
                print(f"    ✅ New best for model {model_idx + 1}: {best_val_bal_acc:.3f}")
            else:
                epochs_no_improve += 1
                
            if epochs_no_improve >= 5:  # Early stopping
                print(f"    ⚠️ Early stopping model {model_idx + 1} after {epoch} epochs")
                break
        
        # Load best state for this model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            
        ensemble_metrics.append({
            'model_idx': model_idx,
            'best_val_bal_acc': best_val_bal_acc
        })
    
    # Set all models to eval mode
    ensemble.eval_all()
    
    return ensemble, ensemble_metrics


def evaluate_ensemble(ensemble, test_loader, device, optimal_threshold=0.5):
    """Evaluate ensemble model with TTA"""
    from utils.tta import tta_predict
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
    
    ensemble.eval_all()
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for images_dict, mask, labels in test_loader:
            images_dict = {k: v.to(device) for k, v in images_dict.items()}
            mask = mask.to(device)
            
            # Get ensemble prediction with TTA
            logits = tta_predict(ensemble, images_dict, mask)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
    
    # Apply threshold
    all_preds = (np.array(all_probs) >= optimal_threshold).astype(int)
    
    # Calculate metrics
    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'balanced_accuracy': balanced_accuracy_score(all_labels, all_preds),
        'f1_score': f1_score(all_labels, all_preds), 
        'precision': precision_score(all_labels, all_preds),
        'recall': recall_score(all_labels, all_preds),
        'auc': roc_auc_score(all_labels, all_probs),
        'confusion_matrix': confusion_matrix(all_labels, all_preds)
    }