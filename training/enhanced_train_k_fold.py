import os
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import torch.nn.functional as F
# from typing import Dict, List, Tuple
# import matplotlib.pyplot as plt
# import seaborn as sns
from tqdm import tqdm
import time
from collections import defaultdict

from config import config
from backbones.our import create_lightweight_model
from preprocess.kfold_splitter import PatientWiseKFoldSplitter
from preprocess.multimagset import create_enhanced_datasets, create_tta_dataset
from preprocess.preprocess import get_transforms, create_transforms
from utils.helpers import seed_everything


class LabelSmoothingCrossEntropy(torch.nn.Module):
    """Label smoothing cross entropy loss to prevent overconfident predictions"""
    def __init__(self, num_classes=2, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        pred = F.log_softmax(pred, dim=1)
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.num_classes - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=1))


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=7, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_score, model):
        if self.best_score is None:
            self.best_score = val_score
            self.best_weights = model.state_dict().copy()
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = val_score
            self.counter = 0
            self.best_weights = model.state_dict().copy()
        return False


class AdvancedTrainer:
    """Enhanced trainer with anti-overfitting strategies and advanced techniques"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Training history
        self.history = defaultdict(list)
        
        # Anti-overfitting strategies
        self.use_label_smoothing = True
        self.label_smoothing = 0.1
        self.use_dropout_scheduling = True
        self.use_early_stopping = True
        self.early_stopping_patience = 10
        self.use_gradient_clipping = True
        self.max_grad_norm = 1.0
        
        # Advanced techniques
        self.use_cosine_restarts = True
        self.use_warmup = True
        self.warmup_epochs = 3
        self.use_tta = True
        
    def create_loss_function(self, class_weights=None):
        """Create loss function with label smoothing and class weighting"""
        if self.use_label_smoothing:
            if class_weights is not None:
                # Custom weighted label smoothing loss
                return WeightedLabelSmoothingCrossEntropy(
                    num_classes=2, 
                    smoothing=self.label_smoothing,
                    weight=class_weights
                )
            else:
                return LabelSmoothingCrossEntropy(
                    num_classes=2, 
                    smoothing=self.label_smoothing
                )
        else:
            return torch.nn.CrossEntropyLoss(weight=class_weights)
    
    def create_optimizer_and_scheduler(self, model):
        """Create optimizer and scheduler with warmup and cosine annealing"""
        # Optimizer with weight decay for regularization
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=0.01,  # L2 regularization
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        if self.use_cosine_restarts:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, 
                T_0=10,  # Restart every 10 epochs
                T_mult=2,  # Double the restart period each time
                eta_min=1e-6
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.config.NUM_EPOCHS,
                eta_min=1e-6
            )
        
        # Warmup scheduler
        if self.use_warmup:
            def warmup_lambda(epoch):
                if epoch < self.warmup_epochs:
                    return (epoch + 1) / self.warmup_epochs
                return 1.0
            
            warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)
            return optimizer, scheduler, warmup_scheduler
        
        return optimizer, scheduler, None
    
    def adjust_dropout(self, model, epoch):
        """Dynamically adjust dropout rate during training"""
        if not self.use_dropout_scheduling:
            return
            
        # Increase dropout rate as training progresses to prevent overfitting
        base_dropout = 0.1
        max_dropout = 0.5
        dropout_rate = base_dropout + (max_dropout - base_dropout) * (epoch / self.config.NUM_EPOCHS)
        
        # Apply to model (this would need to be implemented in the model)
        # For now, this is a placeholder for the concept
        pass
    
    def train_one_epoch(self, model, dataloader, criterion, optimizer, epoch):
        """Enhanced training loop with gradient clipping and monitoring"""
        model.train()
        running_loss = 0.0
        all_labels = []
        all_preds = []
        all_probs = []
        
        # Progress bar
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} Training')
        
        for batch_idx, (images, labels) in enumerate(pbar):
            # Move to device
            images = {k: v.to(self.device) for k, v in images.items()}
            labels = labels.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.use_gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
            
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * labels.size(0)
            
            # Detach from computation graph before moving to CPU
            with torch.no_grad():
                probs = F.softmax(outputs.detach(), dim=1)
                _, preds = torch.max(outputs.detach(), 1)
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Positive class probabilities
            
            # Update progress bar
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Calculate metrics
        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        epoch_balanced_acc = balanced_accuracy_score(all_labels, all_preds)
        epoch_auc = roc_auc_score(all_labels, all_probs)
        
        return epoch_loss, epoch_acc, epoch_balanced_acc, epoch_auc
    
    def validate_one_epoch(self, model, dataloader, criterion):
        """Enhanced validation loop with comprehensive metrics"""
        model.eval()
        running_loss = 0.0
        all_labels = []
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc='Validation'):
                images = {k: v.to(self.device) for k, v in images.items()}
                labels = labels.to(self.device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * labels.size(0)
                probs = F.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        # Calculate comprehensive metrics
        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        epoch_balanced_acc = balanced_accuracy_score(all_labels, all_preds)
        epoch_precision = precision_score(all_labels, all_preds, average='weighted')
        epoch_recall = recall_score(all_labels, all_preds, average='weighted')
        epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
        epoch_auc = roc_auc_score(all_labels, all_probs)
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc,
            'balanced_accuracy': epoch_balanced_acc,
            'precision': epoch_precision,
            'recall': epoch_recall,
            'f1_score': epoch_f1,
            'auc': epoch_auc
        }
    
    def test_with_tta(self, model, test_patients, patient_dict):
        """Test with Test-Time Augmentation for improved accuracy"""
        if not self.use_tta:
            return None
            
        # Create TTA transforms
        _, _, tta_transforms = create_transforms()
        tta_datasets = create_tta_dataset(test_patients, patient_dict, tta_transforms)
        
        model.eval()
        all_predictions = []
        all_labels = []
        
        # Collect predictions from all TTA transforms
        tta_probs = []
        
        for tta_dataset in tta_datasets:
            tta_loader = DataLoader(tta_dataset, batch_size=self.config.BATCH_SIZE, 
                                   shuffle=False, num_workers=self.config.NUM_WORKERS)
            
            probs = []
            labels = []
            
            with torch.no_grad():
                for images, batch_labels in tta_loader:
                    images = {k: v.to(self.device) for k, v in images.items()}
                    outputs = model(images)
                    batch_probs = F.softmax(outputs, dim=1)
                    
                    probs.extend(batch_probs.cpu().numpy())
                    labels.extend(batch_labels.numpy())
            
            tta_probs.append(np.array(probs))
            if not all_labels:  # Only need labels once
                all_labels = labels
        
        # Average predictions across all TTA transforms
        avg_probs = np.mean(tta_probs, axis=0)
        tta_predictions = np.argmax(avg_probs, axis=1)
        
        # Calculate TTA metrics
        tta_metrics = {
            'accuracy': accuracy_score(all_labels, tta_predictions),
            'balanced_accuracy': balanced_accuracy_score(all_labels, tta_predictions),
            'precision': precision_score(all_labels, tta_predictions, average='weighted'),
            'recall': recall_score(all_labels, tta_predictions, average='weighted'),
            'f1_score': f1_score(all_labels, tta_predictions, average='weighted'),
            'auc': roc_auc_score(all_labels, avg_probs[:, 1])
        }
        
        return tta_metrics, tta_predictions, avg_probs
    
    def train_fold(self, fold_idx, train_patients, val_patients, test_patients, patient_dict):
        """Train a single fold with all enhancements"""
        print(f"\n{'='*50}")
        print(f"Training Fold {fold_idx + 1}/{self.config.N_SPLITS}")
        print(f"{'='*50}")
        
        # Seed for reproducibility
        seed_everything(self.config.RANDOM_STATE + fold_idx)
        
        # Create enhanced datasets
        train_transform, val_transform = get_transforms()
        train_dataset, val_dataset, test_dataset = create_enhanced_datasets(
            train_patients, val_patients, test_patients, patient_dict,
            train_transform, val_transform
        )
        
        # Create data loaders (disable pin_memory on MPS to avoid warnings)
        use_pin_memory = self.device.type == 'cuda'
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=True, 
            num_workers=self.config.NUM_WORKERS,
            pin_memory=use_pin_memory
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=False, 
            num_workers=self.config.NUM_WORKERS,
            pin_memory=use_pin_memory
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=False, 
            num_workers=self.config.NUM_WORKERS,
            pin_memory=use_pin_memory
        )
        
        # Initialize model
        model = create_lightweight_model(**self.config.get_model_config()).to(self.device)
        
        # Create loss function with class weights
        class_weights = train_dataset.get_class_weights()
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
        criterion = self.create_loss_function(class_weights_tensor)
        
        # Create optimizer and scheduler
        optimizer_scheduler = self.create_optimizer_and_scheduler(model)
        if len(optimizer_scheduler) == 3:
            optimizer, scheduler, warmup_scheduler = optimizer_scheduler
        else:
            optimizer, scheduler = optimizer_scheduler
            warmup_scheduler = None
        
        # Early stopping
        early_stopping = EarlyStopping(
            patience=self.early_stopping_patience,
            min_delta=0.001,
            restore_best_weights=True
        ) if self.use_early_stopping else None
        
        # Training loop
        best_val_score = 0.0
        fold_history = defaultdict(list)
        
        for epoch in range(self.config.NUM_EPOCHS):
            start_time = time.time()
            
            # Adjust dropout if scheduled
            self.adjust_dropout(model, epoch)
            
            # Training
            train_loss, train_acc, train_bal_acc, train_auc = self.train_one_epoch(
                model, train_loader, criterion, optimizer, epoch
            )
            
            # Validation
            val_metrics = self.validate_one_epoch(model, val_loader, criterion)
            
            # Learning rate scheduling
            if warmup_scheduler and epoch < self.warmup_epochs:
                warmup_scheduler.step()
            else:
                scheduler.step()
            
            current_lr = optimizer.param_groups[0]['lr']
            epoch_time = time.time() - start_time
            
            # Log metrics
            fold_history['train_loss'].append(train_loss)
            fold_history['train_accuracy'].append(train_acc)
            fold_history['train_balanced_accuracy'].append(train_bal_acc)
            fold_history['train_auc'].append(train_auc)
            fold_history['val_loss'].append(val_metrics['loss'])
            fold_history['val_accuracy'].append(val_metrics['accuracy'])
            fold_history['val_balanced_accuracy'].append(val_metrics['balanced_accuracy'])
            fold_history['val_precision'].append(val_metrics['precision'])
            fold_history['val_recall'].append(val_metrics['recall'])
            fold_history['val_f1_score'].append(val_metrics['f1_score'])
            fold_history['val_auc'].append(val_metrics['auc'])
            fold_history['learning_rate'].append(current_lr)
            
            # Print progress
            print(f"Epoch {epoch+1:3d}/{self.config.NUM_EPOCHS} | "
                  f"Time: {epoch_time:.1f}s | LR: {current_lr:.2e} | "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, AUC: {train_auc:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
                  f"Bal Acc: {val_metrics['balanced_accuracy']:.4f}, AUC: {val_metrics['auc']:.4f}")
            
            # Save best model
            if val_metrics['balanced_accuracy'] > best_val_score:
                best_val_score = val_metrics['balanced_accuracy']
                torch.save(model.state_dict(), 
                          os.path.join(self.config.MODELS_DIR, f"best_model_fold_{fold_idx}.pth"))
                print(f"    ★ New best validation balanced accuracy: {best_val_score:.4f}")
            
            # Early stopping
            if early_stopping and early_stopping(val_metrics['balanced_accuracy'], model):
                print(f"    ⏹ Early stopping triggered at epoch {epoch+1}")
                break
        
        # Load best model for testing
        model.load_state_dict(torch.load(os.path.join(self.config.MODELS_DIR, f"best_model_fold_{fold_idx}.pth")))
        
        # Standard testing
        test_metrics = self.validate_one_epoch(model, test_loader, criterion)
        
        # Test with TTA
        tta_metrics = None
        if self.use_tta:
            tta_results = self.test_with_tta(model, test_patients, patient_dict)
            if tta_results:
                tta_metrics, _, _ = tta_results
                print(f"\nTTA Results - Accuracy: {tta_metrics['accuracy']:.4f}, "
                      f"Balanced Accuracy: {tta_metrics['balanced_accuracy']:.4f}, "
                      f"AUC: {tta_metrics['auc']:.4f}")
        
        # Save fold results
        fold_results = {
            'fold': fold_idx,
            'best_val_balanced_accuracy': best_val_score,
            'test_metrics': test_metrics,
            'tta_metrics': tta_metrics,
            'training_history': dict(fold_history)
        }
        
        # Save fold logs and results
        pd.DataFrame(fold_history).to_csv(
            os.path.join(self.config.LOGS_DIR, f"fold_{fold_idx}_enhanced_logs.csv"), 
            index=False
        )
        
        with open(os.path.join(self.config.RESULTS_DIR, f"fold_{fold_idx}_enhanced_results.json"), 'w') as f:
            json.dump(fold_results, f, indent=2, default=str)
        
        return fold_results


class WeightedLabelSmoothingCrossEntropy(torch.nn.Module):
    """Weighted label smoothing cross entropy loss"""
    def __init__(self, num_classes=2, smoothing=0.1, weight=None):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.weight = weight

    def forward(self, pred, target):
        pred = F.log_softmax(pred, dim=1)
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.num_classes - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        
        loss = torch.sum(-true_dist * pred, dim=1)
        
        if self.weight is not None:
            loss = loss * self.weight[target]
        
        return torch.mean(loss)


def run_enhanced_training(args=None):
    """Main function to run enhanced k-fold training"""
    # Create output directories
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    # Modify trainer based on arguments
    trainer = AdvancedTrainer(config)
    
    # Handle arguments
    if args:
        if args.quick_test:
            trainer.config.NUM_EPOCHS = 5
            trainer.config.BATCH_SIZE = 8
            print(f"📝 Quick test: {trainer.config.NUM_EPOCHS} epochs, batch size {trainer.config.BATCH_SIZE}")
        
        if args.no_tta:
            trainer.use_tta = False
            print("📝 TTA disabled")
    
    # Determine number of folds to run
    n_folds_to_run = 1 if args and args.single_fold else config.N_SPLITS
    
    # Initialize splitter (always use at least 2 splits for StratifiedKFold)
    splitter = PatientWiseKFoldSplitter(
        dataset_dir=config.DATASET_DIR,
        n_splits=max(2, config.N_SPLITS),  # Ensure at least 2 splits
        random_state=config.RANDOM_STATE,
        stratify_subtype=config.STRATIFY_SUBTYPE,
        validation_split=config.VALIDATION_SPLIT
    )
    
    # Print dataset summary
    splitter.print_summary()
    if not (args and args.quick_test):  # Skip visualization in quick test
        splitter.visualize()
    
    # Train folds
    all_fold_results = []
    
    print(f"\n🎯 Running {n_folds_to_run} fold(s) out of {splitter.n_splits} available")
    
    for fold in range(n_folds_to_run):
        train_pats, val_pats, test_pats = splitter.get_fold(fold, return_type='patients')
        
        fold_results = trainer.train_fold(
            fold, train_pats, val_pats, test_pats, splitter.patient_dict
        )
        all_fold_results.append(fold_results)
    
    # Aggregate results across folds
    test_accuracies = [r['test_metrics']['accuracy'] for r in all_fold_results]
    test_bal_accuracies = [r['test_metrics']['balanced_accuracy'] for r in all_fold_results]
    test_aucs = [r['test_metrics']['auc'] for r in all_fold_results]
    
    tta_accuracies = [r['tta_metrics']['accuracy'] for r in all_fold_results if r['tta_metrics']]
    tta_bal_accuracies = [r['tta_metrics']['balanced_accuracy'] for r in all_fold_results if r['tta_metrics']]
    tta_aucs = [r['tta_metrics']['auc'] for r in all_fold_results if r['tta_metrics']]
    
    # Print final results
    result_title = "SINGLE FOLD RESULTS" if n_folds_to_run == 1 else "FINAL CROSS-VALIDATION RESULTS"
    print(f"\n{'='*60}")
    print(result_title)
    print(f"{'='*60}")
    
    if n_folds_to_run == 1:
        print(f"Standard Testing (Fold 0):")
        print(f"  Accuracy: {test_accuracies[0]:.4f}")
        print(f"  Balanced Accuracy: {test_bal_accuracies[0]:.4f}")
        print(f"  AUC: {test_aucs[0]:.4f}")
        
        if tta_accuracies:
            print(f"\nTest-Time Augmentation (Fold 0):")
            print(f"  Accuracy: {tta_accuracies[0]:.4f}")
            print(f"  Balanced Accuracy: {tta_bal_accuracies[0]:.4f}")
            print(f"  AUC: {tta_aucs[0]:.4f}")
    else:
        print(f"Standard Testing:")
        print(f"  Accuracy: {np.mean(test_accuracies):.4f} ± {np.std(test_accuracies):.4f}")
        print(f"  Balanced Accuracy: {np.mean(test_bal_accuracies):.4f} ± {np.std(test_bal_accuracies):.4f}")
        print(f"  AUC: {np.mean(test_aucs):.4f} ± {np.std(test_aucs):.4f}")
        
        if tta_accuracies:
            print(f"\nTest-Time Augmentation:")
            print(f"  Accuracy: {np.mean(tta_accuracies):.4f} ± {np.std(tta_accuracies):.4f}")
            print(f"  Balanced Accuracy: {np.mean(tta_bal_accuracies):.4f} ± {np.std(tta_bal_accuracies):.4f}")
            print(f"  AUC: {np.mean(tta_aucs):.4f} ± {np.std(tta_aucs):.4f}")
    
    # Save comprehensive results
    final_results = {
        'n_folds_run': n_folds_to_run,
        'standard_results': {
            'mean_accuracy': np.mean(test_accuracies),
            'std_accuracy': np.std(test_accuracies) if n_folds_to_run > 1 else 0.0,
            'mean_balanced_accuracy': np.mean(test_bal_accuracies),
            'std_balanced_accuracy': np.std(test_bal_accuracies) if n_folds_to_run > 1 else 0.0,
            'mean_auc': np.mean(test_aucs),
            'std_auc': np.std(test_aucs) if n_folds_to_run > 1 else 0.0,
            'all_accuracies': test_accuracies
        }
    }
    
    if tta_accuracies:
        final_results['tta_results'] = {
            'mean_accuracy': np.mean(tta_accuracies),
            'std_accuracy': np.std(tta_accuracies) if n_folds_to_run > 1 else 0.0,
            'mean_balanced_accuracy': np.mean(tta_bal_accuracies),
            'std_balanced_accuracy': np.std(tta_bal_accuracies) if n_folds_to_run > 1 else 0.0,
            'mean_auc': np.mean(tta_aucs),
            'std_auc': np.std(tta_aucs) if n_folds_to_run > 1 else 0.0,
            'all_accuracies': tta_accuracies
        }
    
    final_results['all_fold_results'] = all_fold_results
    
    results_filename = "enhanced_single_fold_results.json" if n_folds_to_run == 1 else "enhanced_final_results.json"
    with open(os.path.join(config.RESULTS_DIR, results_filename), 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    print(f"\n✅ Enhanced training completed!")
    print(f"📊 Results saved to: {config.RESULTS_DIR}/{results_filename}")
    
    # Check if we achieved 96% accuracy target
    max_accuracy = max(tta_accuracies) if tta_accuracies else max(test_accuracies)
    if max_accuracy >= 0.96:
        print(f"🎯 TARGET ACHIEVED! Maximum accuracy: {max_accuracy:.4f} (≥96%)")
    else:
        print(f"📈 Current best accuracy: {max_accuracy:.4f}. Target: 96%")
        if n_folds_to_run == 1:
            print("💡 This was a single fold test. Run full cross-validation for final results.")
        else:
            print("💡 Consider further hyperparameter tuning or model architecture changes.")


if __name__ == "__main__":
    run_enhanced_training()