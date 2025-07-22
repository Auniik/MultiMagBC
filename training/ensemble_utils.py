import torch
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
import os

class MMEEnsemble:
    """Simple ensemble for MMNet using model averaging"""
    
    def __init__(self, model_class, device=torch.device('cuda')):
        self.models = []
        self.weights = []
        self.device = device
        self.model_class = model_class
    
    def add_model(self, model_path, weight=1.0):
        """Add a model to the ensemble"""
        model = self.model_class()
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        self.models.append(model)
        self.weights.append(weight)
    
    def predict(self, images_dict):
        """Predict using ensemble averaging"""
        if not self.models:
            raise ValueError("No models in ensemble")
        
        # Normalize weights
        weights = np.array(self.weights) / np.sum(self.weights)
        
        # Get predictions from all models
        all_probs = []
        with torch.no_grad():
            for model, weight in zip(self.models, weights):
                logits, _ = model(images_dict)
                probs = torch.softmax(logits, dim=1)[:, 1]
                all_probs.append(probs.cpu().numpy() * weight)
        
        # Weighted average
        ensemble_probs = np.sum(all_probs, axis=0)
        return ensemble_probs
    
    def evaluate(self, dataloader, threshold=0.5):
        """Evaluate ensemble on test set"""
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for images_dict, labels in dataloader:
                images = {k: v.to(self.device) for k, v in images_dict.items()}
                labels = labels.numpy()
                
                probs = self.predict(images)
                all_probs.extend(probs)
                all_labels.extend(labels)
        
        # Calculate metrics
        all_preds = (np.array(all_probs) >= threshold).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'balanced_accuracy': balanced_accuracy_score(all_labels, all_preds),
            'f1_score': f1_score(all_labels, all_preds),
            'auc_score': roc_auc_score(all_labels, all_probs)
        }
        
        return metrics, all_probs, all_labels

class CrossFoldEnsemble:
    """Ensemble model using top-performing folds"""
    
    def __init__(self, model_class, output_dir='./output/models', top_k=3):
        self.model_class = model_class
        self.output_dir = output_dir
        self.top_k = top_k
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def create_ensemble(self, fold_results):
        """Create ensemble from top-k performing folds"""
        ensemble = MMEEnsemble(self.model_class, self.device)
        
        # Sort folds by validation balanced accuracy
        sorted_folds = sorted(fold_results.items(), 
                            key=lambda x: x[1]['val_bal_acc'], 
                            reverse=True)[:self.top_k]
        
        for fold_idx, results in sorted_folds:
            model_path = os.path.join(self.output_dir, f'best_model_fold_{fold_idx}.pth')
            if os.path.exists(model_path):
                weight = results['val_bal_acc']  # Use validation performance as weight
                ensemble.add_model(model_path, weight)
        
        return ensemble
    
    def test_time_augmentation(self, model, dataloader, num_augmentations=5):
        """Test-time augmentation for single model"""
        all_probs = []
        all_labels = []
        
        for aug_idx in range(num_augmentations):
            probs = []
            labels = []
            
            with torch.no_grad():
                for images_dict, batch_labels in dataloader:
                    images = {k: v.to(self.device) for k, v in images_dict.items()}
                    
                    # Apply random augmentations
                    if aug_idx > 0:  # Skip original for first iteration
                        images = self._apply_test_augmentation(images)
                    
                    logits, _ = model(images)
                    batch_probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                    probs.extend(batch_probs)
                    labels.extend(batch_labels.numpy())
            
            all_probs.append(np.array(probs))
            # Use labels from last augmentation (they're the same across augmentations)
            all_labels = labels
        
        # Average predictions across augmentations
        ensemble_probs = np.mean(all_probs, axis=0)
        
        # Calculate metrics
        all_preds = (ensemble_probs >= 0.5).astype(int)
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'balanced_accuracy': balanced_accuracy_score(all_labels, all_preds),
            'f1_score': f1_score(all_labels, all_preds),
            'auc_score': roc_auc_score(all_labels, ensemble_probs)
        }
        
        return metrics, ensemble_probs
    
    def _apply_test_augmentation(self, images):
        """Apply test-time augmentation"""
        augmented = {}
        for key, tensor in images.items():
            # Random horizontal flip
            if torch.rand(1) > 0.5:
                tensor = torch.flip(tensor, dims=[3])
            
            # Random rotation
            if torch.rand(1) > 0.5:
                tensor = torch.rot90(tensor, k=1, dims=[2, 3])
            
            augmented[key] = tensor
        return augmented

def create_snapshot_ensemble(model, optimizer, save_dir, num_snapshots=3):
    """Create snapshot ensemble by saving checkpoints at different training stages"""
    snapshot_paths = []
    
    for i in range(num_snapshots):
        # Save model snapshot
        snapshot_path = os.path.join(save_dir, f'snapshot_{i}.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': i
        }, snapshot_path)
        snapshot_paths.append(snapshot_path)
    
    return snapshot_paths