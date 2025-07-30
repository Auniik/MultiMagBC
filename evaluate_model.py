#!/usr/bin/env python3
"""
Model Evaluation Script
- Loads trained fold checkpoints
- Evaluates with TTA
- Computes per-subtype metrics
- Aggregates fold results
"""

import os, json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from utils.stats import save_as_json
from utils.metrics import per_subtype_metrics
from utils.tta import evaluate_with_tta
from preprocess.kfold_splitter import PatientWiseKFoldSplitter
from preprocess.multimagset import MultiMagPatientDataset
from preprocess.preprocess_light import get_fast_transforms as get_transforms
from backbones.our.lightweight import MultiMagLightweightCNN
from config import SLIDES_PATH, OUTPUT_DIR, get_training_config

def load_model(path, device):
    model = MultiMagLightweightCNN(num_classes=2)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

def evaluate_all_folds():
    config = get_training_config()
    device = config['device']
    _, eval_transform, _ = get_transforms()
    
    splitter = PatientWiseKFoldSplitter(
        dataset_dir=SLIDES_PATH,
        n_splits=5,
        stratify_subtype=True
    )
    patient_dict = splitter.patient_dict
    
    results_dir = os.path.join(OUTPUT_DIR, "evaluation")
    os.makedirs(results_dir, exist_ok=True)
    
    all_metrics = []
    subtype_metrics_all = {}
    
    for fold_idx in range(len(splitter.folds)):
        print(f"\n=== Evaluating Fold {fold_idx} ===")
        _, _, test_pats = splitter.get_fold(fold_idx)
        test_ds = MultiMagPatientDataset(patient_dict, test_pats, transform=eval_transform, mode='test', sampling_mode='relaxed')
        test_loader = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
        
        model_path = os.path.join(OUTPUT_DIR, "models", f"best_model_fold_{fold_idx}.pth")
        model = load_model(model_path, device)
        
        metrics = evaluate_with_tta(model, test_loader, device, optimal_threshold=0.5, return_raw=True)
        subtype_perf = per_subtype_metrics(model, test_loader, device, 0.5, patient_dict)
        
        fold_result = {
            "fold": fold_idx,
            "overall": {k: metrics[k] for k in ['accuracy','balanced_accuracy','f1_score','precision','recall','auc']},
            "subtypes": subtype_perf
        }
        save_as_json(fold_result, os.path.join(results_dir, f"fold_{fold_idx}_evaluation.json"))
        all_metrics.append(fold_result["overall"])
        
        # Aggregate subtype metrics
        for subtype, vals in subtype_perf.items():
            if subtype not in subtype_metrics_all:
                subtype_metrics_all[subtype] = {k: [] for k in vals.keys()}
            for k, v in vals.items():
                subtype_metrics_all[subtype][k].append(v)
    
    # Aggregate across folds
    def aggregate_metric(metric_list):
        arr = np.array(metric_list)
        return {"mean": float(np.mean(arr)), "std": float(np.std(arr))}
    
    summary = {
        "overall": {k: aggregate_metric([m[k] for m in all_metrics]) for k in all_metrics[0].keys()},
        "subtypes": {sub: {k: aggregate_metric(v) for k, v in metrics.items()} for sub, metrics in subtype_metrics_all.items()}
    }
    save_as_json(summary, os.path.join(results_dir, "cross_fold_summary.json"))
    print(f"\n=== Cross-Fold Summary ===")
    print(json.dumps(summary["overall"], indent=2))
    print(f"\nDetailed results saved in {results_dir}")

if __name__ == "__main__":
    evaluate_all_folds()