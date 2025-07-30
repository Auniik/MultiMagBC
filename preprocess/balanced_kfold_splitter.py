#!/usr/bin/env python3
"""
Improved K-fold splitter that balances patient data distribution across folds
"""

import os, glob, random, json
import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

class BalancedPatientWiseKFoldSplitter:
    """
    Improved splitter that considers both class balance AND data volume balance
    """
    
    def __init__(
        self,
        dataset_dir,
        n_splits=5,
        random_state=42,
        validation_split=0.25,
        balance_by_images=True  # New parameter
    ):
        self.dataset_dir = dataset_dir
        self.n_splits = n_splits
        self.random_state = random_state
        self.validation_split = validation_split
        self.balance_by_images = balance_by_images
        
        self.patient_dict, self.magnifications = self._scan_dataset()
        self.folds = self._create_balanced_folds()

    def _scan_dataset(self):
        """Scan dataset and collect patient-level metadata"""
        patient_dict = {}
        magnifications_set = set()

        for cls in os.listdir(self.dataset_dir):
            cls_dir = os.path.join(self.dataset_dir, cls)
            if not os.path.isdir(cls_dir): continue
            label = 0 if cls.lower().startswith('benign') else 1

            for hospital in os.listdir(cls_dir):
                hosp_dir = os.path.join(cls_dir, hospital)
                if not os.path.isdir(hosp_dir): continue

                for subtype in os.listdir(hosp_dir):
                    sub_dir = os.path.join(hosp_dir, subtype)
                    if not os.path.isdir(sub_dir): continue

                    for patient in os.listdir(sub_dir):
                        pat_dir = os.path.join(sub_dir, patient)
                        if not os.path.isdir(pat_dir): continue

                        pid_key = f"{cls}_{hospital}_{subtype}_{patient}"
                        mag_images = {}

                        for mag in os.listdir(pat_dir):
                            mag_name = mag.replace('X', '').replace('x', '')
                            mag_dir = os.path.join(pat_dir, mag)
                            if not os.path.isdir(mag_dir): continue
                            magnifications_set.add(mag_name)
                            images = []
                            for ext in ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff'):
                                images.extend(glob.glob(os.path.join(mag_dir, ext)))
                            if images:
                                mag_images[mag_name] = images

                        if not any(mag_images.values()):
                            continue

                        total_images = sum(len(imgs) for imgs in mag_images.values())
                        patient_dict[pid_key] = {
                            'label': label,
                            'subtype': subtype,
                            'images': mag_images,
                            'total_images': total_images  # Track for balancing
                        }

        return patient_dict, sorted(list(magnifications_set))

    def _create_balanced_folds(self):
        """Create folds balanced by both class and data volume"""
        patient_ids = list(self.patient_dict.keys())
        
        if self.balance_by_images:
            # Create stratification groups based on class + image volume
            image_counts = [self.patient_dict[pid]['total_images'] for pid in patient_ids]
            labels = [self.patient_dict[pid]['label'] for pid in patient_ids]
            
            # Categorize patients by image volume (terciles)
            image_thresholds = np.percentile(image_counts, [33, 67])
            
            stratify_groups = []
            for pid in patient_ids:
                label = self.patient_dict[pid]['label']
                img_count = self.patient_dict[pid]['total_images']
                
                # Volume category: 0=low, 1=medium, 2=high
                if img_count <= image_thresholds[0]:
                    vol_cat = 0
                elif img_count <= image_thresholds[1]:
                    vol_cat = 1
                else:
                    vol_cat = 2
                
                # Combined stratification: class_volume (e.g., "0_1" = benign_medium)
                stratify_groups.append(f"{label}_{vol_cat}")
            
            print(f"📊 Stratification groups: {Counter(stratify_groups)}")
            
        else:
            # Simple class-based stratification
            stratify_groups = [self.patient_dict[pid]['label'] for pid in patient_ids]

        # Create stratified k-fold splits
        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state
        )
        
        folds = []
        for train_idx, test_idx in skf.split(patient_ids, stratify_groups):
            train_pats = [patient_ids[i] for i in train_idx]
            test_pats = [patient_ids[i] for i in test_idx]
            folds.append((train_pats, test_pats))
        
        # Verify balance
        self._verify_fold_balance(folds)
        
        return folds

    def _verify_fold_balance(self, folds):
        """Verify that folds are balanced"""
        print(f"\n📊 Fold Balance Verification:")
        print(f"{'Fold':<4} {'Train B/M':<10} {'Test B/M':<9} {'Train Imgs':<11} {'Test Imgs':<10} {'Balance'}")
        print("-" * 70)
        
        for i, (train_pats, test_pats) in enumerate(folds):
            # Class balance
            train_labels = [self.patient_dict[pid]['label'] for pid in train_pats]
            test_labels = [self.patient_dict[pid]['label'] for pid in test_pats]
            
            train_b, train_m = train_labels.count(0), train_labels.count(1)
            test_b, test_m = test_labels.count(0), test_labels.count(1)
            
            # Image balance
            train_imgs = sum(self.patient_dict[pid]['total_images'] for pid in train_pats)
            test_imgs = sum(self.patient_dict[pid]['total_images'] for pid in test_pats)
            
            # Balance score (lower is better)
            train_ratio = train_m / max(train_b, 1)
            test_ratio = test_m / max(test_b, 1)
            ratio_diff = abs(train_ratio - test_ratio)
            
            balance_score = "Good" if ratio_diff < 0.3 else "Poor"
            
            print(f"{i:<4} {train_b}/{train_m:<9} {test_b}/{test_m:<8} "
                  f"{train_imgs:<11} {test_imgs:<10} {balance_score}")

    def get_fold(self, fold_idx, return_type='patients'):
        """Returns train/val/test splits for a fold"""
        train_pats, test_pats = self.folds[fold_idx]

        # Stratified train/val split (preserving balance)
        if self.balance_by_images:
            # Balance both class and images for train/val split
            train_labels = [self.patient_dict[pid]['label'] for pid in train_pats]
            train_images = [self.patient_dict[pid]['total_images'] for pid in train_pats]
            
            # Create balanced train/val stratification
            image_threshold = np.median(train_images)
            stratify_val = []
            for pid in train_pats:
                label = self.patient_dict[pid]['label']
                imgs = self.patient_dict[pid]['total_images']
                vol_cat = 0 if imgs <= image_threshold else 1
                stratify_val.append(f"{label}_{vol_cat}")
        else:
            stratify_val = [self.patient_dict[pid]['label'] for pid in train_pats]

        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(
            n_splits=1, 
            test_size=self.validation_split, 
            random_state=self.random_state
        )
        
        train_idx, val_idx = next(sss.split(train_pats, stratify_val))
        val_pats = [train_pats[i] for i in val_idx]
        train_pats = [train_pats[i] for i in train_idx]

        if return_type == 'patients':
            return train_pats, val_pats, test_pats
        elif return_type == 'files':
            def flatten(pat_list):
                return [
                    f for pid in pat_list
                    for mag_files in self.patient_dict[pid]['images'].values()
                    for f in mag_files
                ]
            return flatten(train_pats), flatten(val_pats), flatten(test_pats)
        else:
            raise ValueError("return_type must be 'patients' or 'files'")

    def get_splits(self, return_type='patients'):
        """Returns a list of (train, val, test) for all folds"""
        return [self.get_fold(i, return_type=return_type) for i in range(self.n_splits)]

    def print_summary(self):
        """Print detailed fold summary"""
        print("=== Balanced K-Fold Dataset Summary ===")
        for i, (train_pats, test_pats) in enumerate(self.folds):
            train_labels = [self.patient_dict[pid]['label'] for pid in train_pats]
            test_labels = [self.patient_dict[pid]['label'] for pid in test_pats]
            train_images = sum(self.patient_dict[pid]['total_images'] for pid in train_pats)
            test_images = sum(self.patient_dict[pid]['total_images'] for pid in test_pats)
            
            print(f"Fold {i}: Train patients: {len(train_pats)} (images={train_images}, B/M = {train_labels.count(0)}/{train_labels.count(1)}); "
                  f"Test patients: {len(test_pats)} (images={test_images}, B/M = {test_labels.count(0)}/{test_labels.count(1)})")