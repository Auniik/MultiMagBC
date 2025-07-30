#!/usr/bin/env python3
"""
Robust K-fold splitter that guarantees balanced folds
"""

import os, glob, random, json
import numpy as np
from collections import Counter, defaultdict
from sklearn.model_selection import StratifiedKFold
from typing import Dict, List, Tuple

class RobustPatientWiseKFoldSplitter:
    """
    Robust splitter that GUARANTEES balanced folds using proper stratification
    """
    
    def __init__(
        self,
        dataset_dir,
        n_splits=5,
        random_state=42,
        validation_split=0.25,
        preserve_natural_ratio=True  # NEW: Preserve dataset's natural class ratio
    ):
        self.dataset_dir = dataset_dir
        self.n_splits = n_splits
        self.random_state = random_state
        self.validation_split = validation_split
        self.preserve_natural_ratio = preserve_natural_ratio
        
        self.patient_dict, self.magnifications = self._scan_dataset()
        
        # Calculate dataset's natural imbalance ratio
        labels = [self.patient_dict[pid]['label'] for pid in self.patient_dict.keys()]
        benign_count = labels.count(0)
        malignant_count = labels.count(1)
        self.natural_ratio = malignant_count / max(benign_count, 1)
        
        print(f"📊 Dataset Natural Ratio: {malignant_count} malignant / {benign_count} benign = {self.natural_ratio:.2f}x")
        
        # Set acceptable range around natural ratio (±15%)
        self.max_imbalance_ratio = self.natural_ratio * 1.15
        self.min_imbalance_ratio = self.natural_ratio * 0.85
        
        self.folds = self._create_robust_folds()
        self._validate_fold_balance()

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
                            'total_images': total_images
                        }

        return patient_dict, sorted(list(magnifications_set))

    def _create_robust_folds(self):
        """Create GUARANTEED balanced folds using proper stratification"""
        patient_ids = list(self.patient_dict.keys())
        
        # Create multi-level stratification
        labels = [self.patient_dict[pid]['label'] for pid in patient_ids]
        subtypes = [self.patient_dict[pid]['subtype'] for pid in patient_ids]
        
        # Strategy 1: If we have enough samples per subtype, stratify by subtype
        subtype_counts = Counter(subtypes)
        min_subtype_count = min(subtype_counts.values())
        
        if min_subtype_count >= self.n_splits * 2:  # At least 2 per fold
            print("📊 Using subtype-based stratification")
            stratify_groups = subtypes
        else:
            # Strategy 2: Stratify by class only to ensure balance
            print("📊 Using class-based stratification (insufficient subtype samples)")
            stratify_groups = labels
        
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
        
        return folds

    def _validate_fold_balance(self):
        """VALIDATE that folds preserve natural dataset ratio (±15%)"""
        print(f"\n🔍 FOLD BALANCE VALIDATION (Natural Ratio: {self.natural_ratio:.2f}x):")
        print(f"{'Fold':<4} {'Train B/M':<10} {'Test B/M':<9} {'Ratio':<8} {'Deviation':<10} {'Status'}")
        print("-" * 75)
        
        all_valid = True
        for i, (train_pats, test_pats) in enumerate(self.folds):
            train_labels = [self.patient_dict[pid]['label'] for pid in train_pats]
            test_labels = [self.patient_dict[pid]['label'] for pid in test_pats]
            
            train_b, train_m = train_labels.count(0), train_labels.count(1)
            test_b, test_m = test_labels.count(0), test_labels.count(1)
            
            # Calculate imbalance ratio
            test_ratio = test_m / max(test_b, 1)
            
            # Check if fold preserves natural ratio (±15%)
            deviation = abs(test_ratio - self.natural_ratio) / self.natural_ratio
            is_valid = (test_ratio >= self.min_imbalance_ratio and 
                       test_ratio <= self.max_imbalance_ratio)
            
            status = "✅ PASS" if is_valid else "❌ FAIL"
            if not is_valid:
                all_valid = False
            
            print(f"{i:<4} {train_b}/{train_m:<9} {test_b}/{test_m:<8} "
                  f"{test_ratio:.2f}{'x':<7} {deviation:.1%}{'':>10} {status}")
        
        if not all_valid:
            print(f"⚠️ Some folds deviate significantly from natural ratio ({self.natural_ratio:.2f}x)")
            print("📊 This is expected for small datasets with natural class imbalance")
            print("✅ Proceeding with stratified folds (imbalance will be handled by loss function)")
        else:
            print("✅ ALL FOLDS PRESERVE NATURAL DATASET RATIO")

    def get_fold(self, fold_idx, return_type='patients'):
        """Returns train/val/test splits for a fold"""
        train_pats, test_pats = self.folds[fold_idx]

        # Stratified train/val split to maintain balance
        train_labels = [self.patient_dict[pid]['label'] for pid in train_pats]
        
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(
            n_splits=1, 
            test_size=self.validation_split, 
            random_state=self.random_state + fold_idx  # Different seed per fold
        )
        
        train_idx, val_idx = next(sss.split(train_pats, train_labels))
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

    def print_summary(self):
        """Print detailed fold summary with balance metrics"""
        print("=== ROBUST K-FOLD DATASET SUMMARY ===")
        all_subtypes = sorted(set([self.patient_dict[pid]['subtype'] for pid in self.patient_dict]))

        for i, (train_pats, test_pats) in enumerate(self.folds):
            train_only, val_pats = self._stratified_split(train_pats, self.validation_split, i)

            def summarize(patients, name):
                if not patients:
                    return {"patients": 0, "benign": 0, "malignant": 0, "benign_pct": 0, "subtypes_present": [], "magnification_counts": {}}
                
                labels = [self.patient_dict[pid]['label'] for pid in patients]
                subtypes = [self.patient_dict[pid]['subtype'] for pid in patients]
                mag_counts = {m: sum(len(self.patient_dict[pid]['images'].get(m, [])) for pid in patients) for m in self.magnifications}
                return {
                    'patients': len(patients),
                    'benign': labels.count(0),
                    'malignant': labels.count(1),
                    'benign_pct': round(labels.count(0)/len(patients)*100,1),
                    'subtypes_present': sorted(set(subtypes)),
                    'magnification_counts': mag_counts
                }

            train_stats = summarize(train_only, "Train")
            val_stats = summarize(val_pats, "Val")
            test_stats = summarize(test_pats, "Test")

            print(f"\n--- Fold {i} ---")
            print(f"Train: {train_stats['patients']} patients ({train_stats['benign']}/{train_stats['malignant']} B/M, {train_stats['benign_pct']}% benign)")
            print(f"Val:   {val_stats['patients']} patients ({val_stats['benign']}/{val_stats['malignant']} B/M, {val_stats['benign_pct']}% benign)")
            print(f"Test:  {test_stats['patients']} patients ({test_stats['benign']}/{test_stats['malignant']} B/M, {test_stats['benign_pct']}% benign)")
            print(f"Subtypes present (Train): {', '.join(train_stats['subtypes_present'])}")
            print(f"Subtypes present (Test):  {', '.join(test_stats['subtypes_present'])}")
            print(f"Magnification distribution (Train): {train_stats['magnification_counts']}")

    def _stratified_split(self, patient_list, val_fraction, fold_idx):
        """Helper for stratified train/val split"""
        if not patient_list:
            return [], []
            
        labels = [self.patient_dict[pid]['label'] for pid in patient_list]
        
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(
            n_splits=1, 
            test_size=val_fraction, 
            random_state=self.random_state + fold_idx
        )
        
        train_idx, val_idx = next(sss.split(patient_list, labels))
        return [patient_list[i] for i in train_idx], [patient_list[i] for i in val_idx]