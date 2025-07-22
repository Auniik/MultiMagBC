import os
import glob
import pandas as pd
from sklearn.model_selection import StratifiedKFold


class PatientWiseKFoldSplitter:
    """
    Splits a histopathology dataset into patient-wise K-fold splits.

    Assumes directory structure:
    dataset_dir/
        benign/
            <hospital>/
                <subtype>/
                    <patient_id>/
                        <magnification>/
                            image files...
        malignant/
            ...
    """
    def __init__(
        self,
        dataset_dir,
        n_splits=5,
        random_state=42,
        stratify_subtype=False
    ):
        self.dataset_dir = dataset_dir
        self.n_splits = n_splits
        self.random_state = random_state
        self.stratify_subtype = stratify_subtype
        self.patient_dict = self._scan_dataset()
        self.folds = self._create_folds()

    def _scan_dataset(self):
        patient_dict = {}
        for cls in os.listdir(self.dataset_dir):
            cls_dir = os.path.join(self.dataset_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            # Binary label: benign=0, malignant=1
            label = 0 if cls.lower().startswith('benign') else 1
            for hospital in os.listdir(cls_dir):
                hosp_dir = os.path.join(cls_dir, hospital)
                if not os.path.isdir(hosp_dir):
                    continue
                for subtype in os.listdir(hosp_dir):
                    sub_dir = os.path.join(hosp_dir, subtype)
                    if not os.path.isdir(sub_dir):
                        continue
                    for patient in os.listdir(sub_dir):
                        pat_dir = os.path.join(sub_dir, patient)
                        if not os.path.isdir(pat_dir):
                            continue
                        # Collect all image file paths under patient
                        images = []
                        for mag in os.listdir(pat_dir):
                            mag_dir = os.path.join(pat_dir, mag)
                            if not os.path.isdir(mag_dir):
                                continue
                            for ext in ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff'):
                                images.extend(glob.glob(os.path.join(mag_dir, ext)))
                        if not images:
                            continue
                        patient_dict[patient] = {
                            'label': label,
                            'subtype': subtype,
                            'images': images
                        }
        return patient_dict

    def _create_folds(self):
        patient_ids = list(self.patient_dict.keys())
        labels = [self.patient_dict[pid]['label'] for pid in patient_ids]
        if self.stratify_subtype:
            subtypes = [self.patient_dict[pid]['subtype'] for pid in patient_ids]
            y = [f"{lab}_{sub}" for lab, sub in zip(labels, subtypes)]
        else:
            y = labels
        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state
        )
        folds = []
        for train_idx, test_idx in skf.split(patient_ids, y):
            train_pats = [patient_ids[i] for i in train_idx]
            test_pats = [patient_ids[i] for i in test_idx]
            folds.append((train_pats, test_pats))
        return folds

    def get_fold(self, fold_idx):
        """
        Returns (train_files, test_files) lists for the given fold index.
        """
        train_pats, test_pats = self.folds[fold_idx]
        train_files = [f for pid in train_pats for f in self.patient_dict[pid]['images']]
        test_files  = [f for pid in test_pats  for f in self.patient_dict[pid]['images']]
        return train_files, test_files

    def get_splits(self):
        """
        Returns a list of (train_files, test_files) for all folds.
        """
        return [self.get_fold(i) for i in range(self.n_splits)]
    
    def print_summary(self):
        print("=== Fold-wise Dataset Summary ===")
        for i, (train_files, test_files) in enumerate(self.get_splits()):
            train_labels = [0 if '/benign/' in f.lower() else 1 for f in train_files]
            test_labels  = [0 if '/benign/' in f.lower() else 1 for f in test_files]
            print(f"Fold {i}:",
                f"Train images: {len(train_files)} (B/M = {train_labels.count(0)}/{train_labels.count(1)});",
                f"Test images: {len(test_files)} (B/M = {test_labels.count(0)}/{test_labels.count(1)})")
            