import os, glob, random, json
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit



class PatientWiseKFoldSplitter:
    """
    Splits a histopathology dataset into patient-wise K-fold splits with optional stratification.
    Expects structure:
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
        stratify_subtype=False,
        validation_split=0.2
    ):
        self.dataset_dir = dataset_dir
        self.n_splits = n_splits
        self.random_state = random_state
        self.stratify_subtype = stratify_subtype
        self.validation_split = validation_split
        self.patient_dict, self.magnifications = self._scan_dataset()
        self.folds = self._create_folds()

    def _scan_dataset(self):
        """Scan dataset and collect patient-level metadata with per-magnification images."""
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

                        patient_dict[pid_key] = {
                            'label': label,
                            'subtype': subtype,
                            'images': mag_images
                        }

        return patient_dict, sorted(list(magnifications_set))

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

    def get_fold(self, fold_idx, return_type='patients'):
        """Returns train/val/test splits for a fold."""
        train_pats, test_pats = self.folds[fold_idx]

        # Stratified train/val split
        labels = [self.patient_dict[pid]['label'] for pid in train_pats]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=self.validation_split, random_state=self.random_state)
        train_idx, val_idx = next(sss.split(train_pats, labels))
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
        """Returns a list of (train, val, test) for all folds."""
        return [self.get_fold(i, return_type=return_type) for i in range(self.n_splits)]

    def print_summary(self):
        print("=== Fold-wise Dataset Summary ===")
        for i, (train_pats, test_pats) in enumerate(self.folds):
            train_labels = [self.patient_dict[pid]['label'] for pid in train_pats]
            test_labels = [self.patient_dict[pid]['label'] for pid in test_pats]
            train_images = sum(len(imgs) for pid in train_pats for imgs in self.patient_dict[pid]['images'].values())
            test_images = sum(len(imgs) for pid in test_pats for imgs in self.patient_dict[pid]['images'].values())
            print(f"Fold {i}: Train patients: {len(train_pats)} (images={train_images}, B/M = {train_labels.count(0)}/{train_labels.count(1)}); "
                  f"Test patients: {len(test_pats)} (images={test_images}, B/M = {test_labels.count(0)}/{test_labels.count(1)})")

    def save_metadata(self, path="./output/dataset/fold_splits.json"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {i: {"train": t, "test": te} for i, (t, te) in enumerate(self.folds)}
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved fold splits to {path}")

    def visualize(self):
        save_dir = './output/dataset'
        os.makedirs(save_dir, exist_ok=True)
        fold_stats = []

        for i, (train_pats, test_pats) in enumerate(self.folds):
            pats = train_pats + test_pats
            labels = [self.patient_dict[pid]['label'] for pid in pats]
            label_counts = Counter(labels)

            mag_counts = {m: sum(len(self.patient_dict[pid]['images'].get(m, [])) for pid in pats) for m in self.magnifications}
            fold_stats.append({
                "fold": i,
                "benign": label_counts[0],
                "malignant": label_counts[1],
                "mag_counts": mag_counts
            })

        # Class distribution
        plt.figure(figsize=(8, 5))
        benign = [fs["benign"] for fs in fold_stats]
        malignant = [fs["malignant"] for fs in fold_stats]
        plt.bar(range(len(fold_stats)), benign, label="Benign", color="skyblue")
        plt.bar(range(len(fold_stats)), malignant, bottom=benign, label="Malignant", color="salmon")
        plt.xticks(range(len(fold_stats)), [f"Fold {fs['fold']}" for fs in fold_stats])
        plt.ylabel("Patients")
        plt.title("Class Distribution per Fold")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "class_distribution_per_fold.png"))
        plt.close()

        # Per-magnification counts
        plt.figure(figsize=(10, 6))
        for m in self.magnifications:
            plt.plot(range(len(fold_stats)), [fs["mag_counts"][m] for fs in fold_stats], marker='o', label=f"{m}X")
        plt.xticks(range(len(fold_stats)), [f"Fold {fs['fold']}" for fs in fold_stats])
        plt.ylabel("Image Count")
        plt.title("Magnification-wise Image Counts per Fold")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "magnification_counts_per_fold.png"))
        plt.close()

        print(f"Saved visualizations to {save_dir}")