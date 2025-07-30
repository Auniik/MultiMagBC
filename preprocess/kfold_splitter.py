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
        validation_split=0.25  # 0.25 from 80% train = 20% val, leaving 60% final train
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
        """
        Create subtype- and label-balanced K folds for patient-wise CV.
        Ensures:
        - Each fold has all subtypes (rarest first assignment).
        - Benign/malignant distribution is balanced.
        - Fold sizes are close to equal.
        """
        patient_ids = list(self.patient_dict.keys())

        # Group patients by subtype
        subtype_groups = {}
        for pid in patient_ids:
            subtype = self.patient_dict[pid]['subtype']
            subtype_groups.setdefault(subtype, []).append(pid)

        # Sort subtypes by rarity (rarest first)
        subtype_groups = dict(sorted(subtype_groups.items(), key=lambda x: len(x[1])))

        folds = [[] for _ in range(self.n_splits)]

        # Greedy assignment: balance subtype, then label, then size
        for subtype, patients in subtype_groups.items():
            random.shuffle(patients)
            for pid in patients:
                target_label = self.patient_dict[pid]['label']
                # Compute per-fold scores (subtype count, label count, size)
                fold_scores = []
                for fold in folds:
                    subtype_count = sum(1 for p in fold if self.patient_dict[p]['subtype'] == subtype)
                    label_count = sum(1 for p in fold if self.patient_dict[p]['label'] == target_label)
                    fold_scores.append((subtype_count, label_count, len(fold)))
                # Pick fold with minimal counts (subtype > label > size)
                min_fold = min(range(len(fold_scores)), key=lambda i: (fold_scores[i][0], fold_scores[i][1], fold_scores[i][2]))
                folds[min_fold].append(pid)

        # Standard K-fold: test = 1 fold, train+val = remaining folds
        return [([p for j, f in enumerate(folds) if j != i for p in f], folds[i]) for i in range(self.n_splits)]
    
    def _subtype_stratified_split(self, patient_list, val_fraction=0.25):
        """Greedy subtype-aware split for val inside train."""
        # Group patients by subtype
        subtype_groups = {}
        for pid in patient_list:
            subtype = self.patient_dict[pid]['subtype']
            subtype_groups.setdefault(subtype, []).append(pid)

        # Sort by rarity
        subtype_groups = dict(sorted(subtype_groups.items(), key=lambda x: len(x[1])))

        val_size = int(len(patient_list) * val_fraction)
        val_set = set()
        train_set = set(patient_list)

        for subtype, patients in subtype_groups.items():
            random.shuffle(patients)
            n_val = max(1, int(len(patients) * val_fraction))
            selected = patients[:n_val]
            val_set.update(selected)
            train_set.difference_update(selected)

        return list(train_set), list(val_set)

    def get_fold(self, fold_idx, return_type='patients'):
        """Returns train/val/test splits for a fold."""
        train_pats, test_pats = self.folds[fold_idx]
        train_pats, val_pats = self._subtype_stratified_split(train_pats, self.validation_split)

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

    # def print_summary(self):
    #     print("=== Fold-wise Dataset Summary ===")
    #     for i, (train_pats, test_pats) in enumerate(self.folds):
    #         train_labels = [self.patient_dict[pid]['label'] for pid in train_pats]
    #         test_labels = [self.patient_dict[pid]['label'] for pid in test_pats]
    #         train_images = sum(len(imgs) for pid in train_pats for imgs in self.patient_dict[pid]['images'].values())
    #         test_images = sum(len(imgs) for pid in test_pats for imgs in self.patient_dict[pid]['images'].values())
    #         print(f"Fold {i}: Train patients: {len(train_pats)} (images={train_images}, B/M = {train_labels.count(0)}/{train_labels.count(1)}); "
    #               f"Test patients: {len(test_pats)} (images={test_images}, B/M = {test_labels.count(0)}/{test_labels.count(1)})")

    def print_summary(self):
        print("=== Fold-wise Dataset Fairness Summary ===")
        all_subtypes = sorted(set([self.patient_dict[pid]['subtype'] for pid in self.patient_dict]))

        for i, (train_pats, test_pats) in enumerate(self.folds):
            # Add validation
            train_only, val_pats = self._subtype_stratified_split(train_pats, self.validation_split)

            def summarize(patients):
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

            train_stats = summarize(train_only)
            val_stats = summarize(val_pats)
            test_stats = summarize(test_pats)

            print(f"\n--- Fold {i} ---")
            print(f"Train: {train_stats['patients']} patients ({train_stats['benign']}/{train_stats['malignant']} B/M, {train_stats['benign_pct']}% benign)")
            print(f"Val:   {val_stats['patients']} patients ({val_stats['benign']}/{val_stats['malignant']} B/M, {val_stats['benign_pct']}% benign)")
            print(f"Test:  {test_stats['patients']} patients ({test_stats['benign']}/{test_stats['malignant']} B/M, {test_stats['benign_pct']}% benign)")
            print(f"Subtypes present (Train): {', '.join(train_stats['subtypes_present'])}")
            print(f"Subtypes present (Test):  {', '.join(test_stats['subtypes_present'])}")
            print(f"Magnification distribution (Train): {train_stats['magnification_counts']}")

    def save_metadata(self, path="./output/dataset/fold_splits.json"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {i: {"train": t, "test": te} for i, (t, te) in enumerate(self.folds)}
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved fold splits to {path}")

    def visualize(self):
        save_dir = './output/dataset'
        os.makedirs(save_dir, exist_ok=True)

        all_subtypes = sorted(set([self.patient_dict[pid]['subtype'] for pid in self.patient_dict]))
        fold_stats = []

        for i, (train_pats, test_pats) in enumerate(self.folds):
            # Add validation split
            train_only, val_pats = self._subtype_stratified_split(train_pats, self.validation_split)
            sets = {'Train': train_only, 'Val': val_pats, 'Test': test_pats}

            set_stats = {}
            for name, pats in sets.items():
                labels = [self.patient_dict[pid]['label'] for pid in pats]
                subtypes = [self.patient_dict[pid]['subtype'] for pid in pats]
                mag_counts = {m: sum(len(self.patient_dict[pid]['images'].get(m, [])) for pid in pats) for m in self.magnifications}
                set_stats[name] = {
                    'benign': labels.count(0),
                    'malignant': labels.count(1),
                    'benign_pct': round(labels.count(0)/len(pats)*100,1) if pats else 0,
                    'subtype_counts': {st: subtypes.count(st) for st in all_subtypes},
                    'mag_counts': mag_counts
                }
            fold_stats.append(set_stats)

        # --- 1. Benign vs Malignant per fold (percentages) ---
        plt.figure(figsize=(8,5))
        test_benign_pct = [fs['Test']['benign_pct'] for fs in fold_stats]
        plt.bar(range(len(fold_stats)), test_benign_pct, color="skyblue", label="Benign % (Test)")
        plt.xticks(range(len(fold_stats)), [f"Fold {i}" for i in range(len(fold_stats))])
        plt.ylabel("Benign %")
        plt.title("Benign vs Malignant Ratio in Test Sets")
        plt.ylim(0, 100)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "benign_malignant_ratio_test.png"))
        plt.close()

        # --- 2. Subtype distribution per fold (stacked) ---
        subtype_colors = plt.cm.get_cmap('tab20', len(all_subtypes))
        fig, ax = plt.subplots(figsize=(10,6))
        bottom = [0]*len(fold_stats)
        for idx, subtype in enumerate(all_subtypes):
            values = [fs['Test']['subtype_counts'][subtype] for fs in fold_stats]
            ax.bar(range(len(fold_stats)), values, bottom=bottom, label=subtype, color=subtype_colors(idx))
            bottom = [b+v for b,v in zip(bottom, values)]
        ax.set_xticks(range(len(fold_stats)))
        ax.set_xticklabels([f"Fold {i}" for i in range(len(fold_stats))])
        ax.set_ylabel("Patients")
        ax.set_title("Subtype Distribution in Test Sets")
        ax.legend(bbox_to_anchor=(1.05,1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "subtype_distribution_test.png"))
        plt.close()

        # --- 3. Magnification counts (Train vs Test) ---
        plt.figure(figsize=(10,6))
        for m in self.magnifications:
            plt.plot(range(len(fold_stats)), [fs['Train']['mag_counts'][m] for fs in fold_stats], marker='o', label=f"{m}X (Train)")
            plt.plot(range(len(fold_stats)), [fs['Test']['mag_counts'][m] for fs in fold_stats], marker='x', linestyle='--', label=f"{m}X (Test)")
        plt.xticks(range(len(fold_stats)), [f"Fold {i}" for i in range(len(fold_stats))])
        plt.ylabel("Image Count")
        plt.title("Magnification-wise Image Counts per Fold (Train vs Test)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "magnification_counts_train_test.png"))
        plt.close()

        print(f"Saved detailed visualizations to {save_dir}")