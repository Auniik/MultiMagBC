import os
import random
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from PIL import Image
from collections import defaultdict
import numpy as np

import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from PIL import Image
import random
from collections import defaultdict
import numpy as np

class MultiMagPatientDataset(Dataset):
    def __init__(self, patient_dict, patient_ids, mags=['40','100','200','400'],
                 transform=None, mode='train',
                 samples_per_patient=None, epoch_multiplier=None,
                 class_balanced_sampling=True, subtype_balancing=True,
                 sampling_mode='strict'):  
        """
        Multi-magnification patient dataset (Q1-ready)
        
        Args:
            patient_dict: dict of patient data
            patient_ids: list of patient IDs
            mags: magnifications to include
            mode: 'train', 'val', 'test'
            samples_per_patient: base samples per patient for training
            epoch_multiplier: epochs cycling factor for adaptive sampling
            class_balanced_sampling: balance benign/malignant classes
            subtype_balancing: balance subtypes within classes
            sampling_mode: 'strict' (min-per-mag) or 'relaxed' (max-per-mag with masking)
        """
        self.patient_dict = patient_dict
        self.patient_ids = patient_ids
        self.mags = mags
        self.transform = transform
        self.mode = mode
        self.class_balanced_sampling = class_balanced_sampling
        self.subtype_balancing = subtype_balancing
        self.sampling_mode = sampling_mode
        self.epoch_seed = 0

        # Compute patient image stats
        self.patient_image_counts = self._compute_patient_stats()

        # Dynamic sample allocation
        if self.mode == 'train':
            self.samples_per_patient = samples_per_patient or 5
            self.epoch_multiplier = epoch_multiplier or 3
            self.adaptive_sampling = True
        else:
            self.samples_per_patient = None
            self.epoch_multiplier = 1
            self.adaptive_sampling = False

        # Final sample allocation
        self.effective_samples = self._compute_effective_samples()

        # Setup class/subtype balancing
        if self.class_balanced_sampling and self.mode == 'train':
            self._setup_balanced_sampling()

    def _compute_patient_stats(self):
        stats = {}
        for pid in self.patient_ids:
            entry = self.patient_dict[pid]
            mag_counts = {mag: len(entry['images'].get(mag, [])) for mag in self.mags}
            min_images = min(mag_counts.values())
            max_images = max(mag_counts.values())
            avg_images = sum(mag_counts.values()) / len(self.mags)
            stats[pid] = {
                'min_per_mag': min_images,
                'max_per_mag': max_images,
                'avg_per_mag': avg_images,
                'mag_counts': mag_counts
            }
        return stats

    def _compute_effective_samples(self):
        effective = {}
        for pid in self.patient_ids:
            stats = self.patient_image_counts[pid]
            if self.mode == 'test':
                samples = stats['max_per_mag']  # full test coverage
            elif self.mode == 'val':
                samples = stats['min_per_mag']  # deterministic validation
            else:
                base = self.samples_per_patient or 1
                if self.sampling_mode == 'strict':
                    samples = min(stats['min_per_mag'], base * 3)
                else:  # relaxed
                    samples = min(stats['max_per_mag'], base * 3)
            effective[pid] = max(1, samples)
        return effective

    def set_epoch(self, epoch):
        self.epoch_seed = epoch

    def __len__(self):
        return sum(self.effective_samples.values()) * self.epoch_multiplier

    def __getitem__(self, idx):
        cumulative = 0
        target_patient, sample_idx = None, 0
        epoch_offset = idx // sum(self.effective_samples.values())
        adjusted_idx = idx % sum(self.effective_samples.values())
        for pid in self.patient_ids:
            patient_samples = self.effective_samples[pid]
            if adjusted_idx < cumulative + patient_samples:
                target_patient = pid
                sample_idx = adjusted_idx - cumulative
                break
            cumulative += patient_samples
        return self._get_patient_sample(target_patient, sample_idx, epoch_offset)

    def _get_patient_sample(self, pid, sample_idx, epoch_offset):
        entry = self.patient_dict[pid]
        images_dict, mask = {}, []
        random_state = random.Random(self.epoch_seed * 1000 + hash(pid) + sample_idx + epoch_offset)

        # Synchronized augmentation: same random order & transform
        sync_seed = random_state.randint(0, int(1e6))
        for mag in self.mags:
            files = entry['images'].get(mag, [])
            if files:
                img_path = files[sample_idx % len(files)] if self.mode != 'train' else random_state.choice(files)
                img = Image.open(img_path).convert('RGB')
                if self.transform:
                    random.seed(sync_seed)
                    img = self.transform(img)
                mask.append(1)
            else:
                img = torch.zeros((3, 224, 224))
                mask.append(0)
            images_dict[f'mag_{mag}'] = img
        return images_dict, torch.tensor(mask, dtype=torch.float32), entry['label']

    def _setup_balanced_sampling(self, max_oversample_factor=3.0):
        """
        Setup class + subtype balanced sampling with oversampling cap.
        Args:
            max_oversample_factor: Max allowed weight multiplier relative to median (prevents extreme oversampling)
        """
        # Build class and subtype maps
        self.class_to_patients = defaultdict(list)
        self.subtype_to_patients = defaultdict(list)
        for pid in self.patient_ids:
            label = self.patient_dict[pid]['label']
            subtype = self.patient_dict[pid]['subtype']
            self.class_to_patients[label].append(pid)
            self.subtype_to_patients[subtype].append(pid)

        total = len(self.patient_ids)
        # --- Compute class weights (benign vs malignant) ---
        class_counts = {cls: len(pats) for cls, pats in self.class_to_patients.items()}
        self.class_weights = {
            cls: total / (len(class_counts) * count) for cls, count in class_counts.items()
        }

        if self.subtype_balancing:
            # --- Compute subtype weights ---
            subtype_counts = {sub: len(pats) for sub, pats in self.subtype_to_patients.items()}
            self.subtype_weights = {
                sub: total / (len(subtype_counts) * count) for sub, count in subtype_counts.items()
            }
            # Combine class × subtype
            raw_patient_weights = {
                pid: self.class_weights[self.patient_dict[pid]['label']] *
                    self.subtype_weights[self.patient_dict[pid]['subtype']]
                for pid in self.patient_ids
            }
        else:
            raw_patient_weights = {
                pid: self.class_weights[self.patient_dict[pid]['label']]
                for pid in self.patient_ids
            }

        # --- Cap extreme oversampling ---
        median_weight = np.median(list(raw_patient_weights.values()))
        self.patient_weights = {
            pid: min(weight, median_weight * max_oversample_factor)
            for pid, weight in raw_patient_weights.items()
        }

        # --- Log for transparency ---
        print("📊 [Sampling] Class weights:", self.class_weights)
        if self.subtype_balancing:
            print("📊 [Sampling] Subtype weights:", self.subtype_weights)
        print(f"📊 [Sampling] Weight cap applied at {max_oversample_factor}× median ({median_weight:.3f}).")

    def get_class_balanced_sampler(self):
        if not self.class_balanced_sampling or self.mode != 'train':
            return None
        sample_weights = [self.patient_weights[pid] for pid in self.patient_ids for _ in range(self.effective_samples[pid] * self.epoch_multiplier)]
        return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    def get_sampling_stats(self):
        total_samples = sum(self.effective_samples.values()) * self.epoch_multiplier
        class_samples = defaultdict(int)
        subtype_samples = defaultdict(int)
        for pid in self.patient_ids:
            label = self.patient_dict[pid]['label']
            subtype = self.patient_dict[pid]['subtype']
            count = self.effective_samples[pid] * self.epoch_multiplier
            class_samples[label] += count
            subtype_samples[subtype] += count
        oversampling_factor = round(total_samples / (sum(stats['min_per_mag'] for stats in self.patient_image_counts.values()) or 1), 2)
        return {
            'total_samples_per_epoch': total_samples,
            'class_distribution': dict(class_samples),
            'subtype_distribution': dict(subtype_samples),
            'oversampling_factor': oversampling_factor
        }