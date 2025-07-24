import os
import random
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from PIL import Image
from collections import defaultdict
import numpy as np

class MultiMagPatientDataset(Dataset):
    def __init__(self, patient_dict, patient_ids, mags=['40', '100', '200', '400'], 
                 transform=None, mode='train', samples_per_patient=None, epoch_multiplier=None,
                 class_balanced_sampling=True, full_utilization_mode='all'):
        """
        Multi-magnification patient dataset with automatic mode handling.

        Args:
            patient_dict: dict of patient data
            patient_ids: list of patient IDs
            mags: magnifications to include
            mode: 'train', 'val', 'test'
            samples_per_patient: base samples per patient (for training)
            epoch_multiplier: number of epochs with sampling diversity
            class_balanced_sampling: use class-balanced sampling for training
            full_utilization_mode: 'min' (strict complete sets), 'max', 'all' (avg per mag)
        """
        self.patient_dict = patient_dict
        self.patient_ids = patient_ids
        self.mags = mags
        self.transform = transform
        self.mode = mode
        self.class_balanced_sampling = class_balanced_sampling
        self.full_utilization_mode = full_utilization_mode
        self.epoch_seed = 0

        # Compute patient image stats
        self.patient_image_counts = self._compute_patient_stats()

        # Dynamic sampling configuration
        if self.mode == 'train':
            self.samples_per_patient = samples_per_patient or 5
            self.epoch_multiplier = epoch_multiplier or 3
            self.adaptive_sampling = True
        else:  # val/test: full utilization
            self.samples_per_patient = None
            self.epoch_multiplier = 1
            self.adaptive_sampling = False

        # Compute final samples
        self.effective_samples = self._compute_effective_samples()

        # Setup class balancing
        if self.class_balanced_sampling and self.mode == 'train':
            self._setup_class_balanced_sampling()

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
            if self.mode in ['val', 'test']:
                samples = stats['min_per_mag']  # FULL utilization
            else:
                avg_images = stats['avg_per_mag']
                base = self.samples_per_patient or 1
                if avg_images <= 15:
                    samples = int(avg_images * 0.9)
                elif avg_images <= 30:
                    samples = int(avg_images * 0.85)
                else:
                    samples = int(avg_images * 0.8)
                # Cap to avoid unrealistic oversampling
                samples = min(samples, stats['min_per_mag'] * 2, base * 3)
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
        if target_patient is None:
            target_patient = self.patient_ids[0]
        return self._get_patient_sample(target_patient, sample_idx, epoch_offset)

    def _get_patient_sample(self, pid, sample_idx, epoch_offset):
        entry = self.patient_dict[pid]
        images_dict, mask = {}, []
        random_state = random.Random(self.epoch_seed * 1000 + hash(pid) + sample_idx + epoch_offset)
        for mag in self.mags:
            files = entry['images'].get(mag, [])
            if files:
                if self.mode == 'train':
                    img_path = files[sample_idx % len(files)] if len(files) > sample_idx else random_state.choice(files)
                else:
                    img_path = files[sample_idx % len(files)]  # deterministic for val/test
                img = Image.open(img_path).convert('RGB')
                img = self.transform(img) if self.transform else img
                mask.append(1)
            else:
                img = torch.zeros((3, 224, 224))
                mask.append(0)
            images_dict[f'mag_{mag}'] = img
        return images_dict, torch.tensor(mask, dtype=torch.float32), entry['label']

    def _setup_class_balanced_sampling(self):
        self.class_to_patients = defaultdict(list)
        for pid in self.patient_ids:
            self.class_to_patients[self.patient_dict[pid]['label']].append(pid)
        class_counts = {cls: len(pats) for cls, pats in self.class_to_patients.items()}
        total = len(self.patient_ids)
        self.class_weights = {cls: total / (len(class_counts) * count) for cls, count in class_counts.items()}
        self.patient_weights = {pid: self.class_weights[label] / len(pats)
                                for label, pats in self.class_to_patients.items() for pid in pats}

    def get_class_balanced_sampler(self):
        if not self.class_balanced_sampling or self.mode != 'train':
            return None
        sample_weights = [self.patient_weights[pid] for pid in self.patient_ids for _ in range(self.effective_samples[pid] * self.epoch_multiplier)]
        return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    def get_sampling_stats(self):
        total_samples = sum(self.effective_samples.values()) * self.epoch_multiplier
        class_samples = defaultdict(int)
        for pid in self.patient_ids:
            label = self.patient_dict[pid]['label']
            class_samples[label] += self.effective_samples[pid] * self.epoch_multiplier

        # Oversampling factor: total samples / sum of unique full multi-mag sets
        total_unique_sets = sum(stats['min_per_mag'] for stats in self.patient_image_counts.values())
        oversampling_factor = round(total_samples / (total_unique_sets or 1), 2)

        return {
            'total_samples_per_epoch': total_samples,
            'class_distribution': dict(class_samples),
            'oversampling_factor': oversampling_factor
        }