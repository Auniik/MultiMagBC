import os
import random
from torch.utils.data import Dataset
from PIL import Image
from collections import defaultdict


class MultiMagPatientDataset(Dataset):
    def __init__(self, patient_dict, patient_ids, mags=['40', '100', '200', '400'], 
                 transform=None, samples_per_patient=1, epoch_multiplier=1, 
                 adaptive_sampling=True):
        """
        Enhanced dataset with multi-image sampling per patient
        
        Args:
            patient_dict: Dictionary containing patient data
            patient_ids: List of patient IDs to include
            mags: List of magnifications to use
            transform: Image transformations to apply
            samples_per_patient: Base number of image sets to sample per patient
            epoch_multiplier: Multiplier for creating different combinations across epochs
            adaptive_sampling: Use smart sampling based on available images per patient
        """
        self.patient_dict = patient_dict
        self.patient_ids = patient_ids
        self.mags = mags
        self.transform = transform
        self.samples_per_patient = samples_per_patient
        self.epoch_multiplier = epoch_multiplier
        self.adaptive_sampling = adaptive_sampling
        self.epoch_seed = 0
        
        # Pre-compute image distribution per patient for adaptive sampling
        self.patient_image_counts = self._compute_patient_stats()
        self.effective_samples = self._compute_effective_samples()
        
    def _compute_patient_stats(self):
        """Compute image count statistics per patient per magnification"""
        stats = {}
        for pid in self.patient_ids:
            entry = self.patient_dict[pid]
            mag_counts = defaultdict(int)
            
            for fpath in entry['images']:
                mag = os.path.basename(os.path.dirname(fpath)).replace('X', '')
                if mag in self.mags:
                    mag_counts[mag] += 1
            
            # Calculate min images across mags for this patient
            min_images = min([mag_counts.get(mag, 0) for mag in self.mags])
            avg_images = sum([mag_counts.get(mag, 0) for mag in self.mags]) / len(self.mags)
            
            stats[pid] = {
                'min_per_mag': min_images,
                'avg_per_mag': avg_images,
                'mag_counts': dict(mag_counts)
            }
            
        return stats
    
    def _compute_effective_samples(self):
        """Compute effective number of samples per patient using adaptive strategy"""
        if not self.adaptive_sampling:
            return {pid: self.samples_per_patient for pid in self.patient_ids}
            
        effective = {}
        for pid in self.patient_ids:
            stats = self.patient_image_counts[pid]
            avg_images = stats['avg_per_mag']
            
            if avg_images <= 15:
                # Low volume: use more samples to maximize data
                samples = min(int(avg_images * 0.8), self.samples_per_patient * 3)
            elif avg_images <= 30:
                # Medium volume: moderate sampling
                samples = min(int(avg_images * 0.6), self.samples_per_patient * 2)
            else:
                # High volume: conservative sampling to prevent overfitting
                samples = min(int(avg_images * 0.4), self.samples_per_patient * 2)
                
            effective[pid] = max(1, samples)  # At least 1 sample per patient
            
        return effective
    
    def set_epoch(self, epoch):
        """Set epoch for deterministic sampling across workers"""
        self.epoch_seed = epoch
        
    def __len__(self):
        return sum(self.effective_samples.values()) * self.epoch_multiplier

    def __getitem__(self, idx):
        # Map flat index to (patient_id, sample_within_patient)
        cumulative = 0
        target_patient = None
        sample_idx = 0
        
        # Adjust index for epoch multiplier
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
            sample_idx = 0
            
        return self._get_patient_sample(target_patient, sample_idx, epoch_offset)
    
    def _get_patient_sample(self, pid, sample_idx, epoch_offset):
        """Get a specific sample for a patient"""
        entry = self.patient_dict[pid]
        images_dict = {}
        label = entry['label']
        
        # Create deterministic but varied sampling using epoch and sample index
        random_state = random.Random(self.epoch_seed * 1000 + hash(pid) + sample_idx + epoch_offset)
        
        # Group images by magnification
        mag_to_files = {mag: [] for mag in self.mags}
        for fpath in entry['images']:
            mag = os.path.basename(os.path.dirname(fpath)).replace('X', '')
            if mag in mag_to_files:
                mag_to_files[mag].append(fpath)
        
        # Sample images for each magnification
        for mag in self.mags:
            files = mag_to_files[mag]
            if not files:
                # Fallback: sample from any available images
                files = [f for f in entry['images'] 
                        if os.path.basename(os.path.dirname(f)).replace('X', '') in self.mags]
                if not files:
                    files = entry['images']
            
            # Deterministic sampling with replacement if needed
            if len(files) > sample_idx:
                img_path = files[sample_idx % len(files)]
            else:
                img_path = random_state.choice(files)
                
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            images_dict[f'mag_{mag}'] = img
            
        return images_dict, label
    
    def get_sampling_stats(self):
        """Get statistics about sampling strategy"""
        total_samples = sum(self.effective_samples.values())
        patient_stats = []
        
        for pid in self.patient_ids:
            stats = self.patient_image_counts[pid]
            samples = self.effective_samples[pid]
            patient_stats.append({
                'patient_id': pid,
                'avg_images_per_mag': stats['avg_per_mag'],
                'samples_per_epoch': samples,
                'utilization_rate': samples / stats['avg_per_mag'] if stats['avg_per_mag'] > 0 else 0
            })
            
        return {
            'total_samples_per_epoch': total_samples,
            'total_with_multiplier': total_samples * self.epoch_multiplier,
            'patient_details': patient_stats,
            'avg_utilization': sum(p['utilization_rate'] for p in patient_stats) / len(patient_stats)
        }


# Legacy dataset for backward compatibility
class LegacyMultiMagPatientDataset(MultiMagPatientDataset):
    """Legacy dataset that maintains old behavior (1 sample per patient)"""
    def __init__(self, patient_dict, patient_ids, mags=['40', '100', '200', '400'], transform=None):
        super().__init__(patient_dict, patient_ids, mags, transform, 
                        samples_per_patient=1, adaptive_sampling=False)