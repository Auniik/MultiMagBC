#!/usr/bin/env python3
"""
Validate the enhanced sampling logic without requiring PyTorch
"""

import os
import random
from collections import defaultdict

# Mock patient data for testing
def create_mock_patient_dict():
    """Create mock patient dictionary based on real patient_wise_image_count.txt structure"""
    mock_patients = {
        "SOB_M_DC_14-12312": {  # High volume malignant
            "label": 1,
            "images": [
                f"/path/to/images/{mag}X/image_{i}.png" 
                for mag in ["40", "100", "200", "400"] 
                for i in range(35)  # ~35 images per mag
            ]
        },
        "SOB_M_DC_14-8168": {  # Low volume malignant 
            "label": 1,
            "images": [
                f"/path/to/images/{mag}X/image_{i}.png" 
                for mag in ["40", "100", "200", "400"] 
                for i in range(10)  # ~10 images per mag
            ]
        },
        "SOB_B_F_14-14134": {  # Medium volume benign
            "label": 0,
            "images": [
                f"/path/to/images/{mag}X/image_{i}.png" 
                for mag in ["40", "100", "200", "400"] 
                for i in range(25)  # ~25 images per mag
            ]
        },
        "SOB_B_A_14-29960CD": {  # Low volume benign
            "label": 0,
            "images": [
                f"/path/to/images/{mag}X/image_{i}.png" 
                for mag in ["40", "100", "200", "400"] 
                for i in range(15)  # ~15 images per mag
            ]
        }
    }
    return mock_patients

def compute_patient_stats(patient_dict, patient_ids, mags=['40', '100', '200', '400']):
    """Mock version of _compute_patient_stats from MultiMagPatientDataset"""
    stats = {}
    for pid in patient_ids:
        entry = patient_dict[pid]
        mag_counts = defaultdict(int)
        
        for fpath in entry['images']:
            # Extract magnification from file path
            for mag in mags:
                if f"/{mag}X/" in fpath:
                    mag_counts[mag] += 1
                    break
        
        # Calculate statistics
        min_images = min([mag_counts.get(mag, 0) for mag in mags])
        avg_images = sum([mag_counts.get(mag, 0) for mag in mags]) / len(mags)
        
        stats[pid] = {
            'min_per_mag': min_images,
            'avg_per_mag': avg_images,
            'mag_counts': dict(mag_counts)
        }
        
    return stats

def compute_effective_samples(patient_stats, patient_ids, samples_per_patient=5, adaptive_sampling=True):
    """Mock version of _compute_effective_samples from MultiMagPatientDataset"""
    if not adaptive_sampling:
        return {pid: samples_per_patient for pid in patient_ids}
        
    effective = {}
    for pid in patient_ids:
        stats = patient_stats[pid]
        avg_images = stats['avg_per_mag']
        
        if avg_images <= 15:
            # Low volume: use more samples to maximize data
            samples = min(int(avg_images * 0.8), samples_per_patient * 3)
        elif avg_images <= 30:
            # Medium volume: moderate sampling
            samples = min(int(avg_images * 0.6), samples_per_patient * 2)
        else:
            # High volume: conservative sampling to prevent overfitting
            samples = min(int(avg_images * 0.4), samples_per_patient * 2)
            
        effective[pid] = max(1, samples)  # At least 1 sample per patient
        
    return effective

def test_adaptive_sampling():
    """Test the adaptive sampling logic"""
    print("=== Enhanced Multi-Image Sampling Logic Validation ===\n")
    
    # Create mock data
    patient_dict = create_mock_patient_dict()
    patient_ids = list(patient_dict.keys())
    
    print("Mock Patient Data:")
    for pid, data in patient_dict.items():
        label_name = "Malignant" if data['label'] == 1 else "Benign"
        total_images = len(data['images'])
        images_per_mag = total_images // 4
        print(f"  {pid}: {label_name}, {total_images} total images (~{images_per_mag} per mag)")
    
    print(f"\n=== Legacy Sampling (1 sample per patient) ===")
    legacy_samples = len(patient_ids)  # 1 sample per patient
    print(f"Total samples per epoch: {legacy_samples}")
    
    print(f"\n=== Enhanced Adaptive Sampling ===")
    # Compute patient statistics
    patient_stats = compute_patient_stats(patient_dict, patient_ids)
    
    print("Patient Statistics:")
    for pid, stats in patient_stats.items():
        print(f"  {pid}: {stats['avg_per_mag']:.1f} avg images per magnification")
    
    # Test different sampling configurations
    configs = [
        {"samples_per_patient": 3, "adaptive_sampling": True},
        {"samples_per_patient": 5, "adaptive_sampling": True},
        {"samples_per_patient": 5, "adaptive_sampling": False},
    ]
    
    for config in configs:
        effective_samples = compute_effective_samples(
            patient_stats, patient_ids, 
            config['samples_per_patient'], 
            config['adaptive_sampling']
        )
        
        total_samples = sum(effective_samples.values())
        improvement_ratio = total_samples / legacy_samples
        
        print(f"\nConfig: {config}")
        print(f"Effective samples per patient:")
        for pid, samples in effective_samples.items():
            utilization = samples / patient_stats[pid]['avg_per_mag']
            print(f"  {pid}: {samples} samples ({utilization:.1%} utilization)")
        
        print(f"Total samples per epoch: {total_samples}")
        print(f"Improvement over legacy: {improvement_ratio:.1f}x")
    
    # Test realistic scenario with more patients
    print(f"\n=== Realistic Scenario Simulation ===")
    # Simulate 65 training patients with realistic distribution
    realistic_stats = []
    for i in range(65):
        # Simulate realistic image counts per magnification
        if i < 15:  # Low volume patients
            avg_images = random.uniform(8, 15)
        elif i < 45:  # Medium volume patients  
            avg_images = random.uniform(16, 30)
        else:  # High volume patients
            avg_images = random.uniform(31, 50)
        realistic_stats.append(avg_images)
    
    # Calculate samples for realistic scenario
    total_enhanced = 0
    total_utilization = 0
    
    for avg_images in realistic_stats:
        if avg_images <= 15:
            samples = min(int(avg_images * 0.8), 15)  # samples_per_patient * 3
        elif avg_images <= 30:
            samples = min(int(avg_images * 0.6), 10)  # samples_per_patient * 2
        else:
            samples = min(int(avg_images * 0.4), 10)  # samples_per_patient * 2
        
        samples = max(1, samples)
        total_enhanced += samples
        total_utilization += samples / avg_images
    
    avg_utilization = total_utilization / len(realistic_stats)
    improvement_realistic = total_enhanced / 65  # 65 legacy samples
    
    print(f"Realistic 65-patient scenario:")
    print(f"Legacy samples per epoch: 65")
    print(f"Enhanced samples per epoch: {total_enhanced}")
    print(f"Improvement ratio: {improvement_realistic:.1f}x") 
    print(f"Average utilization rate: {avg_utilization:.1%}")
    
    print(f"\n✅ Enhanced sampling logic validation completed!")
    return True

if __name__ == "__main__":
    try:
        test_adaptive_sampling()
    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()