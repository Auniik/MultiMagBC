#!/usr/bin/env python3
"""
Test script to validate the enhanced multi-image sampling strategy
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from preprocess.multimagset import MultiMagPatientDataset, LegacyMultiMagPatientDataset
from preprocess.kfold_splitter import PatientWiseKFoldSplitter
from config import SLIDES_PATH
import torchvision.transforms as T


def test_sampling_comparison():
    """Compare old vs new sampling strategies"""
    print("=== Enhanced Multi-Image Sampling Test ===\n")
    
    # Initialize splitter
    splitter = PatientWiseKFoldSplitter(
        dataset_dir=SLIDES_PATH,
        n_splits=5,
        stratify_subtype=False
    )
    patient_dict = splitter.patient_dict
    
    # Get first fold for testing
    train_pats, test_pats = splitter.folds[0]
    print(f"Testing with fold 0: {len(train_pats)} train, {len(test_pats)} test patients\n")
    
    # Simple transform for testing
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    
    # Test legacy dataset (old approach)
    print("=== Legacy Dataset (1 sample per patient) ===")
    legacy_ds = LegacyMultiMagPatientDataset(patient_dict, train_pats[:10], transform=transform)
    print(f"Legacy dataset length: {len(legacy_ds)}")
    
    # Test a few samples
    for i in range(min(3, len(legacy_ds))):
        images_dict, label = legacy_ds[i]
        print(f"Sample {i}: Label={label}, Images shape: {[f'{k}:{v.shape}' for k, v in images_dict.items()]}")
    
    print("\n=== Enhanced Dataset (adaptive multi-image sampling) ===")
    # Test enhanced dataset
    enhanced_ds = MultiMagPatientDataset(
        patient_dict, train_pats[:10], transform=transform,
        samples_per_patient=5, adaptive_sampling=True
    )
    
    print(f"Enhanced dataset length: {len(enhanced_ds)}")
    
    # Get sampling statistics
    stats = enhanced_ds.get_sampling_stats()
    print(f"Total samples per epoch: {stats['total_samples_per_epoch']}")
    print(f"Average utilization rate: {stats['avg_utilization']:.1%}")
    
    print("\nPer-patient sampling details:")
    for patient_detail in stats['patient_details'][:5]:  # Show first 5 patients
        print(f"  {patient_detail['patient_id']}: "
              f"{patient_detail['avg_images_per_mag']:.1f} avg imgs/mag, "
              f"{patient_detail['samples_per_epoch']} samples, "
              f"{patient_detail['utilization_rate']:.1%} utilization")
    
    # Test a few enhanced samples
    print(f"\nTesting enhanced samples:")
    for i in range(min(5, len(enhanced_ds))):
        images_dict, label = enhanced_ds[i]
        print(f"Sample {i}: Label={label}, Images shape: {[f'{k}:{v.shape}' for k, v in images_dict.items()]}")
    
    # Test epoch diversity
    print(f"\n=== Testing Epoch-based Sampling Diversity ===")
    enhanced_ds.set_epoch(0)
    sample_0_epoch_0, _ = enhanced_ds[0]
    
    enhanced_ds.set_epoch(1) 
    sample_0_epoch_1, _ = enhanced_ds[0]
    
    # Check if samples are different (they should be with high probability)
    different = any(not torch.equal(sample_0_epoch_0[k], sample_0_epoch_1[k]) 
                   for k in sample_0_epoch_0.keys())
    print(f"Sample 0 different between epochs 0 and 1: {different}")
    
    print(f"\n=== Improvement Summary ===")
    improvement_ratio = len(enhanced_ds) / len(legacy_ds)
    print(f"Data utilization improvement: {improvement_ratio:.1f}x")
    print(f"Samples per epoch: {len(legacy_ds)} → {len(enhanced_ds)}")
    print(f"Average utilization rate: {stats['avg_utilization']:.1%}")


if __name__ == "__main__":
    try:
        import torch
        test_sampling_comparison()
        print("\n✅ Enhanced sampling test completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()