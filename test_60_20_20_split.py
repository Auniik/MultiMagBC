#!/usr/bin/env python3
"""
Test script to verify 60-20-20 split implementation
"""

from preprocess.kfold_splitter import PatientWiseKFoldSplitter
from config import SLIDES_PATH

def test_60_20_20_split():
    print("🧪 Testing 60-20-20 split implementation...")
    
    # Create splitter with new validation_split
    splitter = PatientWiseKFoldSplitter(
        dataset_dir=SLIDES_PATH,
        n_splits=5,
        stratify_subtype=False,
        validation_split=0.25  # This should give us 60-20-20
    )
    
    total_patients = len(splitter.patient_dict)
    print(f"📊 Total patients: {total_patients}")
    
    print("\n📈 Split Analysis per Fold:")
    print("Fold | Train | Val | Test | Train% | Val% | Test%")
    print("-" * 50)
    
    for fold_idx in range(5):
        train_pats, val_pats, test_pats = splitter.get_fold(fold_idx)
        
        train_count = len(train_pats)
        val_count = len(val_pats) 
        test_count = len(test_pats)
        
        train_pct = (train_count / total_patients) * 100
        val_pct = (val_count / total_patients) * 100
        test_pct = (test_count / total_patients) * 100
        
        print(f"  {fold_idx}  |  {train_count:2d}  | {val_count:2d}  | {test_count:2d}  | {train_pct:5.1f}% | {val_pct:4.1f}% | {test_pct:4.1f}%")
    
    # Calculate averages
    all_splits = [splitter.get_fold(i) for i in range(5)]
    avg_train = sum(len(split[0]) for split in all_splits) / 5
    avg_val = sum(len(split[1]) for split in all_splits) / 5  
    avg_test = sum(len(split[2]) for split in all_splits) / 5
    
    avg_train_pct = (avg_train / total_patients) * 100
    avg_val_pct = (avg_val / total_patients) * 100
    avg_test_pct = (avg_test / total_patients) * 100
    
    print("-" * 50)
    print(f"Avg  | {avg_train:4.1f} | {avg_val:3.1f} | {avg_test:4.1f} | {avg_train_pct:5.1f}% | {avg_val_pct:4.1f}% | {avg_test_pct:4.1f}%")
    
    # Check if we achieved 60-20-20
    target_achieved = (
        abs(avg_train_pct - 60.0) < 2.0 and
        abs(avg_val_pct - 20.0) < 2.0 and 
        abs(avg_test_pct - 20.0) < 2.0
    )
    
    if target_achieved:
        print(f"\n✅ SUCCESS: Achieved ~60-20-20 split!")
        print(f"   Target: 60%-20%-20%")
        print(f"   Actual: {avg_train_pct:.1f}%-{avg_val_pct:.1f}%-{avg_test_pct:.1f}%")
    else:
        print(f"\n❌ MISS: Did not achieve 60-20-20 split")
        print(f"   Target: 60%-20%-20%") 
        print(f"   Actual: {avg_train_pct:.1f}%-{avg_val_pct:.1f}%-{avg_test_pct:.1f}%")
        
        # Calculate correct validation_split needed
        # If K-fold gives us 80% train, and we want 60% final train:
        # final_train = 0.8 * (1 - val_split) = 0.6
        # val_split = 1 - (0.6 / 0.8) = 1 - 0.75 = 0.25 ✓
        # But let's calculate what we actually need:
        actual_train_from_kfold = 1.0 - (1.0 / 5)  # 80% from 5-fold
        needed_val_split = 1.0 - (0.6 / actual_train_from_kfold)
        print(f"   Suggested validation_split: {needed_val_split:.3f}")
    
    return target_achieved

if __name__ == "__main__":
    success = test_60_20_20_split()
    if success:
        print(f"\n🎉 Ready to train with improved 60-20-20 splits!")
    else:
        print(f"\n🔧 Need to adjust validation_split parameter.")