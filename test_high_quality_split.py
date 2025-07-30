#!/usr/bin/env python3
"""
Test high-quality dataset filtering for 95%+ accuracy target
"""

from config import SLIDES_PATH
from preprocess.high_quality_splitter import HighQualitySplitter

def test_high_quality_filtering():
    print("🎯 Testing High-Quality Dataset for 95%+ Accuracy")
    
    # Create high-quality splitter
    splitter = HighQualitySplitter(
        dataset_dir=SLIDES_PATH,
        n_splits=5,
        validation_split=0.2,  # 80-20 for more training data
        min_images_per_patient=80,  # High-quality threshold
        balanced_subtypes=True
    )
    
    # Print summary
    splitter.print_summary()
    
    # Test first fold
    train_pats, val_pats, test_pats = splitter.get_fold(0)
    
    print(f"\n🔬 Sample Fold 0 Analysis:")
    print(f"Train: {len(train_pats)} patients")
    print(f"Val: {len(val_pats)} patients") 
    print(f"Test: {len(test_pats)} patients")
    
    # Quality verification
    train_images = [splitter.patient_dict[pid]['total_images'] for pid in train_pats]
    val_images = [splitter.patient_dict[pid]['total_images'] for pid in val_pats]
    test_images = [splitter.patient_dict[pid]['total_images'] for pid in test_pats]
    
    print(f"\nImage Quality Check:")
    print(f"Train avg images: {sum(train_images)/len(train_images):.1f}")
    print(f"Val avg images: {sum(val_images)/len(val_images):.1f}")
    print(f"Test avg images: {sum(test_images)/len(test_images):.1f}")
    
    print(f"Min images per patient: {min(train_images + val_images + test_images)}")
    
    print(f"\n💡 Expected Improvements:")
    print(f"• Higher quality data → Better model performance")
    print(f"• Consistent high image counts → Stable training")
    print(f"• Balanced subtypes → Robust generalization")
    print(f"• Target: 95%+ accuracy achievable")
    
    return splitter

if __name__ == "__main__":
    splitter = test_high_quality_filtering()