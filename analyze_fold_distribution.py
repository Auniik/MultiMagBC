#!/usr/bin/env python3
"""
Analyze fold distribution to identify dataset issues causing performance variance
"""

import os
from collections import Counter, defaultdict
from config import SLIDES_PATH
from preprocess.kfold_splitter import PatientWiseKFoldSplitter

def analyze_fold_distribution():
    print("🔍 Analyzing Fold Distribution for Dataset Issues")
    
    splitter = PatientWiseKFoldSplitter(
        dataset_dir=SLIDES_PATH,
        n_splits=5,
        validation_split=0.25
    )
    
    patient_dict = splitter.patient_dict
    
    print(f"\n📊 Overall Dataset Statistics:")
    print(f"Total patients: {len(patient_dict)}")
    
    # Overall statistics
    overall_stats = {
        'benign': 0, 'malignant': 0,
        'total_images': 0,
        'subtypes': Counter(),
        'mag_availability': defaultdict(int),
        'images_per_patient': [],
        'images_per_mag': defaultdict(int)
    }
    
    for pid, data in patient_dict.items():
        overall_stats['benign' if data['label'] == 0 else 'malignant'] += 1
        overall_stats['subtypes'][data['subtype']] += 1
        
        patient_images = 0
        available_mags = 0
        for mag in ['40', '100', '200', '400']:
            if mag in data['images'] and len(data['images'][mag]) > 0:
                mag_count = len(data['images'][mag])
                overall_stats['images_per_mag'][mag] += mag_count
                overall_stats['total_images'] += mag_count
                patient_images += mag_count
                available_mags += 1
        
        overall_stats['mag_availability'][available_mags] += 1
        overall_stats['images_per_patient'].append(patient_images)
    
    print(f"Benign: {overall_stats['benign']}, Malignant: {overall_stats['malignant']}")
    print(f"Total images: {overall_stats['total_images']}")
    print(f"Images per magnification: {dict(overall_stats['images_per_mag'])}")
    
    # Analyze each fold
    print(f"\n🔬 Per-Fold Analysis:")
    print(f"{'Fold':<4} {'Train B/M':<10} {'Val B/M':<8} {'Test B/M':<9} {'Train Imgs':<10} {'Val Imgs':<8} {'Subtypes':<15} {'Issues'}")
    print("-" * 85)
    
    fold_issues = []
    
    for fold_idx in range(5):
        train_pats, val_pats, test_pats = splitter.get_fold(fold_idx)
        
        # Analyze train set
        train_labels = [patient_dict[pid]['label'] for pid in train_pats]
        train_benign = train_labels.count(0)
        train_malignant = train_labels.count(1)
        train_images = sum(len(img_list) for pid in train_pats 
                          for img_list in patient_dict[pid]['images'].values())
        
        # Analyze val set  
        val_labels = [patient_dict[pid]['label'] for pid in val_pats]
        val_benign = val_labels.count(0)
        val_malignant = val_labels.count(1)
        val_images = sum(len(img_list) for pid in val_pats
                        for img_list in patient_dict[pid]['images'].values())
        
        # Analyze test set
        test_labels = [patient_dict[pid]['label'] for pid in test_pats]
        test_benign = test_labels.count(0)
        test_malignant = test_labels.count(1)
        
        # Subtype analysis
        train_subtypes = Counter([patient_dict[pid]['subtype'] for pid in train_pats])
        val_subtypes = Counter([patient_dict[pid]['subtype'] for pid in val_pats])
        
        # Issue detection
        issues = []
        
        # 1. Class imbalance issues
        train_ratio = train_malignant / max(train_benign, 1)
        val_ratio = val_malignant / max(val_benign, 1)
        if abs(train_ratio - val_ratio) > 0.5:
            issues.append("ImbalanceGap")
        
        # 2. Very small validation classes
        if val_benign < 3 or val_malignant < 10:
            issues.append("SmallValClass")
            
        # 3. Image count issues
        avg_train_imgs = train_images / len(train_pats) if train_pats else 0
        avg_val_imgs = val_images / len(val_pats) if val_pats else 0
        if abs(avg_train_imgs - avg_val_imgs) > 20:
            issues.append("ImageCountGap")
        
        # 4. Subtype dominance
        max_train_subtype = max(train_subtypes.values()) if train_subtypes else 0
        if max_train_subtype > len(train_pats) * 0.7:
            issues.append("SubtypeDominance")
            
        # 5. Missing magnifications
        missing_mags = 0
        for pid in val_pats:
            available = sum(1 for mag in ['40', '100', '200', '400'] 
                          if mag in patient_dict[pid]['images'] and len(patient_dict[pid]['images'][mag]) > 0)
            if available < 4:
                missing_mags += 1
        if missing_mags > len(val_pats) * 0.3:
            issues.append("MissingMags")
        
        fold_issues.append({
            'fold': fold_idx,
            'issues': issues,
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'val_benign': val_benign,
            'val_malignant': val_malignant,
            'avg_train_imgs': avg_train_imgs,
            'avg_val_imgs': avg_val_imgs
        })
        
        print(f"{fold_idx:<4} {train_benign}/{train_malignant:<9} {val_benign}/{val_malignant:<7} {test_benign}/{test_malignant:<8} "
              f"{train_images:<10} {val_images:<8} {len(train_subtypes):<15} {','.join(issues) if issues else 'None'}")
    
    # Detailed issue analysis
    print(f"\n🚨 Detailed Issue Analysis:")
    
    problematic_folds = [f for f in fold_issues if f['issues']]
    if problematic_folds:
        for fold_data in problematic_folds:
            print(f"\nFold {fold_data['fold']} Issues:")
            for issue in fold_data['issues']:
                if issue == "SmallValClass":
                    print(f"  • Small validation classes: {fold_data['val_benign']} benign, {fold_data['val_malignant']} malignant")
                elif issue == "ImbalanceGap":
                    print(f"  • Class ratio mismatch: Train {fold_data['train_ratio']:.1f}, Val {fold_data['val_ratio']:.1f}")
                elif issue == "ImageCountGap":
                    print(f"  • Image count gap: Train avg {fold_data['avg_train_imgs']:.1f}, Val avg {fold_data['avg_val_imgs']:.1f}")
                else:
                    print(f"  • {issue}")
    else:
        print("No major structural issues detected.")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    
    if any('SmallValClass' in f['issues'] for f in fold_issues):
        print("  1. Small validation classes detected - consider stratified sampling with minimum class sizes")
    
    if any('ImbalanceGap' in f['issues'] for f in fold_issues):
        print("  2. Class ratio mismatches between train/val - improve stratification")
        
    if any('ImageCountGap' in f['issues'] for f in fold_issues):
        print("  3. Uneven image distribution - some patients have much more data")
        
    if any('MissingMags' in f['issues'] for f in fold_issues):
        print("  4. Missing magnifications in validation - affects model consistency")
    
    # Show most problematic patients (outliers)
    print(f"\n📋 Patient Data Distribution:")
    images_per_patient = [(pid, sum(len(imgs) for imgs in data['images'].values())) 
                         for pid, data in patient_dict.items()]
    images_per_patient.sort(key=lambda x: x[1], reverse=True)
    
    print("Top 5 patients with most images:")
    for pid, img_count in images_per_patient[:5]:
        label = "Malignant" if patient_dict[pid]['label'] == 1 else "Benign"
        subtype = patient_dict[pid]['subtype']
        print(f"  {pid}: {img_count} images ({label}, {subtype})")
        
    print("Bottom 5 patients with least images:")
    for pid, img_count in images_per_patient[-5:]:
        label = "Malignant" if patient_dict[pid]['label'] == 1 else "Benign"
        subtype = patient_dict[pid]['subtype']
        print(f"  {pid}: {img_count} images ({label}, {subtype})")

if __name__ == "__main__":
    analyze_fold_distribution()