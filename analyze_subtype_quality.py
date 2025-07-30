#!/usr/bin/env python3
"""
Analyze subtype quality to identify candidates for removal to achieve 95%+ accuracy
"""

import os
from collections import Counter, defaultdict
import numpy as np
from config import SLIDES_PATH
from preprocess.kfold_splitter import PatientWiseKFoldSplitter

def analyze_subtype_quality():
    print("🔬 Analyzing Subtype Quality for 95%+ Accuracy Target")
    
    splitter = PatientWiseKFoldSplitter(
        dataset_dir=SLIDES_PATH,
        n_splits=5,
        validation_split=0.25
    )
    
    patient_dict = splitter.patient_dict
    
    # Group patients by subtype
    subtype_analysis = defaultdict(lambda: {
        'patients': [],
        'total_images': 0,
        'benign_count': 0,
        'malignant_count': 0,
        'avg_images_per_patient': 0,
        'min_images': float('inf'),
        'max_images': 0,
        'missing_mags': 0
    })
    
    for pid, data in patient_dict.items():
        subtype = data['subtype']
        total_imgs = sum(len(imgs) for imgs in data['images'].values())
        
        subtype_analysis[subtype]['patients'].append(pid)
        subtype_analysis[subtype]['total_images'] += total_imgs
        
        if data['label'] == 0:
            subtype_analysis[subtype]['benign_count'] += 1
        else:
            subtype_analysis[subtype]['malignant_count'] += 1
            
        subtype_analysis[subtype]['min_images'] = min(subtype_analysis[subtype]['min_images'], total_imgs)
        subtype_analysis[subtype]['max_images'] = max(subtype_analysis[subtype]['max_images'], total_imgs)
        
        # Check missing magnifications
        available_mags = sum(1 for mag in ['40', '100', '200', '400'] 
                           if mag in data['images'] and len(data['images'][mag]) > 0)
        if available_mags < 4:
            subtype_analysis[subtype]['missing_mags'] += 1
    
    # Calculate averages
    for subtype, stats in subtype_analysis.items():
        patient_count = len(stats['patients'])
        stats['avg_images_per_patient'] = stats['total_images'] / patient_count if patient_count > 0 else 0
        stats['missing_mags_pct'] = (stats['missing_mags'] / patient_count) * 100 if patient_count > 0 else 0
    
    # Sort by quality score (higher is better)
    def calculate_quality_score(stats):
        patient_count = len(stats['patients'])
        avg_images = stats['avg_images_per_patient']
        missing_pct = stats['missing_mags_pct']
        
        # Quality factors:
        # 1. Sufficient patients (penalty if < 5)
        patient_factor = min(patient_count / 5.0, 1.0)
        
        # 2. Good image availability (penalty if < 80 avg images)
        image_factor = min(avg_images / 80.0, 1.0)
        
        # 3. Complete magnifications (penalty for missing mags)
        mag_factor = (100 - missing_pct) / 100.0
        
        # Combined score (0-1, higher is better)
        quality_score = patient_factor * image_factor * mag_factor
        
        return quality_score
    
    # Analyze each subtype
    print(f"\n📊 Subtype Quality Analysis:")
    print(f"{'Subtype':<20} {'Patients':<8} {'Images':<7} {'Avg/Pat':<8} {'Missing%':<9} {'Quality':<8} {'Recommendation'}")
    print("-" * 95)
    
    quality_scores = []
    removal_candidates = []
    
    for subtype, stats in subtype_analysis.items():
        patient_count = len(stats['patients'])
        quality_score = calculate_quality_score(stats)
        quality_scores.append((subtype, quality_score, stats))
        
        # Determine recommendation
        if patient_count < 3:
            recommendation = "REMOVE (too few)"
            removal_candidates.append((subtype, "insufficient_patients", stats))
        elif stats['avg_images_per_patient'] < 40:
            recommendation = "REMOVE (low data)"
            removal_candidates.append((subtype, "insufficient_data", stats))
        elif stats['missing_mags_pct'] > 50:
            recommendation = "REMOVE (missing mags)"
            removal_candidates.append((subtype, "missing_magnifications", stats))
        elif quality_score < 0.3:
            recommendation = "CONSIDER removing"
            removal_candidates.append((subtype, "low_quality", stats))
        elif quality_score < 0.6:
            recommendation = "Monitor"
        else:
            recommendation = "Keep (good)"
        
        print(f"{subtype:<20} {patient_count:<8} {stats['total_images']:<7} "
              f"{stats['avg_images_per_patient']:<8.1f} {stats['missing_mags_pct']:<8.1f}% "
              f"{quality_score:<8.2f} {recommendation}")
    
    # Sort by quality score
    quality_scores.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🎯 Recommendations for 95%+ Accuracy:")
    
    if removal_candidates:
        print(f"\n❌ Suggested Removals ({len(removal_candidates)} subtypes):")
        current_patients = len(patient_dict)
        current_images = sum(sum(len(imgs) for imgs in data['images'].values()) for data in patient_dict.values())
        
        removed_patients = 0
        removed_images = 0
        
        for subtype, reason, stats in removal_candidates:
            removed_patients += len(stats['patients'])
            removed_images += stats['total_images']
            
            print(f"  • {subtype}: {len(stats['patients'])} patients, {stats['total_images']} images ({reason})")
        
        remaining_patients = current_patients - removed_patients
        remaining_images = current_images - removed_images
        
        print(f"\n📊 Impact of Removals:")
        print(f"  Before: {current_patients} patients, {current_images} images")
        print(f"  After:  {remaining_patients} patients, {remaining_images} images")
        print(f"  Removed: {removed_patients} patients ({removed_patients/current_patients*100:.1f}%), "
              f"{removed_images} images ({removed_images/current_images*100:.1f}%)")
        
        # Calculate new class distribution
        remaining_benign = 0
        remaining_malignant = 0
        
        for pid, data in patient_dict.items():
            if data['subtype'] not in [r[0] for r in removal_candidates]:
                if data['label'] == 0:
                    remaining_benign += 1
                else:
                    remaining_malignant += 1
        
        print(f"  Class balance after removal: {remaining_benign} benign, {remaining_malignant} malignant")
        print(f"  New ratio: {remaining_malignant/max(remaining_benign,1):.1f}:1 (malignant:benign)")
    
    else:
        print("  No clear candidates for removal identified.")
    
    # Additional strategies
    print(f"\n🚀 Additional Strategies for 95%+ Accuracy:")
    
    print(f"1. **Focus on High-Quality Subtypes:**")
    best_subtypes = [item[0] for item in quality_scores[:5]]
    print(f"   Top subtypes: {', '.join(best_subtypes)}")
    
    print(f"\n2. **Magnification Filtering:**")
    complete_mag_patients = sum(1 for data in patient_dict.values() 
                               if sum(1 for mag in ['40', '100', '200', '400'] 
                                     if mag in data['images'] and len(data['images'][mag]) > 0) == 4)
    print(f"   Patients with all 4 magnifications: {complete_mag_patients}/{len(patient_dict)} ({complete_mag_patients/len(patient_dict)*100:.1f}%)")
    
    print(f"\n3. **Image Quality Thresholds:**")
    high_quality_patients = [pid for pid, data in patient_dict.items() 
                           if sum(len(imgs) for imgs in data['images'].values()) >= 80]
    print(f"   Patients with ≥80 images: {len(high_quality_patients)}/{len(patient_dict)} ({len(high_quality_patients)/len(patient_dict)*100:.1f}%)")
    
    print(f"\n4. **Conservative Splitting:**")
    print(f"   Consider 70-15-15 split for more stable training")
    print(f"   Use only patients with complete data for validation/test")
    
    # Generate filtered dataset recommendation
    if removal_candidates:
        print(f"\n💡 **RECOMMENDED ACTION:**")
        print(f"   1. Remove {len(removal_candidates)} low-quality subtypes")
        print(f"   2. Keep only patients with ≥60 images")
        print(f"   3. Require all 4 magnifications")
        print(f"   4. Expected remaining: ~{len(high_quality_patients)} high-quality patients")
        print(f"   5. Should achieve 95%+ accuracy with this cleaner dataset")
    
    return removal_candidates, quality_scores

if __name__ == "__main__":
    removal_candidates, quality_scores = analyze_subtype_quality()