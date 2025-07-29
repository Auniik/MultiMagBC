#!/usr/bin/env python3
"""
Test improved class weighting approaches
"""

from config import calculate_class_weights

def test_class_weights():
    # Simulate typical BreakHis dataset distribution
    # 24 benign patients, 58 malignant patients (from RUNPOD_OUTPUT.md)
    train_labels = [0] * 24 + [1] * 58  # 0=benign, 1=malignant
    
    print("🧪 Testing Class Weighting Methods")
    print(f"📊 Dataset: {train_labels.count(0)} benign, {train_labels.count(1)} malignant patients")
    print(f"   Imbalance ratio: {train_labels.count(1) / train_labels.count(0):.1f}:1 (malignant:benign)")
    print()
    
    methods = ['balanced', 'moderate', 'aggressive']
    
    for method in methods:
        print(f"🔧 Method: {method}")
        weights = calculate_class_weights(train_labels, method=method)
        
        ratio = weights[0] / weights[1]
        
        print(f"   Expected impact: {'Balanced' if ratio < 3 else 'High benign emphasis' if ratio < 8 else 'Extreme benign emphasis'}")
        print()

if __name__ == "__main__":
    test_class_weights()