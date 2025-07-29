#!/usr/bin/env python3
"""
Test JSON serialization fix for numpy arrays
"""

import numpy as np
import tempfile
import os
from utils.stats import save_as_json

def test_json_serialization():
    print("🔧 Testing JSON serialization fix for numpy arrays...")
    
    # Create test data with various numpy types
    test_data = {
        'confusion_matrix': np.array([[130, 2], [27, 194]]),
        'fpr': np.array([0.0, 0.015, 1.0]),
        'tpr': np.array([0.0, 0.878, 1.0]),
        'thresholds': np.array([1.996, 0.496, 0.0]),
        'accuracy': np.float64(0.918),
        'balanced_accuracy': np.float64(0.931),
        'f1_score': np.float64(0.930),
        'precision': np.float64(0.990),
        'recall': np.float64(0.878),
        'auc': np.float64(0.992),
        'avg_inference_time': np.float64(0.0085),
        'fold': np.int64(0),
        'train_patients': 52,
        'test_patients': 17,
        'regular_string': "test",
        'regular_list': [1, 2, 3],
        'regular_dict': {'key': 'value'}
    }
    
    try:
        # Test saving to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        save_as_json(test_data, temp_path)
        
        # Test loading back
        import json
        with open(temp_path, 'r') as f:
            loaded_data = json.load(f)
        
        print("✅ JSON serialization successful!")
        print(f"✅ Original confusion matrix shape: {test_data['confusion_matrix'].shape}")
        print(f"✅ Loaded confusion matrix: {loaded_data['confusion_matrix']}")
        print(f"✅ Original accuracy type: {type(test_data['accuracy'])}")
        print(f"✅ Loaded accuracy type: {type(loaded_data['accuracy'])}")
        print(f"✅ Accuracy value preserved: {loaded_data['accuracy']}")
        
        # Cleanup
        os.unlink(temp_path)
        
        return True
        
    except Exception as e:
        print(f"❌ JSON serialization failed: {e}")
        if 'temp_path' in locals():
            try:
                os.unlink(temp_path)
            except:
                pass
        return False

if __name__ == "__main__":
    success = test_json_serialization()
    if success:
        print(f"\n🎉 JSON serialization fix successful! The TypeError should be resolved.")
    else:
        print(f"\n💥 JSON serialization still has issues.")