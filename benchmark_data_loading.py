#!/usr/bin/env python3
"""
Benchmark data loading performance to identify bottlenecks
"""

import time
import torch
from torch.utils.data import DataLoader
from config import get_training_config, SLIDES_PATH
from preprocess.kfold_splitter import PatientWiseKFoldSplitter
from preprocess.multimagset import MultiMagPatientDataset
from preprocess.preprocess import get_transforms
from preprocess.preprocess_light import get_fast_transforms

def benchmark_data_loading():
    print("⏱️ Benchmarking Data Loading Performance")
    
    config = get_training_config()
    device = config['device']
    
    if device.type != 'cuda':
        print("❌ Need CUDA for 4090 optimization")
        return
    
    # Setup data
    splitter = PatientWiseKFoldSplitter(SLIDES_PATH, n_splits=5, validation_split=0.25)
    train_pats, val_pats, test_pats = splitter.get_fold(0)
    patient_dict = splitter.patient_dict
    
    # Test both transform versions
    transforms_to_test = [
        ("Original Heavy", get_transforms),
        ("Optimized Light", get_fast_transforms)
    ]
    
    results = {}
    
    for name, transform_fn in transforms_to_test:
        print(f"\n🧪 Testing {name} Transforms:")
        
        train_transform, eval_transform, _ = transform_fn()
        train_ds = MultiMagPatientDataset(patient_dict, train_pats, transform=train_transform, mode='train')
        
        # Test different configurations
        configs = [
            {"batch_size": 32, "num_workers": 8, "prefetch_factor": 2},
            {"batch_size": 32, "num_workers": 16, "prefetch_factor": 4},
            {"batch_size": 64, "num_workers": 16, "prefetch_factor": 4},
        ]
        
        for i, loader_config in enumerate(configs):
            print(f"  Config {i+1}: {loader_config}")
            
            sampler = train_ds.get_class_balanced_sampler()
            train_loader = DataLoader(
                train_ds, 
                batch_size=loader_config["batch_size"],
                sampler=sampler,
                num_workers=loader_config["num_workers"],
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=loader_config["prefetch_factor"],
                drop_last=True
            )
            
            # Benchmark loading speed
            start_time = time.time()
            batch_times = []
            
            for batch_idx, (images_dict, mask, labels) in enumerate(train_loader):
                if batch_idx >= 20:  # Test first 20 batches
                    break
                    
                batch_start = time.time()
                
                # Simulate GPU transfer
                images = {k: v.to(device, non_blocking=True) for k, v in images_dict.items()}
                mask = mask.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                # Simulate forward pass time
                torch.cuda.synchronize()  # Wait for transfer
                
                batch_time = time.time() - batch_start
                batch_times.append(batch_time)
                
                if batch_idx == 0:
                    print(f"    First batch: {labels.size(0)} samples")
                elif batch_idx % 5 == 0:
                    avg_time = sum(batch_times[-5:]) / len(batch_times[-5:])
                    print(f"    Batch {batch_idx}: {avg_time:.3f}s/batch")
            
            total_time = time.time() - start_time
            avg_batch_time = sum(batch_times) / len(batch_times)
            samples_per_sec = (len(batch_times) * loader_config["batch_size"]) / total_time
            
            result_key = f"{name}_config_{i+1}"
            results[result_key] = {
                "avg_batch_time": avg_batch_time,
                "samples_per_sec": samples_per_sec,
                "config": loader_config
            }
            
            print(f"    ⚡ {avg_batch_time:.3f}s/batch, {samples_per_sec:.1f} samples/sec")
    
    print(f"\n📊 PERFORMANCE COMPARISON:")
    print(f"{'Configuration':<30} {'Batch Time':<12} {'Samples/sec':<12} {'Speedup':<10}")
    print("-" * 70)
    
    baseline = None
    for key, result in results.items():
        if baseline is None:
            baseline = result["samples_per_sec"]
            speedup = "1.0x"
        else:
            speedup = f"{result['samples_per_sec']/baseline:.1f}x"
        
        print(f"{key:<30} {result['avg_batch_time']:.3f}s{'':<7} {result['samples_per_sec']:.1f}{'':<7} {speedup}")
    
    # Recommendations
    best_config = max(results.items(), key=lambda x: x[1]["samples_per_sec"])
    print(f"\n🏆 BEST CONFIGURATION: {best_config[0]}")
    print(f"   Speed: {best_config[1]['samples_per_sec']:.1f} samples/sec")
    print(f"   Config: {best_config[1]['config']}")
    
    if best_config[1]["samples_per_sec"] > 100:
        print(f"   ✅ Excellent performance for 4090!")
    elif best_config[1]["samples_per_sec"] > 50:
        print(f"   👍 Good performance")
    else:
        print(f"   ⚠️ Still has bottlenecks")

if __name__ == "__main__":
    benchmark_data_loading()