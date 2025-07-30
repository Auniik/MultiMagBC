#!/usr/bin/env python3
"""
Test GPU optimization improvements
"""

import torch
import time
from config import get_training_config
from preprocess.kfold_splitter import PatientWiseKFoldSplitter
from preprocess.multimagset import MultiMagPatientDataset
from preprocess.preprocess import get_transforms
from backbones.our.lightweight import MultiMagLightweightCNN
from torch.utils.data import DataLoader
from config import SLIDES_PATH

def test_gpu_optimization():
    print("🚀 Testing GPU Optimization Improvements")
    
    config = get_training_config()
    device = config['device']
    
    if device.type != 'cuda':
        print("❌ CUDA not available. GPU optimization test requires CUDA.")
        return
    
    print(f"📊 Configuration:")
    print(f"   Device: {device}")
    print(f"   Batch size: {config['batch_size']}")
    print(f"   Num workers: {config['num_workers']}")
    print(f"   Pin memory: {config['pin_memory']}")
    print(f"   Persistent workers: {config.get('persistent_workers', False)}")
    
    # Create test data
    splitter = PatientWiseKFoldSplitter(
        dataset_dir=SLIDES_PATH,
        n_splits=5,
        validation_split=0.25
    )
    
    train_pats, val_pats, test_pats = splitter.get_fold(0)
    patient_dict = splitter.patient_dict
    train_transform, eval_transform, _ = get_transforms()
    
    # Create dataset
    train_ds = MultiMagPatientDataset(patient_dict, train_pats, transform=train_transform, mode='train')
    
    # Create optimized data loader
    samples_per_epoch = train_ds.get_total_samples()
    effective_batch_size = min(max(32, samples_per_epoch // 150), 64)
    
    print(f"   Effective batch size: {effective_batch_size}")
    print(f"   Total training samples: {samples_per_epoch}")
    
    sampler = train_ds.get_class_balanced_sampler()
    train_loader = DataLoader(
        train_ds, batch_size=effective_batch_size,
        sampler=sampler if sampler else None,
        shuffle=(sampler is None),
        num_workers=config['num_workers'], 
        pin_memory=config['pin_memory'],
        persistent_workers=config.get('persistent_workers', False),
        drop_last=True
    )
    
    # Create model
    model = MultiMagLightweightCNN(num_classes=2).to(device)
    print(f"🧠 Model parameters: {model.get_model_info()['total_parameters']:,}")
    
    # Test data loading speed
    print(f"\n⏱️ Testing data loading speed...")
    model.eval()
    
    start_time = time.time()
    batch_count = 0
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, (images_dict, mask, labels) in enumerate(train_loader):
            if batch_idx >= 10:  # Test first 10 batches
                break
                
            # Move to GPU
            images = {k: v.to(device, non_blocking=True) for k, v in images_dict.items()}
            mask = mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Forward pass
            with torch.cuda.amp.autocast(enabled=True):
                outputs = model(images, mask)
            
            batch_count += 1
            sample_count += labels.size(0)
            
            if batch_idx == 0:
                print(f"✅ First batch processed successfully:")
                print(f"   Batch size: {labels.size(0)}")
                print(f"   Input shapes: {[f'{k}:{v.shape}' for k, v in images.items()]}")
                print(f"   Output shape: {outputs.shape}")
    
    end_time = time.time()
    total_time = end_time - start_time
    samples_per_sec = sample_count / total_time
    batches_per_sec = batch_count / total_time
    
    print(f"\n📈 Performance Results:")
    print(f"   Processed {batch_count} batches ({sample_count} samples) in {total_time:.2f}s")
    print(f"   Speed: {samples_per_sec:.1f} samples/sec, {batches_per_sec:.1f} batches/sec")
    
    # GPU memory usage
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(device) / 1e9
        memory_reserved = torch.cuda.memory_reserved(device) / 1e9
        print(f"   GPU memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
    
    print(f"\n🎯 Optimization Status:")
    if effective_batch_size >= 32:
        print(f"   ✅ Good batch size: {effective_batch_size}")
    else:
        print(f"   ⚠️ Small batch size: {effective_batch_size}")
        
    if samples_per_sec > 50:
        print(f"   ✅ Good throughput: {samples_per_sec:.1f} samples/sec")
    else:
        print(f"   ⚠️ Low throughput: {samples_per_sec:.1f} samples/sec")
    
    return True

if __name__ == "__main__":
    test_gpu_optimization()