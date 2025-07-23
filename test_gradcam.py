"""
Minimal GradCAM Testing Pipeline
Quick testing script for GradCAM visualization with reduced dataset
"""

import os
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torchvision.transforms as T

from backbones.our.model import MMNet
from preprocess.multimagset import MultiMagPatientDataset
from preprocess.kfold_splitter import PatientWiseKFoldSplitter
from evaluate.gradcam import GradCAM, visualize_gradcam
from config import get_training_config, FocalLoss, calculate_class_weights
from utils.helpers import seed_everything


def create_minimal_dataset(n_patients_per_class=5):
    """Create a minimal dataset for quick testing"""
    print("Creating minimal dataset...")
    
    # Initialize splitter with full dataset using config paths
    from config import SLIDES_PATH
    print(f"Looking for dataset at: {SLIDES_PATH}")
    
    # Check if path exists
    import os
    if not os.path.exists(SLIDES_PATH):
        print(f"Dataset path not found: {SLIDES_PATH}")
        print("Please check your data directory structure")
        # Use a fallback or raise error
        return None, []
    
    splitter = PatientWiseKFoldSplitter(
        dataset_dir=SLIDES_PATH,
        n_splits=2,  # Minimum required splits
        stratify_subtype=False
    )
    
    patient_dict = splitter.patient_dict
    all_patients = list(patient_dict.keys())
    
    if not all_patients:
        print("No patients found in dataset")
        return None, []
    
    # Stratified sampling: 5 benign + 5 malignant patients
    benign_patients = [p for p in all_patients if patient_dict[p]['label'] == 0]
    malignant_patients = [p for p in all_patients if patient_dict[p]['label'] == 1]
    
    print(f"Found {len(benign_patients)} benign and {len(malignant_patients)} malignant patients")
    
    # Adjust requested patients based on availability
    actual_benign = min(n_patients_per_class, len(benign_patients))
    actual_malignant = min(n_patients_per_class, len(malignant_patients))
    
    selected_benign = benign_patients[:actual_benign]
    selected_malignant = malignant_patients[:actual_malignant]
    selected_patients = selected_benign + selected_malignant
    
    print(f"Selected {len(selected_patients)} patients: {len(selected_benign)} benign, {len(selected_malignant)} malignant")
    
    return patient_dict, selected_patients


def minimal_training_pipeline():
    """Run minimal training pipeline for GradCAM testing"""
    print("Starting minimal GradCAM testing pipeline...")
    
    # Setup
    config = get_training_config()
    device = config['device']
    seed_everything(42)
    
    print(f"Using device: {device}")
    
    # Create minimal dataset
    patient_dict, selected_patients = create_minimal_dataset(n_patients_per_class=5)
    
    if not selected_patients or patient_dict is None:
        print("Failed to create dataset. Exiting...")
        return
    
    # Split into train/test (80/20)
    train_patients, test_patients = train_test_split(
        selected_patients, 
        test_size=0.2, 
        random_state=42,
        stratify=[patient_dict[p]['label'] for p in selected_patients]
    )
    
    print(f"Train patients: {len(train_patients)}, Test patients: {len(test_patients)}")
    
    # Minimal transforms (no augmentation for speed)
    train_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    
    test_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    
    # Create datasets with minimal sampling
    train_ds = MultiMagPatientDataset(
        patient_dict, train_patients, transform=train_transform,
        samples_per_patient=2,  # Minimal samples
        epoch_multiplier=1,     # No epoch multiplier
        adaptive_sampling=False
    )
    
    test_ds = MultiMagPatientDataset(
        patient_dict, test_patients, transform=test_transform,
        samples_per_patient=1,  # Single sample per patient
        adaptive_sampling=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=4,  # Small batch for quick training
        shuffle=True,
        num_workers=0,  # Single process for simplicity
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_ds, 
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    
    print(f"Training samples: {len(train_ds)}, Test samples: {len(test_ds)}")
    
    # Model setup
    model = MMNet(dropout=0.5).to(device)
    
    # Calculate class weights
    train_labels = [patient_dict[pid]['label'] for pid in train_patients]
    class_weights = calculate_class_weights(train_labels).to(device)
    
    criterion = FocalLoss(alpha=0.5, gamma=2.0, weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # Minimal training (3 epochs max)
    best_acc = 0
    for epoch in range(3):
        print(f"\nEpoch {epoch+1}/3")
        
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for images_dict, labels in train_loader:
            images = {k: v.to(device) for k, v in images_dict.items()}
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        print(f"Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}%")
        
        # Quick evaluation
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for images_dict, labels in test_loader:
                images = {k: v.to(device) for k, v in images_dict.items()}
                labels = labels.to(device)
                
                outputs = model(images)
                _, predicted = outputs.max(1)
                
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        test_acc = 100. * test_correct / test_total
        print(f"Test Acc: {test_acc:.2f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'outputs/test_gradcam/best_model.pth')
    
    # GradCAM Testing
    print("\nGenerating GradCAM visualizations...")
    os.makedirs('outputs/test_gradcam', exist_ok=True)
    
    # Load best model
    if os.path.exists('outputs/test_gradcam/best_model.pth'):
        model.load_state_dict(torch.load('outputs/test_gradcam/best_model.pth'))
    
    # Generate GradCAM for test samples
    gradcam = GradCAM(model)
    model.eval()
    
    for i, (images_dict, labels) in enumerate(test_loader):
        if i >= 5:  # Generate for 5 samples max
            break
            
        print(f"Generating GradCAM for sample {i+1}...")
        
        # Move to device
        images = {k: v.to(device) for k, v in images_dict.items()}
        labels = labels.to(device)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(images)
            # Handle both single and multi-output cases
            if isinstance(outputs, tuple):
                logits = outputs[0]  # Binary classification output
            else:
                logits = outputs
            _, predicted = logits.max(1)
        
        # Generate GradCAM
        cams = gradcam.get_cam(images, target_class=predicted.item())
        
        # Visualize and save
        save_path = f'outputs/test_gradcam/gradcam_sample_{i}.png'
        visualize_gradcam(
            cams, 
            images, 
            true_label=labels.item(),
            pred_label=predicted.item(),
            save_path=save_path,
            show=False
        )
        
        print(f"Saved: {save_path}")
    
    print(f"\nGradCAM testing complete!")
    print(f"Results saved in: outputs/test_gradcam/")
    print(f"Generated {min(5, len(test_loader))} GradCAM visualizations")


if __name__ == "__main__":
    minimal_training_pipeline()