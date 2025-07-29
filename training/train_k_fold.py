import os
import json
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score

from config import config
from backbones.our import create_lightweight_model
from preprocess.kfold_splitter import PatientWiseKFoldSplitter
from preprocess.multimagset import MultiMagDataset
from preprocess.preprocess import get_transforms


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    for images, labels in dataloader:
        images = {k: v.to(device) for k, v in images.items()}
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        _, preds = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc, epoch_balanced_acc


def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = {k: v.to(device) for k, v in images.items()}
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc, epoch_balanced_acc


def run_training():
    # Create output directories
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # Get transforms
    train_transform, val_transform = get_transforms()

    # Initialize splitter
    splitter = PatientWiseKFoldSplitter(
        dataset_dir=config.DATASET_DIR,
        n_splits=config.N_SPLITS,
        random_state=config.RANDOM_STATE,
        stratify_subtype=config.STRATIFY_SUBTYPE,
        validation_split=config.VALIDATION_SPLIT
    )

    all_fold_results = []

    for fold in range(config.N_SPLITS):
        print(f"--- Fold {fold + 1}/{config.N_SPLITS} ---")

        # Get data splits
        train_pats, val_pats, test_pats = splitter.get_fold(fold, return_type='patients')

        # Create datasets
        train_dataset = MultiMagDataset(train_pats, splitter.patient_dict, transform=train_transform)
        val_dataset = MultiMagDataset(val_pats, splitter.patient_dict, transform=val_transform)
        test_dataset = MultiMagDataset(test_pats, splitter.patient_dict, transform=val_transform)

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
        test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

        # Initialize model
        model = create_lightweight_model(**config.get_model_config()).to(config.DEVICE)

        # Loss function with weights for class imbalance
        if config.USE_WEIGHTED_LOSS:
            train_labels = [splitter.patient_dict[pid]['label'] for pid in train_pats]
            class_counts = pd.Series(train_labels).value_counts()
            class_weights = torch.tensor([len(train_labels) / class_counts[i] for i in range(len(class_counts))], dtype=torch.float32).to(config.DEVICE)
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = torch.nn.CrossEntropyLoss()

        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)

        best_val_balanced_acc = 0.0
        fold_logs = []

        for epoch in range(config.NUM_EPOCHS):
            train_loss, train_acc, train_balanced_acc = train_one_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
            val_loss, val_acc, val_balanced_acc = validate_one_epoch(model, val_loader, criterion, config.DEVICE)

            scheduler.step()

            print(f"Epoch {epoch + 1}/{config.NUM_EPOCHS} | "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Bal Acc: {train_balanced_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Bal Acc: {val_balanced_acc:.4f}")

            # Log epoch results
            fold_logs.append({
                "epoch": epoch + 1,
                "train_loss": train_loss, "train_acc": train_acc, "train_balanced_acc": train_balanced_acc,
                "val_loss": val_loss, "val_acc": val_acc, "val_balanced_acc": val_balanced_acc
            })

            # Save best model
            if val_balanced_acc > best_val_balanced_acc:
                best_val_balanced_acc = val_balanced_acc
                torch.save(model.state_dict(), os.path.join(config.MODELS_DIR, f"best_model_fold_{fold}.pth"))

        # Save fold logs
        pd.DataFrame(fold_logs).to_csv(os.path.join(config.LOGS_DIR, f"fold_{fold}_logs.csv"), index=False)

        # Test best model
        model.load_state_dict(torch.load(os.path.join(config.MODELS_DIR, f"best_model_fold_{fold}.pth")))
        test_loss, test_acc, test_balanced_acc = validate_one_epoch(model, test_loader, criterion, config.DEVICE)

        # Get detailed test metrics
        model.eval()
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for images, labels in test_loader:
                images = {k: v.to(config.DEVICE) for k, v in images.items()}
                labels = labels.to(config.DEVICE)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        test_precision = precision_score(all_labels, all_preds, average='weighted')
        test_recall = recall_score(all_labels, all_preds, average='weighted')
        test_f1 = f1_score(all_labels, all_preds, average='weighted')

        fold_results = {
            "fold": fold,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "test_balanced_accuracy": test_balanced_acc,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1_score": test_f1
        }
        all_fold_results.append(fold_results)

        # Save fold results
        with open(os.path.join(config.RESULTS_DIR, f"fold_{fold}_results.json"), 'w') as f:
            json.dump(fold_results, f, indent=2)

        # Save test predictions
        pd.DataFrame({'labels': all_labels, 'predictions': all_preds}).to_csv(os.path.join(config.RESULTS_DIR, f"fold_{fold}_predictions.csv"), index=False)

    # Save all fold results
    with open(os.path.join(config.RESULTS_DIR, "all_fold_results.json"), 'w') as f:
        json.dump(all_fold_results, f, indent=2)

    print("--- Cross-Validation Summary ---")
    df_results = pd.DataFrame(all_fold_results)
    print(df_results.mean())

if __name__ == "__main__":
    run_training()