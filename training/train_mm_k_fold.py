
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
import torch
from tqdm import tqdm


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    losses = []
    all_preds, all_labels = [], []
    for images_dict, labels in tqdm(dataloader, desc='Train', leave=False):
        # move to device
        images = {k: v.to(device) for k, v in images_dict.items()}
        labels = labels.to(device)
        optimizer.zero_grad()
        logits, _ = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    return np.mean(losses), acc

def eval_model(model, dataloader, criterion, device):
    model.eval()
    losses = []
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images_dict, labels in tqdm(dataloader, desc='Eval ', leave=False):
            images = {k: v.to(device) for k, v in images_dict.items()}
            labels = labels.to(device)
            logits, _ = model(images)
            loss = criterion(logits, labels)

            losses.append(loss.item())
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    return np.mean(losses), acc, bal_acc, f1, auc