from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

def per_subtype_metrics(model, dataloader, device, threshold, patient_dict):
    model.eval()
    all_labels, all_preds, all_subtypes = [], [], []
    with torch.no_grad():
        for images_dict, mask, labels in dataloader:
            images_dict = {k: v.to(device) for k, v in images_dict.items()}
            mask, labels = mask.to(device), labels.to(device)
            outputs = model(images_dict, mask)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            probs = torch.sigmoid(logits)[:,1].cpu().numpy()
            preds = (probs >= threshold).astype(int)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
            # Map patient to subtype
            batch_subtypes = [patient_dict[pid]['subtype'] for pid in dataloader.dataset.patient_ids[:len(labels)]]
            all_subtypes.extend(batch_subtypes)

    metrics_per_subtype = {}
    for subtype in set(all_subtypes):
        idx = [i for i, s in enumerate(all_subtypes) if s == subtype]
        if not idx:
            continue
        y_true = np.array(all_labels)[idx]
        y_pred = np.array(all_preds)[idx]
        metrics_per_subtype[subtype] = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0)
        }
    return metrics_per_subtype