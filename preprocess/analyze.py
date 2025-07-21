import os
from collections import defaultdict
import pandas as pd

from config import SLIDES_PATH


def analyze_dataset():
    patient_image_counts = defaultdict(int)
    magnification_counts = defaultdict(int)
    class_counts = defaultdict(int)
    class_patient_ids = defaultdict(set)
    subtype_counts = defaultdict(int)
    subtype_patient_ids = defaultdict(set)
    mag_class_counts = defaultdict(lambda: defaultdict(int))

    for root, dirs, files in os.walk(SLIDES_PATH):
        for file in files:
            if not file.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
                continue

            path = os.path.join(root, file)
            rel_path = path.replace(SLIDES_PATH, '')
            parts = rel_path.strip(os.sep).split(os.sep)

            if len(parts) < 6:
                continue
            tumor_class = parts[0]
            subtype = parts[2]
            patient_id = parts[3]
            magnification = parts[4]

            patient_key = patient_id
            patient_image_counts[patient_key] += 1
            magnification_counts[magnification] += 1
            class_counts[tumor_class] += 1
            class_patient_ids[tumor_class].add(patient_key)
            subtype_counts[subtype] += 1
            subtype_patient_ids[subtype].add(patient_key)
            mag_class_counts[magnification][tumor_class] += 1
    summary = {
        "Total Patients": len(patient_image_counts),
        "Benign Patients": len(class_patient_ids["benign"]),
        "Malignant Patients": len(class_patient_ids["malignant"]),
        "Total Images": sum(patient_image_counts.values()),
        "Images per Class": dict(class_counts),
        "Images per Magnification": dict(magnification_counts),
    }

    patient_df = pd.DataFrame([
        {"Patient ID": pid, "Total Images": count}
        for pid, count in patient_image_counts.items()
    ])
    mag_class_df = pd.DataFrame(mag_class_counts).fillna(0).astype(int).T
    subtype_df = pd.DataFrame([
        {"Tumor Subtype": subtype, "Patients": len(patients), "Total Images": subtype_counts[subtype]}
        for subtype, patients in subtype_patient_ids.items()
    ]).sort_values(by="Total Images", ascending=False)

    # patient_df.to_csv("output/patient_image_counts.csv", index=False)
    # mag_class_df.to_csv("output/magnification_class_distribution.csv")
    # subtype_df.to_csv("output/subtype_stats.csv", index=False)

    print("\n=== Dataset Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\nImages per Magnification per Class:\n", mag_class_df)
    print("\nTop Tumor Subtypes:\n", subtype_df.head(8))
