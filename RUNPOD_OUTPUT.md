MMNet - Multi-Magnification Network for Breast Cancer Classification
Using device: cuda
Batch size: 16
Learning rate: 5e-05

Dataset Analysis:

=== Dataset Summary ===
Total Patients: 82
Benign Patients: 24
Malignant Patients: 58
Total Images: 7909
Images per Class: {'benign': 2480, 'malignant': 5429}
Images per Magnification: {'200X': 2013, '40X': 1995, '400X': 1820, '100X': 2081}

Images per Magnification per Class:
       benign  malignant
200X     623       1390
40X      625       1370
400X     588       1232
100X     644       1437

Top Tumor Subtypes:
          Tumor Subtype  Patients  Total Images
5     ductal_carcinoma        38          3451
1         fibroadenoma        10          1014
7   mucinous_carcinoma         9           792
4    lobular_carcinoma         5           626
2      tubular_adenoma         7           569
6  papillary_carcinoma         6           560
0      phyllodes_tumor         3           453
3             adenosis         4           444
=== Fold-wise Dataset Summary ===
Fold 0: Train images: 6246 (B/M = 1913/4333); Test images: 1663 (B/M = 567/1096)
Fold 1: Train images: 6228 (B/M = 1928/4300); Test images: 1681 (B/M = 552/1129)
Fold 2: Train images: 6407 (B/M = 2005/4402); Test images: 1502 (B/M = 475/1027)
Fold 3: Train images: 6217 (B/M = 1913/4304); Test images: 1692 (B/M = 567/1125)
Fold 4: Train images: 6538 (B/M = 2161/4377); Test images: 1371 (B/M = 319/1052)

===== Fold 0 =====
Train patients: 65, Test patients: 17
Training samples per epoch: 389 (avg utilization: 28.4%)
Class weights: Benign=1.71, Malignant=0.71
model.safetensors: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 31.5M/31.5M [00:00<00:00, 169MB/s]
Inner training samples: 270, batch size: 24
Inner split: Train 45, Val 20 patients
Epoch 01: Train: Loss 0.1148, Acc 0.508 |Val: Loss 0.1271, Acc 0.450, BalAcc 0.560, F1 0.421, AUC 0.369, Thresh 0.772
New best validation balanced accuracy: 0.560, threshold: 0.772
Epoch 02: Train: Loss 0.1071, Acc 0.492 |Val: Loss 0.0860, Acc 0.550, BalAcc 0.679, F1 0.526, AUC 0.548, Thresh 0.800
New best validation balanced accuracy: 0.679, threshold: 0.800
Epoch 03: Train: Loss 0.1020, Acc 0.545 |Val: Loss 0.0935, Acc 0.600, BalAcc 0.571, F1 0.692, AUC 0.417, Thresh 0.521
Epoch 04: Train: Loss 0.1006, Acc 0.610 |Val: Loss 0.0633, Acc 0.750, BalAcc 0.774, F1 0.800, AUC 0.679, Thresh 0.490
New best validation balanced accuracy: 0.774, threshold: 0.490
Epoch 05: Train: Loss 0.0820, Acc 0.561 |Val: Loss 0.0826, Acc 0.350, BalAcc 0.536, F1 0.133, AUC 0.429, Thresh 0.900
Epoch 06: Train: Loss 0.0896, Acc 0.553 |Val: Loss 0.0518, Acc 0.900, BalAcc 0.833, F1 0.933, AUC 0.810, Thresh 0.311
New best validation balanced accuracy: 0.833, threshold: 0.311
Epoch 07: Train: Loss 0.0689, Acc 0.598 |Val: Loss 0.0357, Acc 0.800, BalAcc 0.857, F1 0.833, AUC 0.857, Thresh 0.537
New best validation balanced accuracy: 0.857, threshold: 0.537
Epoch 08: Train: Loss 0.0743, Acc 0.602 |Val: Loss 0.0754, Acc 0.600, BalAcc 0.619, F1 0.667, AUC 0.512, Thresh 0.460
Epoch 09: Train: Loss 0.0938, Acc 0.568 |Val: Loss 0.0536, Acc 0.550, BalAcc 0.679, F1 0.526, AUC 0.631, Thresh 0.760
Epoch 10: Train: Loss 0.0741, Acc 0.602 |Val: Loss 0.0310, Acc 0.800, BalAcc 0.857, F1 0.833, AUC 0.857, Thresh 0.523
Epoch 11: Train: Loss 0.0694, Acc 0.595 |Val: Loss 0.1005, Acc 0.600, BalAcc 0.667, F1 0.636, AUC 0.548, Thresh 0.653
Epoch 12: Train: Loss 0.0749, Acc 0.561 |Val: Loss 0.0329, Acc 0.750, BalAcc 0.726, F1 0.815, AUC 0.726, Thresh 0.485
Early stopping after 12 epochs (no improvement for 5 epochs)
Best model saved: ./output/models/best_model_fold_0.pth (Val BalAcc: 0.857)
Test Results: Acc 0.471, BalAcc 0.625, F1 0.400, AUC 0.933 (threshold: 0.537)

===== Fold 1 =====
Train patients: 65, Test patients: 17
Training samples per epoch: 388 (avg utilization: 29.1%)
Class weights: Benign=1.71, Malignant=0.71
Inner training samples: 268, batch size: 24
Inner split: Train 45, Val 20 patients
Epoch 01: Train: Loss 0.1065, Acc 0.561 |Val: Loss 0.0799, Acc 0.750, BalAcc 0.583, F1 0.848, AUC 0.440, Thresh 0.107
New best validation balanced accuracy: 0.583, threshold: 0.107
Epoch 02: Train: Loss 0.0940, Acc 0.557 |Val: Loss 0.0474, Acc 0.500, BalAcc 0.643, F1 0.444, AUC 0.595, Thresh 0.872
New best validation balanced accuracy: 0.643, threshold: 0.872
...
...