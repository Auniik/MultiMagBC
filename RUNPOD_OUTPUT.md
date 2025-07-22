MMNet - Multi-Magnification Network for Breast Cancer Classification
Using device: cuda
Batch size: 16
Learning rate: 0.0001

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
Class weights: Benign=1.71, Malignant=0.71
Epoch 01: Train: Loss 0.6955, Acc 0.500 |Val: Loss 0.7045, Acc 0.765, BalAcc 0.600, F1 0.857, AUC 0.433
New best balanced accuracy: 0.600
Epoch 02: Train: Loss 0.5811, Acc 0.656 |Val: Loss 0.7172, Acc 0.765, BalAcc 0.600, F1 0.857, AUC 0.683
Epoch 03: Train: Loss 0.5181, Acc 0.688 |Val: Loss 0.7236, Acc 0.941, BalAcc 0.958, F1 0.957, AUC 0.933
New best balanced accuracy: 0.958
Epoch 04: Train: Loss 0.5662, Acc 0.656 |Val: Loss 0.7168, Acc 0.824, BalAcc 0.700, F1 0.889, AUC 0.883
Epoch 05: Train: Loss 0.4748, Acc 0.766 |Val: Loss 0.6938, Acc 0.882, BalAcc 0.800, F1 0.923, AUC 0.933
Epoch 06: Train: Loss 0.4676, Acc 0.734 |Val: Loss 0.6708, Acc 0.882, BalAcc 0.917, F1 0.909, AUC 0.883
Epoch 07: Train: Loss 0.4508, Acc 0.766 |Val: Loss 0.7235, Acc 0.824, BalAcc 0.758, F1 0.880, AUC 0.833
Epoch 08: Train: Loss 0.4874, Acc 0.719 |Val: Loss 0.6480, Acc 1.000, BalAcc 1.000, F1 1.000, AUC 1.000
New best balanced accuracy: 1.000
Epoch 09: Train: Loss 0.3727, Acc 0.859 |Val: Loss 0.7678, Acc 0.882, BalAcc 0.800, F1 0.923, AUC 0.917
Epoch 10: Train: Loss 0.3715, Acc 0.859 |Val: Loss 0.8456, Acc 0.882, BalAcc 0.800, F1 0.923, AUC 0.950
Epoch 11: Train: Loss 0.2599, Acc 0.906 |Val: Loss 0.6164, Acc 0.941, BalAcc 0.900, F1 0.960, AUC 0.967
Epoch 12: Train: Loss 0.3607, Acc 0.828 |Val: Loss 0.7329, Acc 1.000, BalAcc 1.000, F1 1.000, AUC 1.000
Epoch 13: Train: Loss 0.2435, Acc 0.938 |Val: Loss 0.8531, Acc 0.941, BalAcc 0.958, F1 0.957, AUC 0.967
Epoch 14: Train: Loss 0.2798, Acc 0.891 |Val: Loss 0.7448, Acc 0.941, BalAcc 0.900, F1 0.960, AUC 0.983
Epoch 15: Train: Loss 0.2860, Acc 0.891 |Val: Loss 0.7010, Acc 0.941, BalAcc 0.900, F1 0.960, AUC 0.967
Early stopping after 15 epochs (no improvement for 7 epochs)
Best model saved: ./output/models/best_model_fold_0.pth (BalAcc: 1.000)

===== Fold 1 =====
Train patients: 65, Test patients: 17
Class weights: Benign=1.71, Malignant=0.71
Epoch 01: Train: Loss 0.7972, Acc 0.672 |Val: Loss 0.6645, Acc 0.824, BalAcc 0.700, F1 0.889, AUC 0.500
New best balanced accuracy: 0.700
Epoch 02: Train: Loss 0.6947, Acc 0.734 |Val: Loss 0.6509, Acc 0.824, BalAcc 0.758, F1 0.880, AUC 0.817
New best balanced accuracy: 0.758
Epoch 03: Train: Loss 0.4787, Acc 0.828 |Val: Loss 0.6477, Acc 0.824, BalAcc 0.758, F1 0.880, AUC 0.733
Epoch 04: Train: Loss 0.3653, Acc 0.875 |Val: Loss 0.6465, Acc 0.882, BalAcc 0.858, F1 0.917, AUC 0.850
New best balanced accuracy: 0.858
Epoch 05: Train: Loss 0.4634, Acc 0.781 |Val: Loss 0.6942, Acc 0.824, BalAcc 0.817, F1 0.870, AUC 0.767
Epoch 06: Train: Loss 0.3184, Acc 0.875 |Val: Loss 0.6486, Acc 0.824, BalAcc 0.700, F1 0.889, AUC 0.833
Epoch 07: Train: Loss 0.4180, Acc 0.781 |Val: Loss 0.5872, Acc 0.824, BalAcc 0.758, F1 0.880, AUC 0.850
Epoch 08: Train: Loss 0.3987, Acc 0.844 |Val: Loss 0.5967, Acc 0.882, BalAcc 0.858, F1 0.917, AUC 0.883
Epoch 09: Train: Loss 0.2853, Acc 0.906 |Val: Loss 0.6074, Acc 0.824, BalAcc 0.758, F1 0.880, AUC 0.783
Epoch 10: Train: Loss 0.3264, Acc 0.891 |Val: Loss 0.6063, Acc 0.824, BalAcc 0.758, F1 0.880, AUC 0.867
Epoch 11: Train: Loss 0.2406, Acc 0.953 |Val: Loss 0.4661, Acc 0.824, BalAcc 0.700, F1 0.889, AUC 0.833
Early stopping after 11 epochs (no improvement for 7 epochs)
Best model saved: ./output/models/best_model_fold_1.pth (BalAcc: 0.858)

===== Fold 2 =====
Train patients: 66, Test patients: 16
Class weights: Benign=1.74, Malignant=0.70
Epoch 01: Train: Loss 0.7132, Acc 0.703 |Val: Loss 0.6943, Acc 0.688, BalAcc 0.500, F1 0.815, AUC 0.345
New best balanced accuracy: 0.500
Epoch 02: Train: Loss 0.4898, Acc 0.844 |Val: Loss 0.6947, Acc 0.812, BalAcc 0.700, F1 0.880, AUC 0.709
New best balanced accuracy: 0.700
Epoch 03: Train: Loss 0.6158, Acc 0.719 |Val: Loss 0.6864, Acc 0.938, BalAcc 0.900, F1 0.957, AUC 0.891
New best balanced accuracy: 0.900
Epoch 04: Train: Loss 0.4444, Acc 0.844 |Val: Loss 0.6635, Acc 0.938, BalAcc 0.900, F1 0.957, AUC 0.982
Epoch 05: Train: Loss 0.4227, Acc 0.797 |Val: Loss 0.6492, Acc 0.875, BalAcc 0.800, F1 0.917, AUC 0.873
Epoch 06: Train: Loss 0.4469, Acc 0.797 |Val: Loss 0.5911, Acc 0.875, BalAcc 0.800, F1 0.917, AUC 0.891
Epoch 07: Train: Loss 0.2995, Acc 0.891 |Val: Loss 0.5442, Acc 0.938, BalAcc 0.900, F1 0.957, AUC 0.855
Epoch 08: Train: Loss 0.3136, Acc 0.859 |Val: Loss 0.4963, Acc 0.938, BalAcc 0.900, F1 0.957, AUC 0.836
Epoch 09: Train: Loss 0.2817, Acc 0.875 |Val: Loss 0.3985, Acc 0.938, BalAcc 0.900, F1 0.957, AUC 0.927
Epoch 10: Train: Loss 0.2580, Acc 0.891 |Val: Loss 0.3978, Acc 0.938, BalAcc 0.900, F1 0.957, AUC 0.927
Early stopping after 10 epochs (no improvement for 7 epochs)
Best model saved: ./output/models/best_model_fold_2.pth (BalAcc: 0.900)

===== Fold 3 =====
Train patients: 66, Test patients: 16
Class weights: Benign=1.74, Malignant=0.70
Epoch 01: Train: Loss 0.7579, Acc 0.469 |Val: Loss 0.6959, Acc 0.750, BalAcc 0.600, F1 0.846, AUC 0.527
New best balanced accuracy: 0.600
Epoch 02: Train: Loss 0.6166, Acc 0.547 |Val: Loss 0.6954, Acc 0.812, BalAcc 0.700, F1 0.880, AUC 0.800
New best balanced accuracy: 0.700
Epoch 03: Train: Loss 0.5677, Acc 0.641 |Val: Loss 0.6986, Acc 0.812, BalAcc 0.809, F1 0.857, AUC 0.691
New best balanced accuracy: 0.809
Epoch 04: Train: Loss 0.5644, Acc 0.688 |Val: Loss 0.6903, Acc 0.812, BalAcc 0.755, F1 0.870, AUC 0.782
Epoch 05: Train: Loss 0.3675, Acc 0.797 |Val: Loss 0.6519, Acc 0.875, BalAcc 0.800, F1 0.917, AUC 0.836
Epoch 06: Train: Loss 0.4037, Acc 0.797 |Val: Loss 0.6520, Acc 0.812, BalAcc 0.700, F1 0.880, AUC 0.691
Epoch 07: Train: Loss 0.4062, Acc 0.812 |Val: Loss 0.5931, Acc 0.875, BalAcc 0.800, F1 0.917, AUC 0.818
Epoch 08: Train: Loss 0.3533, Acc 0.875 |Val: Loss 0.5952, Acc 0.812, BalAcc 0.755, F1 0.870, AUC 0.836
Epoch 09: Train: Loss 0.2819, Acc 0.875 |Val: Loss 0.6236, Acc 0.812, BalAcc 0.700, F1 0.880, AUC 0.782
Epoch 10: Train: Loss 0.2724, Acc 0.891 |Val: Loss 0.6118, Acc 0.812, BalAcc 0.700, F1 0.880, AUC 0.764
Early stopping after 10 epochs (no improvement for 7 epochs)
Best model saved: ./output/models/best_model_fold_3.pth (BalAcc: 0.809)

===== Fold 4 =====
Train patients: 66, Test patients: 16
Class weights: Benign=1.65, Malignant=0.72
Epoch 01: Train: Loss 0.7076, Acc 0.469 |Val: Loss 0.6934, Acc 0.750, BalAcc 0.500, F1 0.857, AUC 0.583
New best balanced accuracy: 0.500
Epoch 02: Train: Loss 0.6355, Acc 0.578 |Val: Loss 0.6930, Acc 0.812, BalAcc 0.625, F1 0.889, AUC 0.812
New best balanced accuracy: 0.625
Epoch 03: Train: Loss 0.5755, Acc 0.719 |Val: Loss 0.6890, Acc 0.938, BalAcc 0.875, F1 0.960, AUC 0.875
New best balanced accuracy: 0.875
Epoch 04: Train: Loss 0.4823, Acc 0.703 |Val: Loss 0.6830, Acc 0.938, BalAcc 0.875, F1 0.960, AUC 0.750
Epoch 05: Train: Loss 0.4816, Acc 0.703 |Val: Loss 0.6348, Acc 1.000, BalAcc 1.000, F1 1.000, AUC 1.000
New best balanced accuracy: 1.000
Epoch 06: Train: Loss 0.3747, Acc 0.812 |Val: Loss 0.6145, Acc 0.938, BalAcc 0.875, F1 0.960, AUC 0.958
Epoch 07: Train: Loss 0.4604, Acc 0.750 |Val: Loss 0.5968, Acc 0.938, BalAcc 0.875, F1 0.960, AUC 0.938
Epoch 08: Train: Loss 0.4144, Acc 0.828 |Val: Loss 0.5599, Acc 0.938, BalAcc 0.875, F1 0.960, AUC 0.979
Epoch 09: Train: Loss 0.3765, Acc 0.797 |Val: Loss 0.5520, Acc 0.938, BalAcc 0.875, F1 0.960, AUC 0.979
Epoch 10: Train: Loss 0.4169, Acc 0.781 |Val: Loss 0.4421, Acc 1.000, BalAcc 1.000, F1 1.000, AUC 1.000
Epoch 11: Train: Loss 0.2657, Acc 0.875 |Val: Loss 0.4581, Acc 0.938, BalAcc 0.875, F1 0.960, AUC 0.958
Epoch 12: Train: Loss 0.3389, Acc 0.906 |Val: Loss 0.3887, Acc 1.000, BalAcc 1.000, F1 1.000, AUC 1.000
Early stopping after 12 epochs (no improvement for 7 epochs)
Best model saved: ./output/models/best_model_fold_4.pth (BalAcc: 1.000)

=== Cross-Validation Results ===
Acc: 0.926 ± 0.026
BalAcc: 0.898 ± 0.059
F1: 0.949 ± 0.016
AUC: 0.931 ± 0.026