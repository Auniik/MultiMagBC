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
Training samples per epoch: 2754 (avg utilization: 64.8%)
Class weights: Benign=1.71, Malignant=0.71
model.safetensors: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 31.5M/31.5M [00:00<00:00, 229MB/s]
Inner training samples: 1896, batch size: 16
Inner split: Train 45, Val 20 patients
Epoch 01: Train: Loss 0.1066, Acc 0.558 | Val: Loss 0.1050, Acc 0.725, BalAcc 0.637, F1 0.814, AUC 0.658, Thresh 0.392
New best validation balanced accuracy: 0.637, threshold: 0.392
Epoch 02: Train: Loss 0.0728, Acc 0.686 | Val: Loss 0.0694, Acc 0.675, BalAcc 0.601, F1 0.772, AUC 0.619, Thresh 0.506
Epoch 03: Train: Loss 0.0626, Acc 0.735 | Val: Loss 0.0662, Acc 0.800, BalAcc 0.738, F1 0.862, AUC 0.815, Thresh 0.519
New best validation balanced accuracy: 0.738, threshold: 0.519
Epoch 04: Train: Loss 0.0574, Acc 0.770 | Val: Loss 0.0851, Acc 0.850, BalAcc 0.821, F1 0.893, AUC 0.756, Thresh 0.585
New best validation balanced accuracy: 0.821, threshold: 0.585
Epoch 05: Train: Loss 0.0536, Acc 0.782 | Val: Loss 0.0748, Acc 0.775, BalAcc 0.696, F1 0.847, AUC 0.750, Thresh 0.486
Epoch 06: Train: Loss 0.0401, Acc 0.806 | Val: Loss 0.0557, Acc 0.850, BalAcc 0.869, F1 0.885, AUC 0.818, Thresh 0.624
New best validation balanced accuracy: 0.869, threshold: 0.624
Epoch 07: Train: Loss 0.0340, Acc 0.851 | Val: Loss 0.0437, Acc 0.775, BalAcc 0.696, F1 0.847, AUC 0.824, Thresh 0.490
Epoch 08: Train: Loss 0.0353, Acc 0.854 | Val: Loss 0.0387, Acc 0.850, BalAcc 0.845, F1 0.889, AUC 0.860, Thresh 0.533
Epoch 09: Train: Loss 0.0332, Acc 0.889 | Val: Loss 0.0412, Acc 0.850, BalAcc 0.845, F1 0.889, AUC 0.875, Thresh 0.558
Epoch 10: Train: Loss 0.0288, Acc 0.895 | Val: Loss 0.0448, Acc 0.850, BalAcc 0.774, F1 0.900, AUC 0.812, Thresh 0.438
Epoch 11: Train: Loss 0.0291, Acc 0.917 | Val: Loss 0.0496, Acc 0.825, BalAcc 0.732, F1 0.885, AUC 0.845, Thresh 0.416
Epoch 12: Train: Loss 0.0256, Acc 0.941 | Val: Loss 0.0336, Acc 0.825, BalAcc 0.732, F1 0.885, AUC 0.896, Thresh 0.344
Epoch 13: Train: Loss 0.0280, Acc 0.938 | Val: Loss 0.0493, Acc 0.825, BalAcc 0.756, F1 0.881, AUC 0.812, Thresh 0.443
Early stopping after 13 epochs (no improvement for 7 epochs)
Best model saved: ./output/models/best_model_fold_0.pth (Val BalAcc: 0.869)
Test Results: Acc 0.941, BalAcc 0.958, F1 0.957, AUC 0.967 (threshold: 0.624)

===== Fold 1 =====
Train patients: 65, Test patients: 17
Training samples per epoch: 2721 (avg utilization: 65.3%)
Class weights: Benign=1.71, Malignant=0.71
Inner training samples: 1896, batch size: 16
Inner split: Train 45, Val 20 patients
Epoch 01: Train: Loss 0.0913, Acc 0.619 | Val: Loss 0.0287, Acc 0.875, BalAcc 0.839, F1 0.912, AUC 0.926, Thresh 0.413
New best validation balanced accuracy: 0.839, threshold: 0.413
Epoch 02: Train: Loss 0.0666, Acc 0.731 | Val: Loss 0.0393, Acc 0.800, BalAcc 0.690, F1 0.871, AUC 0.827, Thresh 0.374
Epoch 03: Train: Loss 0.0486, Acc 0.858 | Val: Loss 0.0876, Acc 0.775, BalAcc 0.720, F1 0.842, AUC 0.786, Thresh 0.359
Epoch 04: Train: Loss 0.0449, Acc 0.862 | Val: Loss 0.0698, Acc 0.875, BalAcc 0.887, F1 0.906, AUC 0.857, Thresh 0.556
New best validation balanced accuracy: 0.887, threshold: 0.556
Epoch 05: Train: Loss 0.0483, Acc 0.851 | Val: Loss 0.0362, Acc 0.850, BalAcc 0.821, F1 0.893, AUC 0.869, Thresh 0.524
Epoch 06: Train: Loss 0.0470, Acc 0.850 | Val: Loss 0.0417, Acc 0.850, BalAcc 0.798, F1 0.897, AUC 0.842, Thresh 0.496
Epoch 07: Train: Loss 0.0349, Acc 0.892 | Val: Loss 0.0708, Acc 0.825, BalAcc 0.827, F1 0.868, AUC 0.798, Thresh 0.605
Epoch 08: Train: Loss 0.0316, Acc 0.899 | Val: Loss 0.1060, Acc 0.850, BalAcc 0.821, F1 0.893, AUC 0.821, Thresh 0.530
Epoch 09: Train: Loss 0.0283, Acc 0.914 | Val: Loss 0.0597, Acc 0.850, BalAcc 0.798, F1 0.897, AUC 0.824, Thresh 0.445
Epoch 10: Train: Loss 0.0298, Acc 0.935 | Val: Loss 0.0597, Acc 0.800, BalAcc 0.762, F1 0.857, AUC 0.848, Thresh 0.468
Epoch 11: Train: Loss 0.0236, Acc 0.947 | Val: Loss 0.0731, Acc 0.900, BalAcc 0.881, F1 0.929, AUC 0.857, Thresh 0.495
Early stopping after 11 epochs (no improvement for 7 epochs)
Best model saved: ./output/models/best_model_fold_1.pth (Val BalAcc: 0.887)
Test Results: Acc 0.882, BalAcc 0.858, F1 0.917, AUC 0.933 (threshold: 0.556)

===== Fold 2 =====
Train patients: 66, Test patients: 16
Training samples per epoch: 2763 (avg utilization: 64.7%)
Class weights: Benign=1.74, Malignant=0.70
Inner training samples: 1917, batch size: 16
Inner split: Train 46, Val 20 patients
Epoch 01: Train: Loss 0.0941, Acc 0.597 | Val: Loss 0.0414, Acc 0.825, BalAcc 0.804, F1 0.873, AUC 0.848, Thresh 0.446
New best validation balanced accuracy: 0.804, threshold: 0.446
Epoch 02: Train: Loss 0.0645, Acc 0.742 | Val: Loss 0.0230, Acc 0.925, BalAcc 0.899, F1 0.947, AUC 0.964, Thresh 0.617
New best validation balanced accuracy: 0.899, threshold: 0.617
Epoch 03: Train: Loss 0.0464, Acc 0.814 | Val: Loss 0.0238, Acc 0.900, BalAcc 0.857, F1 0.931, AUC 0.917, Thresh 0.524
Epoch 04: Train: Loss 0.0374, Acc 0.845 | Val: Loss 0.0144, Acc 0.950, BalAcc 0.917, F1 0.966, AUC 0.979, Thresh 0.533
New best validation balanced accuracy: 0.917, threshold: 0.533
Epoch 05: Train: Loss 0.0376, Acc 0.878 | Val: Loss 0.0586, Acc 0.775, BalAcc 0.673, F1 0.852, AUC 0.741, Thresh 0.462
Epoch 06: Train: Loss 0.0306, Acc 0.902 | Val: Loss 0.0134, Acc 1.000, BalAcc 1.000, F1 1.000, AUC 1.000, Thresh 0.573 [PERFECT VAL - POSSIBLE OVERFITTING]
New best validation balanced accuracy: 1.000, threshold: 0.573
Epoch 07: Train: Loss 0.0323, Acc 0.908 | Val: Loss 0.0168, Acc 0.925, BalAcc 0.899, F1 0.947, AUC 0.938, Thresh 0.560
Epoch 08: Train: Loss 0.0264, Acc 0.924 | Val: Loss 0.0143, Acc 0.975, BalAcc 0.958, F1 0.982, AUC 0.985, Thresh 0.553
Epoch 09: Train: Loss 0.0285, Acc 0.926 | Val: Loss 0.0519, Acc 0.925, BalAcc 0.899, F1 0.947, AUC 0.881, Thresh 0.545
Epoch 10: Train: Loss 0.0243, Acc 0.932 | Val: Loss 0.0137, Acc 0.975, BalAcc 0.958, F1 0.982, AUC 0.988, Thresh 0.483
Epoch 11: Train: Loss 0.0244, Acc 0.951 | Val: Loss 0.0109, Acc 1.000, BalAcc 1.000, F1 1.000, AUC 1.000, Thresh 0.533 [PERFECT VAL - POSSIBLE OVERFITTING]
Epoch 12: Train: Loss 0.0213, Acc 0.949 | Val: Loss 0.0097, Acc 1.000, BalAcc 1.000, F1 1.000, AUC 1.000, Thresh 0.574 [PERFECT VAL - POSSIBLE OVERFITTING]
Epoch 13: Train: Loss 0.0215, Acc 0.964 | Val: Loss 0.0113, Acc 1.000, BalAcc 1.000, F1 1.000, AUC 1.000, Thresh 0.607 [PERFECT VAL - POSSIBLE OVERFITTING]
Early stopping after 13 epochs (no improvement for 7 epochs)
Best model saved: ./output/models/best_model_fold_2.pth (Val BalAcc: 1.000)
Test Results: Acc 0.812, BalAcc 0.809, F1 0.857, AUC 0.855 (threshold: 0.573)

===== Fold 3 =====
Train patients: 66, Test patients: 16
Training samples per epoch: 2778 (avg utilization: 65.8%)
Class weights: Benign=1.74, Malignant=0.70
Inner training samples: 1950, batch size: 16
Inner split: Train 46, Val 20 patients
Epoch 01: Train: Loss 0.0996, Acc 0.606 | Val: Loss 0.0774, Acc 0.675, BalAcc 0.577, F1 0.780, AUC 0.634, Thresh 0.435
New best validation balanced accuracy: 0.577, threshold: 0.435
Epoch 02: Train: Loss 0.0679, Acc 0.761 | Val: Loss 0.0324, Acc 0.800, BalAcc 0.690, F1 0.871, AUC 0.863, Thresh 0.528
New best validation balanced accuracy: 0.690, threshold: 0.528
Epoch 03: Train: Loss 0.0507, Acc 0.798 | Val: Loss 0.0587, Acc 0.825, BalAcc 0.708, F1 0.889, AUC 0.750, Thresh 0.423
New best validation balanced accuracy: 0.708, threshold: 0.423
Epoch 04: Train: Loss 0.0453, Acc 0.824 | Val: Loss 0.0469, Acc 0.900, BalAcc 0.833, F1 0.933, AUC 0.824, Thresh 0.435
New best validation balanced accuracy: 0.833, threshold: 0.435
Epoch 05: Train: Loss 0.0448, Acc 0.858 | Val: Loss 0.0456, Acc 0.850, BalAcc 0.821, F1 0.893, AUC 0.824, Thresh 0.552
Epoch 06: Train: Loss 0.0339, Acc 0.890 | Val: Loss 0.0370, Acc 0.875, BalAcc 0.815, F1 0.915, AUC 0.869, Thresh 0.546
Epoch 07: Train: Loss 0.0303, Acc 0.915 | Val: Loss 0.0298, Acc 0.875, BalAcc 0.815, F1 0.915, AUC 0.887, Thresh 0.520
Epoch 08: Train: Loss 0.0303, Acc 0.917 | Val: Loss 0.0357, Acc 0.950, BalAcc 0.940, F1 0.964, AUC 0.899, Thresh 0.614
New best validation balanced accuracy: 0.940, threshold: 0.614
Epoch 09: Train: Loss 0.0258, Acc 0.925 | Val: Loss 0.0242, Acc 0.925, BalAcc 0.899, F1 0.947, AUC 0.905, Thresh 0.586
Epoch 10: Train: Loss 0.0298, Acc 0.934 | Val: Loss 0.0232, Acc 0.925, BalAcc 0.899, F1 0.947, AUC 0.911, Thresh 0.565
Epoch 11: Train: Loss 0.0218, Acc 0.949 | Val: Loss 0.0635, Acc 0.850, BalAcc 0.798, F1 0.897, AUC 0.801, Thresh 0.466
Epoch 12: Train: Loss 0.0236, Acc 0.943 | Val: Loss 0.0301, Acc 0.875, BalAcc 0.815, F1 0.915, AUC 0.905, Thresh 0.557
Epoch 13: Train: Loss 0.0220, Acc 0.948 | Val: Loss 0.0533, Acc 0.875, BalAcc 0.815, F1 0.915, AUC 0.807, Thresh 0.479
Epoch 14: Train: Loss 0.0249, Acc 0.956 | Val: Loss 0.0275, Acc 0.900, BalAcc 0.857, F1 0.931, AUC 0.893, Thresh 0.494
Epoch 15: Train: Loss 0.0247, Acc 0.953 | Val: Loss 0.0448, Acc 0.925, BalAcc 0.875, F1 0.949, AUC 0.839, Thresh 0.473
Early stopping after 15 epochs (no improvement for 7 epochs)
Best model saved: ./output/models/best_model_fold_3.pth (Val BalAcc: 0.940)
Test Results: Acc 0.938, BalAcc 0.900, F1 0.957, AUC 1.000 (threshold: 0.614)

===== Fold 4 =====
Train patients: 66, Test patients: 16
Training samples per epoch: 2820 (avg utilization: 64.0%)
Class weights: Benign=1.65, Malignant=0.72
Inner training samples: 1989, batch size: 16
Inner split: Train 46, Val 20 patients
Epoch 01: Train: Loss 0.0855, Acc 0.617 | Val: Loss 0.1187, Acc 0.700, BalAcc 0.595, F1 0.800, AUC 0.637, Thresh 0.422
New best validation balanced accuracy: 0.595, threshold: 0.422
Epoch 02: Train: Loss 0.0676, Acc 0.751 | Val: Loss 0.0823, Acc 0.725, BalAcc 0.589, F1 0.825, AUC 0.679, Thresh 0.328
Epoch 03: Train: Loss 0.0638, Acc 0.746 | Val: Loss 0.1017, Acc 0.700, BalAcc 0.571, F1 0.806, AUC 0.613, Thresh 0.333
Epoch 04: Train: Loss 0.0496, Acc 0.780 | Val: Loss 0.0634, Acc 0.800, BalAcc 0.738, F1 0.862, AUC 0.830, Thresh 0.516
New best validation balanced accuracy: 0.738, threshold: 0.516
Epoch 05: Train: Loss 0.0436, Acc 0.835 | Val: Loss 0.1061, Acc 0.725, BalAcc 0.589, F1 0.825, AUC 0.634, Thresh 0.317
Epoch 06: Train: Loss 0.0346, Acc 0.876 | Val: Loss 0.0466, Acc 0.800, BalAcc 0.667, F1 0.875, AUC 0.777, Thresh 0.372
Epoch 07: Train: Loss 0.0377, Acc 0.866 | Val: Loss 0.0461, Acc 0.825, BalAcc 0.756, F1 0.881, AUC 0.798, Thresh 0.496
New best validation balanced accuracy: 0.756, threshold: 0.496
Epoch 08: Train: Loss 0.0324, Acc 0.896 | Val: Loss 0.0693, Acc 0.800, BalAcc 0.738, F1 0.862, AUC 0.798, Thresh 0.432
Epoch 09: Train: Loss 0.0270, Acc 0.904 | Val: Loss 0.0393, Acc 0.750, BalAcc 0.631, F1 0.839, AUC 0.827, Thresh 0.471
Epoch 10: Train: Loss 0.0247, Acc 0.915 | Val: Loss 0.0616, Acc 0.825, BalAcc 0.756, F1 0.881, AUC 0.777, Thresh 0.379
Epoch 11: Train: Loss 0.0274, Acc 0.909 | Val: Loss 0.0489, Acc 0.800, BalAcc 0.738, F1 0.862, AUC 0.795, Thresh 0.453
Epoch 12: Train: Loss 0.0232, Acc 0.936 | Val: Loss 0.0586, Acc 0.800, BalAcc 0.714, F1 0.867, AUC 0.798, Thresh 0.427
Epoch 13: Train: Loss 0.0223, Acc 0.958 | Val: Loss 0.0700, Acc 0.825, BalAcc 0.756, F1 0.881, AUC 0.717, Thresh 0.433
Epoch 14: Train: Loss 0.0224, Acc 0.954 | Val: Loss 0.0410, Acc 0.875, BalAcc 0.839, F1 0.912, AUC 0.827, Thresh 0.500
New best validation balanced accuracy: 0.839, threshold: 0.500
Epoch 15: Train: Loss 0.0244, Acc 0.954 | Val: Loss 0.0438, Acc 0.800, BalAcc 0.762, F1 0.857, AUC 0.836, Thresh 0.514
Epoch 16: Train: Loss 0.0237, Acc 0.958 | Val: Loss 0.0559, Acc 0.800, BalAcc 0.738, F1 0.862, AUC 0.756, Thresh 0.413
Epoch 17: Train: Loss 0.0250, Acc 0.952 | Val: Loss 0.0569, Acc 0.875, BalAcc 0.815, F1 0.915, AUC 0.807, Thresh 0.456
Epoch 18: Train: Loss 0.0209, Acc 0.960 | Val: Loss 0.0518, Acc 0.825, BalAcc 0.780, F1 0.877, AUC 0.798, Thresh 0.514
Epoch 19: Train: Loss 0.0199, Acc 0.967 | Val: Loss 0.0449, Acc 0.875, BalAcc 0.815, F1 0.915, AUC 0.836, Thresh 0.473
Epoch 20: Train: Loss 0.0204, Acc 0.970 | Val: Loss 0.0357, Acc 0.900, BalAcc 0.905, F1 0.926, AUC 0.929, Thresh 0.578
New best validation balanced accuracy: 0.905, threshold: 0.578
Epoch 21: Train: Loss 0.0188, Acc 0.975 | Val: Loss 0.0382, Acc 0.850, BalAcc 0.845, F1 0.889, AUC 0.845, Thresh 0.542
Epoch 22: Train: Loss 0.0213, Acc 0.977 | Val: Loss 0.0543, Acc 0.800, BalAcc 0.738, F1 0.862, AUC 0.774, Thresh 0.478
Epoch 23: Train: Loss 0.0185, Acc 0.980 | Val: Loss 0.0575, Acc 0.875, BalAcc 0.839, F1 0.912, AUC 0.815, Thresh 0.443
Epoch 24: Train: Loss 0.0173, Acc 0.977 | Val: Loss 0.0391, Acc 0.850, BalAcc 0.845, F1 0.889, AUC 0.860, Thresh 0.567
Epoch 25: Train: Loss 0.0208, Acc 0.983 | Val: Loss 0.0485, Acc 0.850, BalAcc 0.821, F1 0.893, AUC 0.818, Thresh 0.497
Best model saved: ./output/models/best_model_fold_4.pth (Val BalAcc: 0.905)
Test Results: Acc 1.000, BalAcc 1.000, F1 1.000, AUC 1.000 (threshold: 0.578)

=== Cross-Validation Results ===
Acc: 0.915 ± 0.063
BalAcc: 0.905 ± 0.068
F1: 0.937 ± 0.048
AUC: 0.951 ± 0.054