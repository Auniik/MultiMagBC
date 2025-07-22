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
model.safetensors: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 31.5M/31.5M [00:00<00:00, 158MB/s]
Inner training samples: 270, batch size: 24
Inner split: Train 45, Val 20 patients
Epoch 01: Train: Loss 0.1623, Acc 0.564 |Val: Loss 0.2570, Acc 0.700, BalAcc 0.500, F1 0.824, AUC 0.476, Thresh 0.094
New best validation balanced accuracy: 0.500, threshold: 0.094
Epoch 02: Train: Loss 0.1343, Acc 0.591 |Val: Loss 0.3258, Acc 0.700, BalAcc 0.500, F1 0.824, AUC 0.619, Thresh 0.066
Epoch 03: Train: Loss 0.1286, Acc 0.591 |Val: Loss 0.0901, Acc 0.800, BalAcc 0.667, F1 0.875, AUC 0.667, Thresh 0.407
New best validation balanced accuracy: 0.667, threshold: 0.407
Epoch 04: Train: Loss 0.1243, Acc 0.617 |Val: Loss 0.0938, Acc 0.700, BalAcc 0.500, F1 0.824, AUC 0.548, Thresh 0.323
Epoch 05: Train: Loss 0.1374, Acc 0.648 |Val: Loss 0.0506, Acc 0.850, BalAcc 0.798, F1 0.897, AUC 0.821, Thresh 0.626
New best validation balanced accuracy: 0.798, threshold: 0.626
Epoch 06: Train: Loss 0.1189, Acc 0.720 |Val: Loss 0.0840, Acc 0.850, BalAcc 0.750, F1 0.903, AUC 0.774, Thresh 0.358
Epoch 07: Train: Loss 0.1030, Acc 0.678 |Val: Loss 0.0907, Acc 0.800, BalAcc 0.667, F1 0.875, AUC 0.810, Thresh 0.260
Epoch 08: Train: Loss 0.0926, Acc 0.735 |Val: Loss 0.0750, Acc 0.850, BalAcc 0.798, F1 0.897, AUC 0.810, Thresh 0.491
Epoch 09: Train: Loss 0.0986, Acc 0.758 |Val: Loss 0.1248, Acc 0.800, BalAcc 0.714, F1 0.867, AUC 0.726, Thresh 0.408
Epoch 10: Train: Loss 0.0842, Acc 0.795 |Val: Loss 0.0707, Acc 0.850, BalAcc 0.798, F1 0.897, AUC 0.893, Thresh 0.433
Epoch 11: Train: Loss 0.0853, Acc 0.758 |Val: Loss 0.0594, Acc 0.850, BalAcc 0.750, F1 0.903, AUC 0.893, Thresh 0.438
Epoch 12: Train: Loss 0.0869, Acc 0.758 |Val: Loss 0.0597, Acc 0.800, BalAcc 0.667, F1 0.875, AUC 0.810, Thresh 0.441
Early stopping after 12 epochs (no improvement for 7 epochs)
Best model saved: ./output/models/best_model_fold_0.pth (Val BalAcc: 0.798)
Test Results: Acc 0.765, BalAcc 0.833, F1 0.800, AUC 0.933 (threshold: 0.626)

===== Fold 1 =====
Train patients: 65, Test patients: 17
Training samples per epoch: 388 (avg utilization: 29.1%)
Class weights: Benign=1.71, Malignant=0.71
Inner training samples: 268, batch size: 24
Inner split: Train 45, Val 20 patients
Epoch 01: Train: Loss 0.1579, Acc 0.549 |Val: Loss 0.1250, Acc 0.750, BalAcc 0.583, F1 0.848, AUC 0.571, Thresh 0.176
New best validation balanced accuracy: 0.583, threshold: 0.176
Epoch 02: Train: Loss 0.1519, Acc 0.644 |Val: Loss 0.0916, Acc 0.700, BalAcc 0.500, F1 0.824, AUC 0.679, Thresh 0.154
Epoch 03: Train: Loss 0.1398, Acc 0.549 |Val: Loss 0.1131, Acc 0.700, BalAcc 0.500, F1 0.824, AUC 0.714, Thresh 0.107
Epoch 04: Train: Loss 0.1121, Acc 0.621 |Val: Loss 0.1128, Acc 0.800, BalAcc 0.667, F1 0.875, AUC 0.726, Thresh 0.118
New best validation balanced accuracy: 0.667, threshold: 0.118
Epoch 05: Train: Loss 0.1132, Acc 0.674 |Val: Loss 0.0733, Acc 0.850, BalAcc 0.750, F1 0.903, AUC 0.738, Thresh 0.397
New best validation balanced accuracy: 0.750, threshold: 0.397
Epoch 06: Train: Loss 0.0914, Acc 0.701 |Val: Loss 0.1266, Acc 0.700, BalAcc 0.500, F1 0.824, AUC 0.667, Thresh 0.253
Epoch 07: Train: Loss 0.0931, Acc 0.701 |Val: Loss 0.0947, Acc 0.800, BalAcc 0.714, F1 0.867, AUC 0.702, Thresh 0.536
Epoch 08: Train: Loss 0.0943, Acc 0.746 |Val: Loss 0.1379, Acc 0.750, BalAcc 0.583, F1 0.848, AUC 0.536, Thresh 0.260
Epoch 09: Train: Loss 0.0853, Acc 0.761 |Val: Loss 0.0665, Acc 0.850, BalAcc 0.845, F1 0.889, AUC 0.833, Thresh 0.670
New best validation balanced accuracy: 0.845, threshold: 0.670
Epoch 10: Train: Loss 0.0785, Acc 0.780 |Val: Loss 0.0520, Acc 0.850, BalAcc 0.750, F1 0.903, AUC 0.917, Thresh 0.420
Epoch 11: Train: Loss 0.0692, Acc 0.852 |Val: Loss 0.0730, Acc 0.800, BalAcc 0.667, F1 0.875, AUC 0.845, Thresh 0.390
Epoch 12: Train: Loss 0.0738, Acc 0.833 |Val: Loss 0.2290, Acc 0.750, BalAcc 0.583, F1 0.848, AUC 0.774, Thresh 0.138
Epoch 13: Train: Loss 0.0572, Acc 0.814 |Val: Loss 0.1007, Acc 0.800, BalAcc 0.667, F1 0.875, AUC 0.774, Thresh 0.295
Epoch 14: Train: Loss 0.0698, Acc 0.818 |Val: Loss 0.1675, Acc 0.800, BalAcc 0.714, F1 0.867, AUC 0.774, Thresh 0.617
Epoch 15: Train: Loss 0.0712, Acc 0.848 |Val: Loss 0.0803, Acc 0.900, BalAcc 0.929, F1 0.923, AUC 0.929, Thresh 0.663
New best validation balanced accuracy: 0.929, threshold: 0.663
Epoch 16: Train: Loss 0.0734, Acc 0.822 |Val: Loss 0.0726, Acc 0.850, BalAcc 0.750, F1 0.903, AUC 0.750, Thresh 0.442
Epoch 17: Train: Loss 0.0499, Acc 0.860 |Val: Loss 0.2262, Acc 0.750, BalAcc 0.583, F1 0.848, AUC 0.583, Thresh 0.084
Epoch 18: Train: Loss 0.0496, Acc 0.875 |Val: Loss 0.0583, Acc 0.800, BalAcc 0.667, F1 0.875, AUC 0.821, Thresh 0.462
Epoch 19: Train: Loss 0.0606, Acc 0.864 |Val: Loss 0.1911, Acc 0.850, BalAcc 0.845, F1 0.889, AUC 0.786, Thresh 0.769
Epoch 20: Train: Loss 0.0587, Acc 0.924 |Val: Loss 0.0803, Acc 0.850, BalAcc 0.798, F1 0.897, AUC 0.821, Thresh 0.582
Epoch 21: Train: Loss 0.0576, Acc 0.898 |Val: Loss 0.0501, Acc 0.900, BalAcc 0.833, F1 0.933, AUC 0.810, Thresh 0.509
Epoch 22: Train: Loss 0.0708, Acc 0.894 |Val: Loss 0.0892, Acc 0.800, BalAcc 0.667, F1 0.875, AUC 0.798, Thresh 0.226
Early stopping after 22 epochs (no improvement for 7 epochs)
Best model saved: ./output/models/best_model_fold_1.pth (Val BalAcc: 0.929)
Test Results: Acc 0.882, BalAcc 0.858, F1 0.917, AUC 0.833 (threshold: 0.663)

===== Fold 2 =====
Train patients: 66, Test patients: 16
Training samples per epoch: 394 (avg utilization: 28.8%)
Class weights: Benign=1.74, Malignant=0.70
Inner training samples: 274, batch size: 24
Inner split: Train 46, Val 20 patients
Epoch 01: Train: Loss 0.2282, Acc 0.455 |Val: Loss 0.1387, Acc 0.700, BalAcc 0.500, F1 0.824, AUC 0.607, Thresh 0.109
New best validation balanced accuracy: 0.500, threshold: 0.109
Epoch 02: Train: Loss 0.2392, Acc 0.572 |Val: Loss 0.1265, Acc 0.750, BalAcc 0.583, F1 0.848, AUC 0.536, Thresh 0.219
New best validation balanced accuracy: 0.583, threshold: 0.219
Epoch 03: Train: Loss 0.1773, Acc 0.489 |Val: Loss 0.2635, Acc 0.750, BalAcc 0.631, F1 0.839, AUC 0.607, Thresh 0.267
New best validation balanced accuracy: 0.631, threshold: 0.267
Epoch 04: Train: Loss 0.1822, Acc 0.553 |Val: Loss 0.1151, Acc 0.800, BalAcc 0.714, F1 0.867, AUC 0.679, Thresh 0.299
New best validation balanced accuracy: 0.714, threshold: 0.299
Epoch 05: Train: Loss 0.1783, Acc 0.587 |Val: Loss 0.2063, Acc 0.750, BalAcc 0.583, F1 0.848, AUC 0.548, Thresh 0.100
Epoch 06: Train: Loss 0.1458, Acc 0.549 |Val: Loss 0.1063, Acc 0.750, BalAcc 0.631, F1 0.839, AUC 0.714, Thresh 0.386
Epoch 07: Train: Loss 0.1323, Acc 0.614 |Val: Loss 0.1798, Acc 0.700, BalAcc 0.500, F1 0.824, AUC 0.393, Thresh 0.177
Epoch 08: Train: Loss 0.1489, Acc 0.625 |Val: Loss 0.1031, Acc 0.750, BalAcc 0.631, F1 0.839, AUC 0.798, Thresh 0.321
Epoch 09: Train: Loss 0.1221, Acc 0.705 |Val: Loss 0.1063, Acc 0.750, BalAcc 0.583, F1 0.848, AUC 0.643, Thresh 0.317
Epoch 10: Train: Loss 0.1293, Acc 0.686 |Val: Loss 0.2435, Acc 0.700, BalAcc 0.500, F1 0.824, AUC 0.250, Thresh 0.063
Epoch 11: Train: Loss 0.1021, Acc 0.712 |Val: Loss 0.0700, Acc 0.850, BalAcc 0.750, F1 0.903, AUC 0.833, Thresh 0.314
New best validation balanced accuracy: 0.750, threshold: 0.314
Epoch 12: Train: Loss 0.1144, Acc 0.723 |Val: Loss 0.0633, Acc 0.750, BalAcc 0.583, F1 0.848, AUC 0.786, Thresh 0.315
Epoch 13: Train: Loss 0.1029, Acc 0.739 |Val: Loss 0.0562, Acc 0.900, BalAcc 0.833, F1 0.933, AUC 0.857, Thresh 0.424
New best validation balanced accuracy: 0.833, threshold: 0.424
Epoch 14: Train: Loss 0.1041, Acc 0.739 |Val: Loss 0.0706, Acc 0.850, BalAcc 0.750, F1 0.903, AUC 0.738, Thresh 0.427
Epoch 15: Train: Loss 0.0928, Acc 0.750 |Val: Loss 0.0603, Acc 0.800, BalAcc 0.667, F1 0.875, AUC 0.857, Thresh 0.249
Epoch 16: Train: Loss 0.0946, Acc 0.780 |Val: Loss 0.0652, Acc 0.900, BalAcc 0.929, F1 0.923, AUC 0.881, Thresh 0.504
New best validation balanced accuracy: 0.929, threshold: 0.504
Epoch 17: Train: Loss 0.0909, Acc 0.742 |Val: Loss 0.0923, Acc 0.750, BalAcc 0.583, F1 0.848, AUC 0.726, Thresh 0.214
Epoch 18: Train: Loss 0.0836, Acc 0.754 |Val: Loss 0.0707, Acc 0.850, BalAcc 0.750, F1 0.903, AUC 0.750, Thresh 0.396
Epoch 19: Train: Loss 0.0780, Acc 0.761 |Val: Loss 0.0529, Acc 0.900, BalAcc 0.881, F1 0.929, AUC 0.869, Thresh 0.575
Epoch 20: Train: Loss 0.0786, Acc 0.830 |Val: Loss 0.0455, Acc 0.950, BalAcc 0.917, F1 0.966, AUC 0.952, Thresh 0.376
Epoch 21: Train: Loss 0.0827, Acc 0.830 |Val: Loss 0.0490, Acc 0.900, BalAcc 0.833, F1 0.933, AUC 0.929, Thresh 0.347
Epoch 22: Train: Loss 0.0769, Acc 0.841 |Val: Loss 0.0490, Acc 0.900, BalAcc 0.833, F1 0.933, AUC 0.857, Thresh 0.372
Epoch 23: Train: Loss 0.0770, Acc 0.837 |Val: Loss 0.1032, Acc 0.800, BalAcc 0.714, F1 0.867, AUC 0.679, Thresh 0.405
Early stopping after 23 epochs (no improvement for 7 epochs)
Best model saved: ./output/models/best_model_fold_2.pth (Val BalAcc: 0.929)
Test Results: Acc 0.875, BalAcc 0.800, F1 0.917, AUC 0.782 (threshold: 0.504)

===== Fold 3 =====
Train patients: 66, Test patients: 16
Training samples per epoch: 394 (avg utilization: 29.0%)
Class weights: Benign=1.74, Malignant=0.70
Inner training samples: 274, batch size: 24
Inner split: Train 46, Val 20 patients
Epoch 01: Train: Loss 0.2521, Acc 0.458 |Val: Loss 0.1887, Acc 0.700, BalAcc 0.500, F1 0.824, AUC 0.690, Thresh 0.050
New best validation balanced accuracy: 0.500, threshold: 0.050
Epoch 02: Train: Loss 0.2038, Acc 0.508 |Val: Loss 0.1370, Acc 0.750, BalAcc 0.583, F1 0.848, AUC 0.726, Thresh 0.147
New best validation balanced accuracy: 0.583, threshold: 0.147
Epoch 03: Train: Loss 0.1940, Acc 0.527 |Val: Loss 0.1167, Acc 0.700, BalAcc 0.500, F1 0.824, AUC 0.524, Thresh 0.171
Epoch 04: Train: Loss 0.1870, Acc 0.598 |Val: Loss 0.1468, Acc 0.750, BalAcc 0.631, F1 0.839, AUC 0.631, Thresh 0.335
New best validation balanced accuracy: 0.631, threshold: 0.335
Epoch 05: Train: Loss 0.1723, Acc 0.580 |Val: Loss 0.2302, Acc 0.700, BalAcc 0.500, F1 0.824, AUC 0.357, Thresh 0.097
Epoch 06: Train: Loss 0.1482, Acc 0.583 |Val: Loss 0.1385, Acc 0.800, BalAcc 0.667, F1 0.875, AUC 0.821, Thresh 0.145
New best validation balanced accuracy: 0.667, threshold: 0.145
Epoch 07: Train: Loss 0.1215, Acc 0.644 |Val: Loss 0.1397, Acc 0.700, BalAcc 0.500, F1 0.824, AUC 0.762, Thresh 0.118
Epoch 08: Train: Loss 0.1212, Acc 0.746 |Val: Loss 0.1153, Acc 0.850, BalAcc 0.798, F1 0.897, AUC 0.833, Thresh 0.364
New best validation balanced accuracy: 0.798, threshold: 0.364
Epoch 09: Train: Loss 0.1210, Acc 0.761 |Val: Loss 0.1521, Acc 0.800, BalAcc 0.667, F1 0.875, AUC 0.738, Thresh 0.209
Epoch 10: Train: Loss 0.1005, Acc 0.769 |Val: Loss 0.0939, Acc 0.900, BalAcc 0.833, F1 0.933, AUC 0.798, Thresh 0.341
New best validation balanced accuracy: 0.833, threshold: 0.341
Epoch 11: Train: Loss 0.1171, Acc 0.742 |Val: Loss 0.0884, Acc 0.750, BalAcc 0.583, F1 0.848, AUC 0.762, Thresh 0.254
Epoch 12: Train: Loss 0.0982, Acc 0.818 |Val: Loss 0.0935, Acc 0.750, BalAcc 0.583, F1 0.848, AUC 0.786, Thresh 0.207
Epoch 13: Train: Loss 0.0807, Acc 0.833 |Val: Loss 0.1299, Acc 0.800, BalAcc 0.810, F1 0.846, AUC 0.750, Thresh 0.634
Epoch 14: Train: Loss 0.0767, Acc 0.848 |Val: Loss 0.0656, Acc 0.850, BalAcc 0.750, F1 0.903, AUC 0.750, Thresh 0.213
Epoch 15: Train: Loss 0.0767, Acc 0.837 |Val: Loss 0.0526, Acc 0.850, BalAcc 0.750, F1 0.903, AUC 0.821, Thresh 0.430
Epoch 16: Train: Loss 0.0820, Acc 0.875 |Val: Loss 0.0444, Acc 0.900, BalAcc 0.833, F1 0.933, AUC 0.821, Thresh 0.593
Epoch 17: Train: Loss 0.0570, Acc 0.852 |Val: Loss 0.0762, Acc 0.750, BalAcc 0.583, F1 0.848, AUC 0.714, Thresh 0.296
Early stopping after 17 epochs (no improvement for 7 epochs)
Best model saved: ./output/models/best_model_fold_3.pth (Val BalAcc: 0.833)
Test Results: Acc 0.750, BalAcc 0.600, F1 0.846, AUC 0.745 (threshold: 0.341)

===== Fold 4 =====
Train patients: 66, Test patients: 16
Training samples per epoch: 395 (avg utilization: 27.7%)
Class weights: Benign=1.65, Malignant=0.72
Inner training samples: 276, batch size: 24
Inner split: Train 46, Val 20 patients
Epoch 01: Train: Loss 0.3457, Acc 0.375 |Val: Loss 0.4572, Acc 0.700, BalAcc 0.500, F1 0.824, AUC 0.381, Thresh 0.059
New best validation balanced accuracy: 0.500, threshold: 0.059
Epoch 02: Train: Loss 0.3035, Acc 0.470 |Val: Loss 0.3675, Acc 0.750, BalAcc 0.583, F1 0.848, AUC 0.512, Thresh 0.130
New best validation balanced accuracy: 0.583, threshold: 0.130
Epoch 03: Train: Loss 0.2730, Acc 0.447 |Val: Loss 0.3730, Acc 0.700, BalAcc 0.500, F1 0.824, AUC 0.381, Thresh 0.082
Epoch 04: Train: Loss 0.2147, Acc 0.496 |Val: Loss 0.3065, Acc 0.700, BalAcc 0.500, F1 0.824, AUC 0.679, Thresh 0.029
Epoch 05: Train: Loss 0.2231, Acc 0.519 |Val: Loss 0.1946, Acc 0.800, BalAcc 0.667, F1 0.875, AUC 0.464, Thresh 0.284
New best validation balanced accuracy: 0.667, threshold: 0.284
Epoch 06: Train: Loss 0.2137, Acc 0.587 |Val: Loss 0.1052, Acc 0.900, BalAcc 0.833, F1 0.933, AUC 0.810, Thresh 0.286
New best validation balanced accuracy: 0.833, threshold: 0.286
Epoch 07: Train: Loss 0.2105, Acc 0.572 |Val: Loss 0.2724, Acc 0.750, BalAcc 0.631, F1 0.839, AUC 0.429, Thresh 0.199
Epoch 08: Train: Loss 0.1677, Acc 0.610 |Val: Loss 0.2092, Acc 0.800, BalAcc 0.714, F1 0.867, AUC 0.679, Thresh 0.414
Epoch 09: Train: Loss 0.1384, Acc 0.617 |Val: Loss 0.1644, Acc 0.700, BalAcc 0.500, F1 0.824, AUC 0.607, Thresh 0.125
Epoch 10: Train: Loss 0.1359, Acc 0.663 |Val: Loss 0.0895, Acc 0.800, BalAcc 0.714, F1 0.867, AUC 0.702, Thresh 0.394
Epoch 11: Train: Loss 0.1114, Acc 0.667 |Val: Loss 0.2843, Acc 0.800, BalAcc 0.714, F1 0.867, AUC 0.702, Thresh 0.272
Epoch 12: Train: Loss 0.1414, Acc 0.697 |Val: Loss 0.1496, Acc 0.750, BalAcc 0.583, F1 0.848, AUC 0.643, Thresh 0.098
Epoch 13: Train: Loss 0.0972, Acc 0.739 |Val: Loss 0.2439, Acc 0.800, BalAcc 0.762, F1 0.857, AUC 0.643, Thresh 0.402
Early stopping after 13 epochs (no improvement for 7 epochs)
Best model saved: ./output/models/best_model_fold_4.pth (Val BalAcc: 0.833)
Test Results: Acc 0.875, BalAcc 0.833, F1 0.917, AUC 0.896 (threshold: 0.286)

=== Cross-Validation Results ===
Acc: 0.829 ± 0.059
BalAcc: 0.785 ± 0.094
F1: 0.879 ± 0.048
AUC: 0.838 ± 0.070