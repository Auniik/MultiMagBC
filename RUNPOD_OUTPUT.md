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
model.safetensors: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 31.5M/31.5M [00:00<00:00, 49.8MB/s]
Inner split: Train 52, Val 13 patients
Epoch 01: Train: Loss 0.1248, Acc 0.521 |Val: Loss 0.1010, Acc 0.692, BalAcc 0.500, F1 0.818, AUC 0.528, Thresh 0.497
New best validation balanced accuracy: 0.500, threshold: 0.497
Epoch 02: Train: Loss 0.1555, Acc 0.562 |Val: Loss 0.0966, Acc 0.692, BalAcc 0.500, F1 0.818, AUC 0.722, Thresh 0.506
Epoch 03: Train: Loss 0.1526, Acc 0.542 |Val: Loss 0.0937, Acc 0.846, BalAcc 0.750, F1 0.900, AUC 0.611, Thresh 0.514
New best validation balanced accuracy: 0.750, threshold: 0.514
Epoch 04: Train: Loss 0.0755, Acc 0.708 |Val: Loss 0.0901, Acc 1.000, BalAcc 1.000, F1 1.000, AUC 1.000, Thresh 0.521
New best validation balanced accuracy: 1.000, threshold: 0.521
Epoch 05: Train: Loss 0.0912, Acc 0.688 |Val: Loss 0.0848, Acc 0.923, BalAcc 0.875, F1 0.947, AUC 0.833, Thresh 0.533
Epoch 06: Train: Loss 0.1115, Acc 0.625 |Val: Loss 0.0805, Acc 0.846, BalAcc 0.819, F1 0.889, AUC 0.889, Thresh 0.550
Epoch 07: Train: Loss 0.1049, Acc 0.792 |Val: Loss 0.0747, Acc 1.000, BalAcc 1.000, F1 1.000, AUC 1.000, Thresh 0.562
Epoch 08: Train: Loss 0.0672, Acc 0.812 |Val: Loss 0.0725, Acc 0.923, BalAcc 0.875, F1 0.947, AUC 0.861, Thresh 0.567
Epoch 09: Train: Loss 0.0705, Acc 0.750 |Val: Loss 0.0739, Acc 0.692, BalAcc 0.500, F1 0.818, AUC 0.722, Thresh 0.557
Epoch 10: Train: Loss 0.0999, Acc 0.688 |Val: Loss 0.0689, Acc 0.923, BalAcc 0.875, F1 0.947, AUC 0.833, Thresh 0.586
Epoch 11: Train: Loss 0.0732, Acc 0.771 |Val: Loss 0.0665, Acc 0.846, BalAcc 0.750, F1 0.900, AUC 0.806, Thresh 0.548
Early stopping after 11 epochs (no improvement for 7 epochs)
Best model saved: ./output/models/best_model_fold_0.pth (Val BalAcc: 1.000)
Test Results: Acc 0.882, BalAcc 0.800, F1 0.923, AUC 0.967 (threshold: 0.521)

===== Fold 1 =====
Train patients: 65, Test patients: 17
Class weights: Benign=1.71, Malignant=0.71
Inner split: Train 52, Val 13 patients
Epoch 01: Train: Loss 0.1213, Acc 0.521 |Val: Loss 0.1069, Acc 0.692, BalAcc 0.500, F1 0.818, AUC 0.139, Thresh 0.483
New best validation balanced accuracy: 0.500, threshold: 0.483
Epoch 02: Train: Loss 0.1363, Acc 0.562 |Val: Loss 0.1078, Acc 0.769, BalAcc 0.625, F1 0.857, AUC 0.667, Thresh 0.481
New best validation balanced accuracy: 0.625, threshold: 0.481
Epoch 03: Train: Loss 0.1145, Acc 0.646 |Val: Loss 0.1056, Acc 0.769, BalAcc 0.625, F1 0.857, AUC 0.667, Thresh 0.484
Epoch 04: Train: Loss 0.1038, Acc 0.583 |Val: Loss 0.1056, Acc 0.846, BalAcc 0.750, F1 0.900, AUC 0.722, Thresh 0.483
New best validation balanced accuracy: 0.750, threshold: 0.483
Epoch 05: Train: Loss 0.0977, Acc 0.708 |Val: Loss 0.0983, Acc 0.846, BalAcc 0.750, F1 0.900, AUC 0.889, Thresh 0.492
Epoch 06: Train: Loss 0.1106, Acc 0.688 |Val: Loss 0.0912, Acc 0.923, BalAcc 0.875, F1 0.947, AUC 0.972, Thresh 0.497
New best validation balanced accuracy: 0.875, threshold: 0.497
Epoch 07: Train: Loss 0.1228, Acc 0.646 |Val: Loss 0.0848, Acc 1.000, BalAcc 1.000, F1 1.000, AUC 1.000, Thresh 0.518
New best validation balanced accuracy: 1.000, threshold: 0.518
Epoch 08: Train: Loss 0.0936, Acc 0.771 |Val: Loss 0.0779, Acc 1.000, BalAcc 1.000, F1 1.000, AUC 1.000, Thresh 0.529
Epoch 09: Train: Loss 0.0946, Acc 0.688 |Val: Loss 0.0731, Acc 1.000, BalAcc 1.000, F1 1.000, AUC 1.000, Thresh 0.540
Epoch 10: Train: Loss 0.0948, Acc 0.729 |Val: Loss 0.0732, Acc 1.000, BalAcc 1.000, F1 1.000, AUC 1.000, Thresh 0.532
Epoch 11: Train: Loss 0.0948, Acc 0.708 |Val: Loss 0.0625, Acc 1.000, BalAcc 1.000, F1 1.000, AUC 1.000, Thresh 0.573
Epoch 12: Train: Loss 0.1033, Acc 0.646 |Val: Loss 0.0557, Acc 1.000, BalAcc 1.000, F1 1.000, AUC 1.000, Thresh 0.557
Epoch 13: Train: Loss 0.0679, Acc 0.750 |Val: Loss 0.0566, Acc 0.923, BalAcc 0.944, F1 0.941, AUC 0.944, Thresh 0.603
Epoch 14: Train: Loss 0.0719, Acc 0.688 |Val: Loss 0.0517, Acc 0.923, BalAcc 0.875, F1 0.947, AUC 0.972, Thresh 0.581
Early stopping after 14 epochs (no improvement for 7 epochs)
Best model saved: ./output/models/best_model_fold_1.pth (Val BalAcc: 1.000)
Test Results: Acc 0.765, BalAcc 0.658, F1 0.846, AUC 0.867 (threshold: 0.518)

===== Fold 2 =====
Train patients: 66, Test patients: 16
Class weights: Benign=1.74, Malignant=0.70
Inner split: Train 52, Val 14 patients
Epoch 01: Train: Loss 0.1481, Acc 0.500 |Val: Loss 0.1040, Acc 0.714, BalAcc 0.500, F1 0.833, AUC 0.625, Thresh 0.493
New best validation balanced accuracy: 0.500, threshold: 0.493
Epoch 02: Train: Loss 0.1199, Acc 0.583 |Val: Loss 0.0977, Acc 0.714, BalAcc 0.500, F1 0.833, AUC 0.400, Thresh 0.508
Epoch 03: Train: Loss 0.1099, Acc 0.729 |Val: Loss 0.0906, Acc 0.714, BalAcc 0.500, F1 0.833, AUC 0.525, Thresh 0.525
Epoch 04: Train: Loss 0.0990, Acc 0.646 |Val: Loss 0.0849, Acc 0.929, BalAcc 0.875, F1 0.952, AUC 0.950, Thresh 0.538
New best validation balanced accuracy: 0.875, threshold: 0.538
Epoch 05: Train: Loss 0.0792, Acc 0.667 |Val: Loss 0.0814, Acc 0.929, BalAcc 0.875, F1 0.952, AUC 0.975, Thresh 0.547
Epoch 06: Train: Loss 0.1017, Acc 0.562 |Val: Loss 0.0754, Acc 0.929, BalAcc 0.875, F1 0.952, AUC 0.950, Thresh 0.564
Epoch 07: Train: Loss 0.0875, Acc 0.708 |Val: Loss 0.0705, Acc 1.000, BalAcc 1.000, F1 1.000, AUC 1.000, Thresh 0.575
New best validation balanced accuracy: 1.000, threshold: 0.575
Epoch 08: Train: Loss 0.1442, Acc 0.667 |Val: Loss 0.0651, Acc 0.929, BalAcc 0.950, F1 0.947, AUC 0.950, Thresh 0.599
Epoch 09: Train: Loss 0.0772, Acc 0.771 |Val: Loss 0.0597, Acc 1.000, BalAcc 1.000, F1 1.000, AUC 1.000, Thresh 0.610
Epoch 10: Train: Loss 0.1018, Acc 0.729 |Val: Loss 0.0589, Acc 1.000, BalAcc 1.000, F1 1.000, AUC 1.000, Thresh 0.597
Epoch 11: Train: Loss 0.0586, Acc 0.792 |Val: Loss 0.0605, Acc 1.000, BalAcc 1.000, F1 1.000, AUC 1.000, Thresh 0.580
Epoch 12: Train: Loss 0.0920, Acc 0.792 |Val: Loss 0.0498, Acc 1.000, BalAcc 1.000, F1 1.000, AUC 1.000, Thresh 0.561
Epoch 13: Train: Loss 0.0908, Acc 0.688 |Val: Loss 0.0489, Acc 1.000, BalAcc 1.000, F1 1.000, AUC 1.000, Thresh 0.573
Epoch 14: Train: Loss 0.0783, Acc 0.708 |Val: Loss 0.0560, Acc 0.857, BalAcc 0.750, F1 0.909, AUC 0.925, Thresh 0.551
Early stopping after 14 epochs (no improvement for 7 epochs)
Best model saved: ./output/models/best_model_fold_2.pth (Val BalAcc: 1.000)
Test Results: Acc 0.688, BalAcc 0.664, F1 0.762, AUC 0.745 (threshold: 0.575)

===== Fold 3 =====
Train patients: 66, Test patients: 16
Class weights: Benign=1.74, Malignant=0.70
Inner split: Train 52, Val 14 patients
Epoch 01: Train: Loss 0.3546, Acc 0.396 |Val: Loss 0.1169, Acc 0.714, BalAcc 0.500, F1 0.833, AUC 0.300, Thresh 0.468
New best validation balanced accuracy: 0.500, threshold: 0.468
Epoch 02: Train: Loss 0.2783, Acc 0.417 |Val: Loss 0.1150, Acc 0.714, BalAcc 0.500, F1 0.833, AUC 0.350, Thresh 0.470
Epoch 03: Train: Loss 0.2828, Acc 0.396 |Val: Loss 0.1112, Acc 0.786, BalAcc 0.625, F1 0.870, AUC 0.675, Thresh 0.476
New best validation balanced accuracy: 0.625, threshold: 0.476
Epoch 04: Train: Loss 0.2427, Acc 0.479 |Val: Loss 0.1076, Acc 0.786, BalAcc 0.625, F1 0.870, AUC 0.725, Thresh 0.479
Epoch 05: Train: Loss 0.1868, Acc 0.542 |Val: Loss 0.1005, Acc 0.857, BalAcc 0.750, F1 0.909, AUC 0.600, Thresh 0.493
New best validation balanced accuracy: 0.750, threshold: 0.493
Epoch 06: Train: Loss 0.2315, Acc 0.500 |Val: Loss 0.0894, Acc 0.786, BalAcc 0.625, F1 0.870, AUC 0.725, Thresh 0.508
Epoch 07: Train: Loss 0.2573, Acc 0.417 |Val: Loss 0.0856, Acc 0.857, BalAcc 0.750, F1 0.909, AUC 0.800, Thresh 0.525
Epoch 08: Train: Loss 0.1889, Acc 0.562 |Val: Loss 0.0775, Acc 0.857, BalAcc 0.750, F1 0.909, AUC 0.825, Thresh 0.513
Epoch 09: Train: Loss 0.1838, Acc 0.521 |Val: Loss 0.0779, Acc 0.857, BalAcc 0.750, F1 0.909, AUC 0.650, Thresh 0.534
Epoch 10: Train: Loss 0.1676, Acc 0.625 |Val: Loss 0.0711, Acc 0.857, BalAcc 0.825, F1 0.900, AUC 0.725, Thresh 0.548
New best validation balanced accuracy: 0.825, threshold: 0.548
Epoch 11: Train: Loss 0.0868, Acc 0.667 |Val: Loss 0.0639, Acc 0.929, BalAcc 0.875, F1 0.952, AUC 0.975, Thresh 0.512
New best validation balanced accuracy: 0.875, threshold: 0.512
Epoch 12: Train: Loss 0.1716, Acc 0.667 |Val: Loss 0.0663, Acc 0.857, BalAcc 0.825, F1 0.900, AUC 0.900, Thresh 0.539
Epoch 13: Train: Loss 0.1652, Acc 0.646 |Val: Loss 0.0821, Acc 0.857, BalAcc 0.750, F1 0.909, AUC 0.775, Thresh 0.457
Epoch 14: Train: Loss 0.1322, Acc 0.729 |Val: Loss 0.0726, Acc 0.929, BalAcc 0.875, F1 0.952, AUC 0.925, Thresh 0.429
Epoch 15: Train: Loss 0.1074, Acc 0.708 |Val: Loss 0.0847, Acc 0.929, BalAcc 0.950, F1 0.947, AUC 0.950, Thresh 0.503
New best validation balanced accuracy: 0.950, threshold: 0.503
Epoch 16: Train: Loss 0.0972, Acc 0.729 |Val: Loss 0.0636, Acc 0.857, BalAcc 0.750, F1 0.909, AUC 0.925, Thresh 0.433
Epoch 17: Train: Loss 0.1115, Acc 0.729 |Val: Loss 0.0635, Acc 0.857, BalAcc 0.750, F1 0.909, AUC 0.900, Thresh 0.465
Epoch 18: Train: Loss 0.1026, Acc 0.854 |Val: Loss 0.0639, Acc 0.857, BalAcc 0.750, F1 0.909, AUC 0.850, Thresh 0.436
Epoch 19: Train: Loss 0.1185, Acc 0.625 |Val: Loss 0.0534, Acc 0.929, BalAcc 0.875, F1 0.952, AUC 0.975, Thresh 0.510
Epoch 20: Train: Loss 0.0917, Acc 0.792 |Val: Loss 0.0777, Acc 0.929, BalAcc 0.875, F1 0.952, AUC 0.900, Thresh 0.359
Epoch 21: Train: Loss 0.0759, Acc 0.833 |Val: Loss 0.1095, Acc 0.786, BalAcc 0.700, F1 0.857, AUC 0.725, Thresh 0.515
Epoch 22: Train: Loss 0.0817, Acc 0.771 |Val: Loss 0.0769, Acc 0.857, BalAcc 0.825, F1 0.900, AUC 0.850, Thresh 0.480
Early stopping after 22 epochs (no improvement for 7 epochs)
Best model saved: ./output/models/best_model_fold_3.pth (Val BalAcc: 0.950)
Test Results: Acc 0.812, BalAcc 0.700, F1 0.880, AUC 0.855 (threshold: 0.503)

===== Fold 4 =====
Train patients: 66, Test patients: 16
Class weights: Benign=1.65, Malignant=0.72
Inner split: Train 52, Val 14 patients
Epoch 01: Train: Loss 0.0915, Acc 0.667 |Val: Loss 0.0908, Acc 0.714, BalAcc 0.500, F1 0.833, AUC 0.600, Thresh 0.525
New best validation balanced accuracy: 0.500, threshold: 0.525
Epoch 02: Train: Loss 0.0907, Acc 0.708 |Val: Loss 0.0858, Acc 0.857, BalAcc 0.750, F1 0.909, AUC 0.625, Thresh 0.540
New best validation balanced accuracy: 0.750, threshold: 0.540
Epoch 03: Train: Loss 0.0731, Acc 0.708 |Val: Loss 0.0846, Acc 0.786, BalAcc 0.775, F1 0.842, AUC 0.650, Thresh 0.543
New best validation balanced accuracy: 0.775, threshold: 0.543
Epoch 04: Train: Loss 0.0952, Acc 0.604 |Val: Loss 0.0781, Acc 0.786, BalAcc 0.700, F1 0.857, AUC 0.625, Thresh 0.563
Epoch 05: Train: Loss 0.0530, Acc 0.812 |Val: Loss 0.0731, Acc 0.786, BalAcc 0.625, F1 0.870, AUC 0.775, Thresh 0.574
Epoch 06: Train: Loss 0.0907, Acc 0.750 |Val: Loss 0.0685, Acc 0.857, BalAcc 0.750, F1 0.909, AUC 0.900, Thresh 0.598
Epoch 07: Train: Loss 0.1075, Acc 0.625 |Val: Loss 0.0661, Acc 0.857, BalAcc 0.750, F1 0.909, AUC 0.850, Thresh 0.627
Epoch 08: Train: Loss 0.0902, Acc 0.646 |Val: Loss 0.0644, Acc 0.786, BalAcc 0.625, F1 0.870, AUC 0.725, Thresh 0.630
Epoch 09: Train: Loss 0.0568, Acc 0.792 |Val: Loss 0.0609, Acc 0.857, BalAcc 0.750, F1 0.909, AUC 0.775, Thresh 0.632
Epoch 10: Train: Loss 0.0778, Acc 0.688 |Val: Loss 0.0662, Acc 0.786, BalAcc 0.625, F1 0.870, AUC 0.500, Thresh 0.646
Early stopping after 10 epochs (no improvement for 7 epochs)
Best model saved: ./output/models/best_model_fold_4.pth (Val BalAcc: 0.775)
Test Results: Acc 0.750, BalAcc 0.500, F1 0.857, AUC 0.917 (threshold: 0.543)

=== Cross-Validation Results ===
Acc: 0.779 ± 0.065
BalAcc: 0.664 ± 0.097
F1: 0.854 ± 0.053
AUC: 0.870 ± 0.074