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
model.safetensors: 100%|---| 31.5M/31.5M [00:00<00:00, 88.2MB/s]
Epoch 01: Train: Loss 0.7820, Acc 0.484 | Val: Loss 0.7082, Acc 0.294, BalAcc 0.500, F1 0.000, AUC 0.517
Epoch 02: Train: Loss 0.6457, Acc 0.625 | Val: Loss 0.7096, Acc 0.294, BalAcc 0.500, F1 0.000, AUC 0.633
Epoch 03: Train: Loss 0.5471, Acc 0.781 | Val: Loss 0.6953, Acc 0.529, BalAcc 0.667, F1 0.500, AUC 0.950
Epoch 04: Train: Loss 0.6659, Acc 0.672 | Val: Loss 0.6747, Acc 0.706, BalAcc 0.500, F1 0.828, AUC 0.867
Epoch 05: Train: Loss 0.5482, Acc 0.719 | Val: Loss 0.6328, Acc 0.706, BalAcc 0.500, F1 0.828, AUC 0.950
Epoch 06: Train: Loss 0.4654, Acc 0.781 | Val: Loss 0.5766, Acc 0.706, BalAcc 0.500, F1 0.828, AUC 0.867
Epoch 07: Train: Loss 0.4797, Acc 0.812 | Val: Loss 0.6209, Acc 0.706, BalAcc 0.500, F1 0.828, AUC 0.917
Epoch 08: Train: Loss 0.5351, Acc 0.734 | Val: Loss 0.6044, Acc 0.706, BalAcc 0.500, F1 0.828, AUC 0.917
Epoch 09: Train: Loss 0.4741, Acc 0.844 | Val: Loss 0.5208, Acc 0.824, BalAcc 0.700, F1 0.889, AUC 0.900
Epoch 10: Train: Loss 0.4514, Acc 0.828 | Val: Loss 0.5349, Acc 0.941, BalAcc 0.900, F1 0.960, AUC 0.983

===== Fold 1 =====
Epoch 01: Train: Loss 0.5996, Acc 0.688 | Val: Loss 0.6802, Acc 0.706, BalAcc 0.500, F1 0.828, AUC 0.367
Epoch 02: Train: Loss 0.5964, Acc 0.719 | Val: Loss 0.6611, Acc 0.706, BalAcc 0.500, F1 0.828, AUC 0.550
Epoch 03: Train: Loss 0.4426, Acc 0.844 | Val: Loss 0.6663, Acc 0.706, BalAcc 0.500, F1 0.828, AUC 0.600
Epoch 04: Train: Loss 0.4269, Acc 0.891 | Val: Loss 0.6353, Acc 0.706, BalAcc 0.500, F1 0.828, AUC 0.817
Epoch 05: Train: Loss 0.3719, Acc 0.828 | Val: Loss 0.6354, Acc 0.706, BalAcc 0.500, F1 0.828, AUC 0.750
Epoch 06: Train: Loss 0.4665, Acc 0.766 | Val: Loss 0.6258, Acc 0.706, BalAcc 0.500, F1 0.828, AUC 0.883
Epoch 07: Train: Loss 0.3257, Acc 0.891 | Val: Loss 0.5382, Acc 0.824, BalAcc 0.700, F1 0.889, AUC 0.933
Epoch 08: Train: Loss 0.3766, Acc 0.750 | Val: Loss 0.5664, Acc 0.882, BalAcc 0.800, F1 0.923, AUC 0.850
Epoch 09: Train: Loss 0.3386, Acc 0.906 | Val: Loss 0.4859, Acc 0.824, BalAcc 0.758, F1 0.880, AUC 0.850
Epoch 10: Train: Loss 0.2871, Acc 0.938 | Val: Loss 0.4799, Acc 0.941, BalAcc 0.900, F1 0.960, AUC 0.883

===== Fold 2 =====
Epoch 01: Train: Loss 0.6235, Acc 0.719 | Val: Loss 0.7009, Acc 0.312, BalAcc 0.500, F1 0.000, AUC 0.564
Epoch 02: Train: Loss 0.5067, Acc 0.766 | Val: Loss 0.6908, Acc 0.750, BalAcc 0.709, F1 0.818, AUC 0.873
Epoch 03: Train: Loss 0.4531, Acc 0.859 | Val: Loss 0.6782, Acc 0.688, BalAcc 0.500, F1 0.815, AUC 0.855
Epoch 04: Train: Loss 0.4524, Acc 0.797 | Val: Loss 0.6600, Acc 0.875, BalAcc 0.800, F1 0.917, AUC 0.945
Epoch 05: Train: Loss 0.4027, Acc 0.859 | Val: Loss 0.6442, Acc 0.875, BalAcc 0.800, F1 0.917, AUC 0.909
Epoch 06: Train: Loss 0.3468, Acc 0.891 | Val: Loss 0.6179, Acc 0.812, BalAcc 0.755, F1 0.870, AUC 0.927
Epoch 07: Train: Loss 0.3229, Acc 0.891 | Val: Loss 0.5599, Acc 0.938, BalAcc 0.900, F1 0.957, AUC 0.945
Epoch 08: Train: Loss 0.4179, Acc 0.859 | Val: Loss 0.5359, Acc 0.875, BalAcc 0.800, F1 0.917, AUC 0.818
Epoch 09: Train: Loss 0.3810, Acc 0.859 | Val: Loss 0.4928, Acc 0.812, BalAcc 0.700, F1 0.880, AUC 0.982
Epoch 10: Train: Loss 0.3109, Acc 0.891 | Val: Loss 0.4807, Acc 0.875, BalAcc 0.855, F1 0.909, AUC 0.909

===== Fold 3 =====
Epoch 01: Train: Loss 0.7701, Acc 0.469 | Val: Loss 0.7080, Acc 0.312, BalAcc 0.500, F1 0.000, AUC 0.327
Epoch 02: Train: Loss 0.7521, Acc 0.516 | Val: Loss 0.7174, Acc 0.312, BalAcc 0.500, F1 0.000, AUC 0.855
Epoch 03: Train: Loss 0.6384, Acc 0.641 | Val: Loss 0.7271, Acc 0.312, BalAcc 0.500, F1 0.000, AUC 0.764
Epoch 04: Train: Loss 0.5856, Acc 0.734 | Val: Loss 0.7203, Acc 0.312, BalAcc 0.500, F1 0.000, AUC 0.782
Epoch 05: Train: Loss 0.4693, Acc 0.766 | Val: Loss 0.7170, Acc 0.312, BalAcc 0.500, F1 0.000, AUC 0.764
Epoch 06: Train: Loss 0.5693, Acc 0.734 | Val: Loss 0.7096, Acc 0.312, BalAcc 0.500, F1 0.000, AUC 0.764
Epoch 07: Train: Loss 0.5096, Acc 0.781 | Val: Loss 0.6783, Acc 0.500, BalAcc 0.636, F1 0.429, AUC 0.800
Epoch 08: Train: Loss 0.4944, Acc 0.688 | Val: Loss 0.6395, Acc 0.688, BalAcc 0.773, F1 0.706, AUC 0.945
Epoch 09: Train: Loss 0.4371, Acc 0.812 | Val: Loss 0.6520, Acc 0.562, BalAcc 0.627, F1 0.588, AUC 0.836
Epoch 10: Train: Loss 0.4496, Acc 0.828 | Val: Loss 0.6330, Acc 0.688, BalAcc 0.718, F1 0.737, AUC 0.782

===== Fold 4 =====
Epoch 01: Train: Loss 0.7168, Acc 0.484 | Val: Loss 0.7057, Acc 0.250, BalAcc 0.500, F1 0.000, AUC 0.479
Epoch 02: Train: Loss 0.6214, Acc 0.703 | Val: Loss 0.6979, Acc 0.250, BalAcc 0.500, F1 0.000, AUC 0.479
Epoch 03: Train: Loss 0.5389, Acc 0.719 | Val: Loss 0.6665, Acc 0.750, BalAcc 0.500, F1 0.857, AUC 0.562
Epoch 04: Train: Loss 0.4472, Acc 0.828 | Val: Loss 0.6403, Acc 0.750, BalAcc 0.500, F1 0.857, AUC 0.688
Epoch 05: Train: Loss 0.5332, Acc 0.812 | Val: Loss 0.6084, Acc 0.750, BalAcc 0.500, F1 0.857, AUC 0.875
Epoch 06: Train: Loss 0.4761, Acc 0.797 | Val: Loss 0.5936, Acc 0.750, BalAcc 0.500, F1 0.857, AUC 0.812

Epoch 07: Train: Loss 0.4961, Acc 0.766 | Val: Loss 0.5745, Acc 0.750, BalAcc 0.500, F1 0.857, AUC 0.896

Epoch 08: Train: Loss 0.4428, Acc 0.828 | Val: Loss 0.5491, Acc 0.812, BalAcc 0.625, F1 0.889, AUC 0.875
Epoch 09: Train: Loss 0.3936, Acc 0.844 | Val: Loss 0.5194, Acc 0.750, BalAcc 0.500, F1 0.857, AUC 0.917
Epoch 10: Train: Loss 0.3457, Acc 0.906 | Val: Loss 0.5076, Acc 0.812, BalAcc 0.625, F1 0.889, AUC 0.979

=== Cross-Validation Results ===
Acc: 0.851 ± 0.095 BalAcc: 0.800 ± 0.110
F1: 0.891 ± 0.082
AUC: 0.907 ± 0.074