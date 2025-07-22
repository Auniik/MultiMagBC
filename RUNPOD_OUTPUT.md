
===== Fold 0 =====
Train patients: 65, Test patients: 17
Training samples per epoch: 640 (avg utilization: 46.4%)
Class weights: Benign=1.71, Malignant=0.71
model.safetensors: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 31.5M/31.5M [00:00<00:00, 207MB/s]
Inner training samples: 513, batch size: 24
Inner split: Train 52, Val 13 patients
Epoch 01: Train: Loss 0.0990, Acc 0.661 |Val: Loss 0.0775, Acc 0.846, BalAcc 0.750, F1 0.900, AUC 0.861, Thresh 0.536
New best validation balanced accuracy: 0.750, threshold: 0.536
Epoch 02: Train: Loss 0.0684, Acc 0.802 |Val: Loss 0.0537, Acc 0.923, BalAcc 0.875, F1 0.947, AUC 0.889, Thresh 0.587
New best validation balanced accuracy: 0.875, threshold: 0.587
Epoch 03: Train: Loss 0.0496, Acc 0.865 |Val: Loss 0.0461, Acc 0.846, BalAcc 0.750, F1 0.900, AUC 0.917, Thresh 0.604
Epoch 04: Train: Loss 0.0391, Acc 0.907 |Val: Loss 0.0478, Acc 0.846, BalAcc 0.750, F1 0.900, AUC 0.861, Thresh 0.523
Epoch 05: Train: Loss 0.0306, Acc 0.921 |Val: Loss 0.0551, Acc 0.846, BalAcc 0.819, F1 0.889, AUC 0.889, Thresh 0.625
Epoch 06: Train: Loss 0.0214, Acc 0.950 |Val: Loss 0.1153, Acc 0.846, BalAcc 0.750, F1 0.900, AUC 0.889, Thresh 0.109
Epoch 07: Train: Loss 0.0298, Acc 0.919 |Val: Loss 0.0594, Acc 0.923, BalAcc 0.875, F1 0.947, AUC 0.806, Thresh 0.414
Epoch 08: Train: Loss 0.0165, Acc 0.960 |Val: Loss 0.0709, Acc 0.923, BalAcc 0.875, F1 0.947, AUC 0.889, Thresh 0.769
Epoch 09: Train: Loss 0.0178, Acc 0.956 |Val: Loss 0.0881, Acc 0.846, BalAcc 0.750, F1 0.900, AUC 0.833, Thresh 0.824
Early stopping after 9 epochs (no improvement for 7 epochs)
Best model saved: ./output/models/best_model_fold_0.pth (Val BalAcc: 0.875)
Test Results: Acc 0.941, BalAcc 0.958, F1 0.957, AUC 1.000 (threshold: 0.587)

===== Fold 1 =====
Train patients: 65, Test patients: 17
Training samples per epoch: 638 (avg utilization: 47.4%)
Class weights: Benign=1.71, Malignant=0.71
Inner training samples: 506, batch size: 24
Inner split: Train 52, Val 13 patients
Epoch 01: Train: Loss 0.1066, Acc 0.661 |Val: Loss 0.0762, Acc 0.923, BalAcc 0.875, F1 0.947, AUC 0.944, Thresh 0.549
New best validation balanced accuracy: 0.875, threshold: 0.549
Epoch 02: Train: Loss 0.0724, Acc 0.778 |Val: Loss 0.0429, Acc 0.923, BalAcc 0.944, F1 0.941, AUC 0.944, Thresh 0.658
New best validation balanced accuracy: 0.944, threshold: 0.658
Epoch 03: Train: Loss 0.0527, Acc 0.851 |Val: Loss 0.0276, Acc 1.000, BalAcc 1.000, F1 1.000, AUC 1.000, Thresh 0.377
New best validation balanced accuracy: 1.000, threshold: 0.377
Epoch 04: Train: Loss 0.0386, Acc 0.899 |Val: Loss 0.0959, Acc 1.000, BalAcc 1.000, F1 1.000, AUC 1.000, Thresh 0.297
Epoch 05: Train: Loss 0.0399, Acc 0.895 |Val: Loss 0.0318, Acc 0.923, BalAcc 0.875, F1 0.947, AUC 0.972, Thresh 0.338
Epoch 06: Train: Loss 0.0256, Acc 0.931 |Val: Loss 0.0073, Acc 1.000, BalAcc 1.000, F1 1.000, AUC 1.000, Thresh 0.788
Epoch 07: Train: Loss 0.0296, Acc 0.925 |Val: Loss 0.1126, Acc 1.000, BalAcc 1.000, F1 1.000, AUC 1.000, Thresh 0.100
Epoch 08: Train: Loss 0.0229, Acc 0.954 |Val: Loss 0.0122, Acc 1.000, BalAcc 1.000, F1 1.000, AUC 1.000, Thresh 0.576
Epoch 09: Train: Loss 0.0180, Acc 0.962 |Val: Loss 0.0208, Acc 1.000, BalAcc 1.000, F1 1.000, AUC 1.000, Thresh 0.508
Epoch 10: Train: Loss 0.0187, Acc 0.950 |Val: Loss 0.1653, Acc 1.000, BalAcc 1.000, F1 1.000, AUC 1.000, Thresh 0.170
Early stopping after 10 epochs (no improvement for 7 epochs)
Best model saved: ./output/models/best_model_fold_1.pth (Val BalAcc: 1.000)
Test Results: Acc 0.824, BalAcc 0.875, F1 0.857, AUC 1.000 (threshold: 0.377)

===== Fold 2 =====
Train patients: 66, Test patients: 16
Training samples per epoch: 652 (avg utilization: 47.4%)
Class weights: Benign=1.74, Malignant=0.70
Inner training samples: 512, batch size: 24
Inner split: Train 52, Val 14 patients
Epoch 01: Train: Loss 0.0937, Acc 0.675 |Val: Loss 0.0697, Acc 1.000, BalAcc 1.000, F1 1.000, AUC 1.000, Thresh 0.582
New best validation balanced accuracy: 1.000, threshold: 0.582
Epoch 02: Train: Loss 0.0593, Acc 0.788 |Val: Loss 0.0405, Acc 1.000, BalAcc 1.000, F1 1.000, AUC 1.000, Thresh 0.681
Epoch 03: Train: Loss 0.0388, Acc 0.875 |Val: Loss 0.0251, Acc 1.000, BalAcc 1.000, F1 1.000, AUC 1.000, Thresh 0.690
Epoch 04: Train: Loss 0.0400, Acc 0.897 |Val: Loss 0.0190, Acc 1.000, BalAcc 1.000, F1 1.000, AUC 1.000, Thresh 0.700
Epoch 05: Train: Loss 0.0360, Acc 0.905 |Val: Loss 0.0183, Acc 1.000, BalAcc 1.000, F1 1.000, AUC 1.000, Thresh 0.512
Epoch 06: Train: Loss 0.0193, Acc 0.950 |Val: Loss 0.0200, Acc 1.000, BalAcc 1.000, F1 1.000, AUC 1.000, Thresh 0.565
Epoch 07: Train: Loss 0.0171, Acc 0.962 |Val: Loss 0.0096, Acc 1.000, BalAcc 1.000, F1 1.000, AUC 1.000, Thresh 0.664
Epoch 08: Train: Loss 0.0165, Acc 0.964 |Val: Loss 0.0141, Acc 1.000, BalAcc 1.000, F1 1.000, AUC 1.000, Thresh 0.552
Early stopping after 8 epochs (no improvement for 7 epochs)
Best model saved: ./output/models/best_model_fold_2.pth (Val BalAcc: 1.000)
Test Results: Acc 0.812, BalAcc 0.809, F1 0.857, AUC 0.836 (threshold: 0.582)

===== Fold 3 =====
Train patients: 66, Test patients: 16
Training samples per epoch: 652 (avg utilization: 47.7%)
Class weights: Benign=1.74, Malignant=0.70
Inner training samples: 511, batch size: 24
Inner split: Train 52, Val 14 patients
Epoch 01: Train: Loss 0.1439, Acc 0.554 |Val: Loss 0.0901, Acc 0.857, BalAcc 0.750, F1 0.909, AUC 0.700, Thresh 0.511
New best validation balanced accuracy: 0.750, threshold: 0.511
Epoch 02: Train: Loss 0.0828, Acc 0.770 |Val: Loss 0.0514, Acc 0.929, BalAcc 0.875, F1 0.952, AUC 0.975, Thresh 0.613
New best validation balanced accuracy: 0.875, threshold: 0.613
Epoch 03: Train: Loss 0.0462, Acc 0.861 |Val: Loss 0.0297, Acc 0.929, BalAcc 0.875, F1 0.952, AUC 0.950, Thresh 0.629
Epoch 04: Train: Loss 0.0404, Acc 0.917 |Val: Loss 0.0263, Acc 0.929, BalAcc 0.875, F1 0.952, AUC 0.975, Thresh 0.689
Epoch 05: Train: Loss 0.0358, Acc 0.909 |Val: Loss 0.0302, Acc 0.929, BalAcc 0.875, F1 0.952, AUC 0.950, Thresh 0.609
Epoch 06: Train: Loss 0.0351, Acc 0.923 |Val: Loss 0.0514, Acc 0.857, BalAcc 0.750, F1 0.909, AUC 0.900, Thresh 0.387
Epoch 07: Train: Loss 0.0240, Acc 0.937 |Val: Loss 0.0134, Acc 1.000, BalAcc 1.000, F1 1.000, AUC 1.000, Thresh 0.702
New best validation balanced accuracy: 1.000, threshold: 0.702
Epoch 08: Train: Loss 0.0189, Acc 0.960 |Val: Loss 0.0123, Acc 1.000, BalAcc 1.000, F1 1.000, AUC 1.000, Thresh 0.783
Epoch 09: Train: Loss 0.0177, Acc 0.954 |Val: Loss 0.0167, Acc 1.000, BalAcc 1.000, F1 1.000, AUC 1.000, Thresh 0.809
Epoch 10: Train: Loss 0.0116, Acc 0.984 |Val: Loss 0.0063, Acc 1.000, BalAcc 1.000, F1 1.000, AUC 1.000, Thresh 0.795
Epoch 11: Train: Loss 0.0123, Acc 0.978 |Val: Loss 0.0072, Acc 1.000, BalAcc 1.000, F1 1.000, AUC 1.000, Thresh 0.726
Epoch 12: Train: Loss 0.0151, Acc 0.972 |Val: Loss 0.0075, Acc 1.000, BalAcc 1.000, F1 1.000, AUC 1.000, Thresh 0.732
Epoch 13: Train: Loss 0.0111, Acc 0.982 |Val: Loss 0.0127, Acc 1.000, BalAcc 1.000, F1 1.000, AUC 1.000, Thresh 0.640
Epoch 14: Train: Loss 0.0109, Acc 0.986 |Val: Loss 0.0049, Acc 1.000, BalAcc 1.000, F1 1.000, AUC 1.000, Thresh 0.868
Early stopping after 14 epochs (no improvement for 7 epochs)
Best model saved: ./output/models/best_model_fold_3.pth (Val BalAcc: 1.000)
Test Results: Acc 0.938, BalAcc 0.900, F1 0.957, AUC 0.964 (threshold: 0.702)

===== Fold 4 =====
Train patients: 66, Test patients: 16
Training samples per epoch: 654 (avg utilization: 45.7%)
Class weights: Benign=1.65, Malignant=0.72
Inner training samples: 520, batch size: 24
Inner split: Train 52, Val 14 patients
Epoch 01: Train: Loss 0.1155, Acc 0.587 |Val: Loss 0.0828, Acc 0.929, BalAcc 0.875, F1 0.952, AUC 0.825, Thresh 0.535
New best validation balanced accuracy: 0.875, threshold: 0.535
Epoch 02: Train: Loss 0.0700, Acc 0.782 |Val: Loss 0.0745, Acc 0.929, BalAcc 0.875, F1 0.952, AUC 0.775, Thresh 0.514
Epoch 03: Train: Loss 0.0476, Acc 0.851 |Val: Loss 0.0749, Acc 0.857, BalAcc 0.750, F1 0.909, AUC 0.725, Thresh 0.300
Epoch 04: Train: Loss 0.0372, Acc 0.881 |Val: Loss 0.0689, Acc 0.929, BalAcc 0.875, F1 0.952, AUC 0.950, Thresh 0.829
Epoch 05: Train: Loss 0.0316, Acc 0.893 |Val: Loss 0.0461, Acc 0.857, BalAcc 0.750, F1 0.909, AUC 0.850, Thresh 0.515
Epoch 06: Train: Loss 0.0237, Acc 0.950 |Val: Loss 0.0416, Acc 0.929, BalAcc 0.875, F1 0.952, AUC 0.950, Thresh 0.733
Epoch 07: Train: Loss 0.0142, Acc 0.968 |Val: Loss 0.0469, Acc 0.929, BalAcc 0.950, F1 0.947, AUC 0.950, Thresh 0.883
New best validation balanced accuracy: 0.950, threshold: 0.883
Epoch 08: Train: Loss 0.0231, Acc 0.948 |Val: Loss 0.0386, Acc 0.929, BalAcc 0.875, F1 0.952, AUC 0.975, Thresh 0.747
Epoch 09: Train: Loss 0.0232, Acc 0.948 |Val: Loss 0.0469, Acc 0.929, BalAcc 0.875, F1 0.952, AUC 0.925, Thresh 0.440
Epoch 10: Train: Loss 0.0171, Acc 0.962 |Val: Loss 0.0628, Acc 0.929, BalAcc 0.875, F1 0.952, AUC 0.975, Thresh 0.848
Epoch 11: Train: Loss 0.0205, Acc 0.964 |Val: Loss 0.0530, Acc 0.929, BalAcc 0.950, F1 0.947, AUC 0.950, Thresh 0.916
Epoch 12: Train: Loss 0.0091, Acc 0.982 |Val: Loss 0.0599, Acc 0.929, BalAcc 0.875, F1 0.952, AUC 0.975, Thresh 0.871
Epoch 13: Train: Loss 0.0126, Acc 0.978 |Val: Loss 0.0543, Acc 0.929, BalAcc 0.875, F1 0.952, AUC 0.975, Thresh 0.829
Epoch 14: Train: Loss 0.0134, Acc 0.970 |Val: Loss 0.0643, Acc 0.929, BalAcc 0.875, F1 0.952, AUC 0.975, Thresh 0.872
Early stopping after 14 epochs (no improvement for 7 epochs)
Best model saved: ./output/models/best_model_fold_4.pth (Val BalAcc: 0.950)
Test Results: Acc 0.875, BalAcc 0.917, F1 0.909, AUC 0.979 (threshold: 0.883)

=== Cross-Validation Results ===
Acc: 0.878 ± 0.054
BalAcc: 0.892 ± 0.049
F1: 0.907 ± 0.044
AUC: 0.956 ± 0.061