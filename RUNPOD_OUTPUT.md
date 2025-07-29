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
Fold 0: Train patients: 65 (images=6246, B/M = 19/46); Test patients: 17 (images=1663, B/M = 5/12)
Fold 1: Train patients: 65 (images=6228, B/M = 19/46); Test patients: 17 (images=1681, B/M = 5/12)
Fold 2: Train patients: 66 (images=6407, B/M = 19/47); Test patients: 16 (images=1502, B/M = 5/11)
Fold 3: Train patients: 66 (images=6217, B/M = 19/47); Test patients: 16 (images=1692, B/M = 5/11)
Fold 4: Train patients: 66 (images=6538, B/M = 20/46); Test patients: 16 (images=1371, B/M = 4/12)

===== Fold 0 =====
Train patients: 52, Val Patients: 13, Test patients: 17
Training samples per epoch: {'total_samples_per_epoch': 2184, 'class_distribution': {0: 636, 1: 1548}, 'oversampling_factor': 2.08}
Validation samples: 241, Test samples: 353
Patients with full 4 mags: 52
Inner training samples: 2184, batch size: 16
Class weights: Benign=3.47, Malignant=0.21
Epoch 01: Train: Loss 0.0272, Acc 0.554 | Val: Loss 0.0300, Acc 0.751, BalAcc 0.613, F1 0.844, AUC 0.709, Prec 0.747, Rec 0.970, Thresh 0.326 | LR: 0.000100
✅ New best validation balanced accuracy: 0.613, threshold: 0.326
📊 Mag Importance (Val BalAcc: 0.613): {'40': 0.25233668088912964, '100': 0.26298919320106506, '200': 0.25237640738487244, '400': 0.23229774832725525}
Epoch 02: Train: Loss 0.0251, Acc 0.608 | Val: Loss 0.0380, Acc 0.768, BalAcc 0.648, F1 0.851, AUC 0.671, Prec 0.766, Rec 0.958, Thresh 0.391 | LR: 0.000100
✅ New best validation balanced accuracy: 0.648, threshold: 0.391
📊 Mag Importance (Val BalAcc: 0.648): {'40': 0.2535244822502136, '100': 0.2623767554759979, '200': 0.24935278296470642, '400': 0.23474600911140442}
Epoch 03: Train: Loss 0.0208, Acc 0.642 | Val: Loss 0.0255, Acc 0.763, BalAcc 0.630, F1 0.851, AUC 0.723, Prec 0.755, Rec 0.976, Thresh 0.342 | LR: 0.000100
Epoch 04: Train: Loss 0.0188, Acc 0.656 | Val: Loss 0.0421, Acc 0.793, BalAcc 0.685, F1 0.866, AUC 0.699, Prec 0.785, Rec 0.964, Thresh 0.367 | LR: 0.000100
✅ New best validation balanced accuracy: 0.685, threshold: 0.367
📊 Mag Importance (Val BalAcc: 0.685): {'40': 0.2515738308429718, '100': 0.26277047395706177, '200': 0.253928005695343, '400': 0.2317277193069458}
Epoch 05: Train: Loss 0.0173, Acc 0.640 | Val: Loss 0.0189, Acc 0.813, BalAcc 0.760, F1 0.870, AUC 0.789, Prec 0.843, Rec 0.898, Thresh 0.439 | LR: 0.000100
✅ New best validation balanced accuracy: 0.760, threshold: 0.439
📊 Mag Importance (Val BalAcc: 0.760): {'40': 0.25184082984924316, '100': 0.2636050581932068, '200': 0.25152260065078735, '400': 0.23303155601024628}
Epoch 06: Train: Loss 0.0170, Acc 0.665 | Val: Loss 0.0234, Acc 0.813, BalAcc 0.719, F1 0.877, AUC 0.789, Prec 0.805, Rec 0.964, Thresh 0.415 | LR: 0.000100
Epoch 07: Train: Loss 0.0168, Acc 0.679 | Val: Loss 0.0288, Acc 0.813, BalAcc 0.703, F1 0.880, AUC 0.751, Prec 0.793, Rec 0.988, Thresh 0.367 | LR: 0.000100
Epoch 08: Train: Loss 0.0159, Acc 0.696 | Val: Loss 0.0281, Acc 0.813, BalAcc 0.726, F1 0.876, AUC 0.746, Prec 0.811, Rec 0.952, Thresh 0.423 | LR: 0.000100
Epoch 09: Train: Loss 0.0142, Acc 0.692 | Val: Loss 0.0406, Acc 0.826, BalAcc 0.776, F1 0.878, AUC 0.718, Prec 0.853, Rec 0.904, Thresh 0.464 | LR: 0.000100
✅ New best validation balanced accuracy: 0.776, threshold: 0.464
📊 Mag Importance (Val BalAcc: 0.776): {'40': 0.24841678142547607, '100': 0.2623597979545593, '200': 0.2593705654144287, '400': 0.22985291481018066}
Epoch 10: Train: Loss 0.0168, Acc 0.696 | Val: Loss 0.0227, Acc 0.830, BalAcc 0.764, F1 0.884, AUC 0.750, Prec 0.839, Rec 0.934, Thresh 0.472 | LR: 0.000100
Epoch 11: Train: Loss 0.0139, Acc 0.705 | Val: Loss 0.0406, Acc 0.817, BalAcc 0.759, F1 0.874, AUC 0.752, Prec 0.840, Rec 0.910, Thresh 0.456 | LR: 0.000100
Epoch 12: Train: Loss 0.0136, Acc 0.693 | Val: Loss 0.0208, Acc 0.846, BalAcc 0.780, F1 0.896, AUC 0.808, Prec 0.846, Rec 0.952, Thresh 0.423 | LR: 0.000100
✅ New best validation balanced accuracy: 0.780, threshold: 0.423
📊 Mag Importance (Val BalAcc: 0.780): {'40': 0.24615243077278137, '100': 0.26452356576919556, '200': 0.257068932056427, '400': 0.2322550266981125}
Epoch 13: Train: Loss 0.0143, Acc 0.700 | Val: Loss 0.0172, Acc 0.838, BalAcc 0.793, F1 0.886, AUC 0.829, Prec 0.864, Rec 0.910, Thresh 0.472 | LR: 0.000100
✅ New best validation balanced accuracy: 0.793, threshold: 0.472
📊 Mag Importance (Val BalAcc: 0.793): {'40': 0.24603161215782166, '100': 0.2601153254508972, '200': 0.2615094780921936, '400': 0.23234359920024872}
Epoch 14: Train: Loss 0.0128, Acc 0.722 | Val: Loss 0.0182, Acc 0.855, BalAcc 0.797, F1 0.900, AUC 0.816, Prec 0.859, Rec 0.946, Thresh 0.423 | LR: 0.000100
✅ New best validation balanced accuracy: 0.797, threshold: 0.423
📊 Mag Importance (Val BalAcc: 0.797): {'40': 0.24804481863975525, '100': 0.2608364224433899, '200': 0.2589976489543915, '400': 0.23212113976478577}
Epoch 15: Train: Loss 0.0134, Acc 0.706 | Val: Loss 0.0193, Acc 0.842, BalAcc 0.803, F1 0.888, AUC 0.804, Prec 0.873, Rec 0.904, Thresh 0.480 | LR: 0.000100
✅ New best validation balanced accuracy: 0.803, threshold: 0.480
📊 Mag Importance (Val BalAcc: 0.803): {'40': 0.24575527012348175, '100': 0.2634449005126953, '200': 0.2559121549129486, '400': 0.23488768935203552}
Epoch 16: Train: Loss 0.0132, Acc 0.721 | Val: Loss 0.0188, Acc 0.834, BalAcc 0.790, F1 0.883, AUC 0.804, Prec 0.863, Rec 0.904, Thresh 0.472 | LR: 0.000100
Epoch 17: Train: Loss 0.0126, Acc 0.726 | Val: Loss 0.0123, Acc 0.846, BalAcc 0.806, F1 0.891, AUC 0.897, Prec 0.874, Rec 0.910, Thresh 0.496 | LR: 0.000100
✅ New best validation balanced accuracy: 0.806, threshold: 0.496
📊 Mag Importance (Val BalAcc: 0.806): {'40': 0.24276702105998993, '100': 0.264133095741272, '200': 0.2584904134273529, '400': 0.234609454870224}
Epoch 18: Train: Loss 0.0115, Acc 0.728 | Val: Loss 0.0145, Acc 0.838, BalAcc 0.782, F1 0.888, AUC 0.866, Prec 0.852, Rec 0.928, Thresh 0.488 | LR: 0.000100
Epoch 19: Train: Loss 0.0113, Acc 0.740 | Val: Loss 0.0227, Acc 0.846, BalAcc 0.795, F1 0.893, AUC 0.810, Prec 0.861, Rec 0.928, Thresh 0.480 | LR: 0.000100
Epoch 20: Train: Loss 0.0124, Acc 0.746 | Val: Loss 0.0223, Acc 0.842, BalAcc 0.800, F1 0.889, AUC 0.788, Prec 0.869, Rec 0.910, Thresh 0.504 | LR: 0.000100
Epoch 21: Train: Loss 0.0111, Acc 0.728 | Val: Loss 0.0192, Acc 0.838, BalAcc 0.789, F1 0.887, AUC 0.812, Prec 0.860, Rec 0.916, Thresh 0.447 | LR: 0.000050
Epoch 22: Train: Loss 0.0109, Acc 0.760 | Val: Loss 0.0182, Acc 0.838, BalAcc 0.793, F1 0.886, AUC 0.843, Prec 0.864, Rec 0.910, Thresh 0.496 | LR: 0.000050
Epoch 23: Train: Loss 0.0120, Acc 0.726 | Val: Loss 0.0172, Acc 0.838, BalAcc 0.789, F1 0.887, AUC 0.856, Prec 0.860, Rec 0.916, Thresh 0.472 | LR: 0.000050
Epoch 24: Train: Loss 0.0122, Acc 0.732 | Val: Loss 0.0146, Acc 0.855, BalAcc 0.816, F1 0.897, AUC 0.869, Prec 0.879, Rec 0.916, Thresh 0.504 | LR: 0.000050
✅ New best validation balanced accuracy: 0.816, threshold: 0.504
📊 Mag Importance (Val BalAcc: 0.816): {'40': 0.24318084120750427, '100': 0.26276981830596924, '200': 0.2593640685081482, '400': 0.2346852719783783}
Epoch 25: Train: Loss 0.0110, Acc 0.721 | Val: Loss 0.0174, Acc 0.830, BalAcc 0.772, F1 0.883, AUC 0.832, Prec 0.846, Rec 0.922, Thresh 0.480 | LR: 0.000050
✅ Best model saved: ./output/models/best_model_fold_0.pth (Val BalAcc: 0.816)
⚡️ Test Results: Acc 0.949, BalAcc 0.950, F1 0.959, AUC 0.992, Precision 0.972, Recall 0.946 (threshold: 0.504)
📊 Confusion Matrix (Fold 0):
   [[TN: 126, FP:   6]
    [FN:  12, TP: 209]]
⚡ Avg Inference Time: 0.0092s per sample
Traceback (most recent call last):
  File "/workspace/MultiMagBC/main.py", line 343, in <module>
    main()
  File "/workspace/MultiMagBC/main.py", line 240, in main
    'fpr': metrics['fpr'],
           ~~~~~~~^^^^^^^
KeyError: 'fpr'