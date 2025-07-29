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
Train patients: 48, Val Patients: 17, Test patients: 17
Training samples per epoch: {'total_samples_per_epoch': 2010, 'class_distribution': {1: 1413, 0: 597}, 'oversampling_factor': 2.05}
Validation samples: 311, Test samples: 353
Patients with full 4 mags: 48
Inner training samples: 2010, batch size: 16
📊 Class weights (balanced): Benign=1.71, Malignant=0.71
   Weight ratio: 2.4x (was 16.5x)
Epoch 01: Train: Loss 0.1012, Acc 0.548 | Val: Loss 0.1078, Acc 0.762, BalAcc 0.617, F1 0.851, AUC 0.676, Prec 0.771, Rec 0.951, Thresh 0.415 | LR: 0.000100
✅ New best validation balanced accuracy: 0.617, threshold: 0.415
📊 Mag Importance (Val BalAcc: 0.617): {'40': 0.2532173991203308, '100': 0.2629815936088562, '200': 0.2544262409210205, '400': 0.2293746918439865}
Epoch 02: Train: Loss 0.0903, Acc 0.615 | Val: Loss 0.0960, Acc 0.762, BalAcc 0.641, F1 0.847, AUC 0.684, Prec 0.785, Rec 0.919, Thresh 0.456 | LR: 0.000100
✅ New best validation balanced accuracy: 0.641, threshold: 0.456
📊 Mag Importance (Val BalAcc: 0.641): {'40': 0.25296229124069214, '100': 0.26554247736930847, '200': 0.2525258958339691, '400': 0.2289692610502243}
Epoch 03: Train: Loss 0.0850, Acc 0.650 | Val: Loss 0.1031, Acc 0.797, BalAcc 0.742, F1 0.860, AUC 0.759, Prec 0.851, Rec 0.870, Thresh 0.472 | LR: 0.000100
✅ New best validation balanced accuracy: 0.742, threshold: 0.472
📊 Mag Importance (Val BalAcc: 0.742): {'40': 0.25349903106689453, '100': 0.26724180579185486, '200': 0.2505045533180237, '400': 0.22875456511974335}
Epoch 04: Train: Loss 0.0791, Acc 0.668 | Val: Loss 0.0792, Acc 0.833, BalAcc 0.756, F1 0.889, AUC 0.812, Prec 0.849, Rec 0.933, Thresh 0.472 | LR: 0.000100
✅ New best validation balanced accuracy: 0.756, threshold: 0.472
📊 Mag Importance (Val BalAcc: 0.756): {'40': 0.25369760394096375, '100': 0.2658001482486725, '200': 0.25031188130378723, '400': 0.23019035160541534}
Epoch 05: Train: Loss 0.0742, Acc 0.674 | Val: Loss 0.0695, Acc 0.820, BalAcc 0.730, F1 0.882, AUC 0.830, Prec 0.833, Rec 0.937, Thresh 0.464 | LR: 0.000100
Epoch 06: Train: Loss 0.0743, Acc 0.675 | Val: Loss 0.0591, Acc 0.814, BalAcc 0.715, F1 0.879, AUC 0.835, Prec 0.824, Rec 0.942, Thresh 0.472 | LR: 0.000100
Epoch 07: Train: Loss 0.0761, Acc 0.686 | Val: Loss 0.0756, Acc 0.804, BalAcc 0.677, F1 0.876, AUC 0.757, Prec 0.800, Rec 0.969, Thresh 0.423 | LR: 0.000100
Epoch 08: Train: Loss 0.0686, Acc 0.708 | Val: Loss 0.0693, Acc 0.855, BalAcc 0.796, F1 0.902, AUC 0.836, Prec 0.874, Rec 0.933, Thresh 0.504 | LR: 0.000100
✅ New best validation balanced accuracy: 0.796, threshold: 0.504
📊 Mag Importance (Val BalAcc: 0.796): {'40': 0.25329482555389404, '100': 0.26592618227005005, '200': 0.24814943969249725, '400': 0.23262961208820343}
Epoch 09: Train: Loss 0.0653, Acc 0.719 | Val: Loss 0.0583, Acc 0.849, BalAcc 0.767, F1 0.901, AUC 0.865, Prec 0.852, Rec 0.955, Thresh 0.504 | LR: 0.000100
Epoch 10: Train: Loss 0.0624, Acc 0.728 | Val: Loss 0.0665, Acc 0.855, BalAcc 0.772, F1 0.905, AUC 0.849, Prec 0.853, Rec 0.964, Thresh 0.480 | LR: 0.000100
Epoch 11: Train: Loss 0.0662, Acc 0.709 | Val: Loss 0.0636, Acc 0.865, BalAcc 0.796, F1 0.910, AUC 0.856, Prec 0.869, Rec 0.955, Thresh 0.472 | LR: 0.000100
Epoch 12: Train: Loss 0.0644, Acc 0.739 | Val: Loss 0.0634, Acc 0.875, BalAcc 0.823, F1 0.915, AUC 0.864, Prec 0.890, Rec 0.942, Thresh 0.528 | LR: 0.000100
✅ New best validation balanced accuracy: 0.823, threshold: 0.528
📊 Mag Importance (Val BalAcc: 0.823): {'40': 0.2558528780937195, '100': 0.2625475823879242, '200': 0.25039809942245483, '400': 0.2312014400959015}
Epoch 13: Train: Loss 0.0642, Acc 0.718 | Val: Loss 0.0699, Acc 0.862, BalAcc 0.824, F1 0.904, AUC 0.854, Prec 0.898, Rec 0.910, Thresh 0.520 | LR: 0.000100
✅ New best validation balanced accuracy: 0.824, threshold: 0.520
📊 Mag Importance (Val BalAcc: 0.824): {'40': 0.25548455119132996, '100': 0.26503604650497437, '200': 0.24863329529762268, '400': 0.2308460772037506}
Epoch 14: Train: Loss 0.0604, Acc 0.733 | Val: Loss 0.0577, Acc 0.859, BalAcc 0.784, F1 0.906, AUC 0.857, Prec 0.862, Rec 0.955, Thresh 0.464 | LR: 0.000100
Epoch 15: Train: Loss 0.0597, Acc 0.715 | Val: Loss 0.0599, Acc 0.871, BalAcc 0.824, F1 0.912, AUC 0.867, Prec 0.893, Rec 0.933, Thresh 0.496 | LR: 0.000100
Epoch 16: Train: Loss 0.0602, Acc 0.709 | Val: Loss 0.0539, Acc 0.881, BalAcc 0.828, F1 0.920, AUC 0.890, Prec 0.891, Rec 0.951, Thresh 0.480 | LR: 0.000100
✅ New best validation balanced accuracy: 0.828, threshold: 0.480
📊 Mag Importance (Val BalAcc: 0.828): {'40': 0.2522147297859192, '100': 0.2650991678237915, '200': 0.25142818689346313, '400': 0.23125791549682617}
Epoch 17: Train: Loss 0.0579, Acc 0.746 | Val: Loss 0.0527, Acc 0.878, BalAcc 0.846, F1 0.915, AUC 0.890, Prec 0.911, Rec 0.919, Thresh 0.504 | LR: 0.000100
✅ New best validation balanced accuracy: 0.846, threshold: 0.504
📊 Mag Importance (Val BalAcc: 0.846): {'40': 0.2543245255947113, '100': 0.2649274170398712, '200': 0.24848487973213196, '400': 0.23226313292980194}
Epoch 18: Train: Loss 0.0640, Acc 0.751 | Val: Loss 0.0523, Acc 0.875, BalAcc 0.840, F1 0.913, AUC 0.891, Prec 0.907, Rec 0.919, Thresh 0.536 | LR: 0.000100
Epoch 19: Train: Loss 0.0555, Acc 0.759 | Val: Loss 0.0542, Acc 0.868, BalAcc 0.812, F1 0.911, AUC 0.892, Prec 0.882, Rec 0.942, Thresh 0.504 | LR: 0.000100
Epoch 20: Train: Loss 0.0571, Acc 0.722 | Val: Loss 0.0489, Acc 0.881, BalAcc 0.845, F1 0.918, AUC 0.915, Prec 0.908, Rec 0.928, Thresh 0.569 | LR: 0.000100
Epoch 21: Train: Loss 0.0538, Acc 0.741 | Val: Loss 0.0536, Acc 0.865, BalAcc 0.806, F1 0.909, AUC 0.890, Prec 0.879, Rec 0.942, Thresh 0.520 | LR: 0.000050
Epoch 22: Train: Loss 0.0568, Acc 0.739 | Val: Loss 0.0619, Acc 0.859, BalAcc 0.805, F1 0.904, AUC 0.883, Prec 0.881, Rec 0.928, Thresh 0.528 | LR: 0.000050
Epoch 23: Train: Loss 0.0523, Acc 0.760 | Val: Loss 0.0535, Acc 0.884, BalAcc 0.850, F1 0.920, AUC 0.915, Prec 0.912, Rec 0.928, Thresh 0.577 | LR: 0.000050
✅ New best validation balanced accuracy: 0.850, threshold: 0.577
📊 Mag Importance (Val BalAcc: 0.850): {'40': 0.251420259475708, '100': 0.2655254006385803, '200': 0.2515873312950134, '400': 0.23146696388721466}
Epoch 24: Train: Loss 0.0536, Acc 0.750 | Val: Loss 0.0534, Acc 0.868, BalAcc 0.850, F1 0.907, AUC 0.905, Prec 0.921, Rec 0.892, Thresh 0.512 | LR: 0.000050
Epoch 25: Train: Loss 0.0545, Acc 0.743 | Val: Loss 0.0526, Acc 0.862, BalAcc 0.821, F1 0.905, AUC 0.907, Prec 0.895, Rec 0.915, Thresh 0.585 | LR: 0.000050
✅ Best model saved: ./output/models/best_model_fold_0.pth (Val BalAcc: 0.850)
⚡️ Test Results: Acc 0.932, BalAcc 0.946, F1 0.943, AUC 0.996, Precision 1.000, Recall 0.891 (threshold: 0.577)
📊 Confusion Matrix (Fold 0):
   [[TN: 132, FP:   0]
    [FN:  24, TP: 197]]
⚡ Avg Inference Time: 0.0104s per sample
📌 Final Magnification Importance (Fold 0): {'40': 0.2443598210811615, '100': 0.2682781219482422, '200': 0.25157809257507324, '400': 0.23578400909900665}
💾 Results saved to: ./output/results/fold_0_results.json

📊 Generating GradCAM visualizations for fold 0...
✅ Generated 5 GradCAM visualizations for fold 0

===== Fold 1 =====
Train patients: 48, Val Patients: 17, Test patients: 17
Training samples per epoch: {'total_samples_per_epoch': 2010, 'class_distribution': {1: 1410, 0: 600}, 'oversampling_factor': 2.08}
Validation samples: 322, Test samples: 359
Patients with full 4 mags: 48
Inner training samples: 2010, batch size: 16
📊 Class weights (balanced): Benign=1.71, Malignant=0.71
   Weight ratio: 2.4x (was 16.5x)
Epoch 01: Train: Loss 0.1011, Acc 0.548 | Val: Loss 0.0711, Acc 0.829, BalAcc 0.739, F1 0.889, AUC 0.811, Prec 0.849, Rec 0.932, Thresh 0.431 | LR: 0.000100
✅ New best validation balanced accuracy: 0.739, threshold: 0.431
📊 Mag Importance (Val BalAcc: 0.739): {'40': 0.24906949698925018, '100': 0.278440922498703, '200': 0.23247691988945007, '400': 0.24001264572143555}
Epoch 02: Train: Loss 0.0831, Acc 0.622 | Val: Loss 0.0476, Acc 0.866, BalAcc 0.791, F1 0.913, AUC 0.915, Prec 0.875, Rec 0.953, Thresh 0.456 | LR: 0.000100
✅ New best validation balanced accuracy: 0.791, threshold: 0.456
📊 Mag Importance (Val BalAcc: 0.791): {'40': 0.24844488501548767, '100': 0.2729051113128662, '200': 0.23506489396095276, '400': 0.24358513951301575}
Epoch 03: Train: Loss 0.0751, Acc 0.672 | Val: Loss 0.0484, Acc 0.866, BalAcc 0.776, F1 0.914, AUC 0.900, Prec 0.864, Rec 0.970, Thresh 0.488 | LR: 0.000100
Epoch 04: Train: Loss 0.0733, Acc 0.689 | Val: Loss 0.0462, Acc 0.907, BalAcc 0.855, F1 0.938, AUC 0.912, Prec 0.912, Rec 0.966, Thresh 0.480 | LR: 0.000100
✅ New best validation balanced accuracy: 0.855, threshold: 0.480
📊 Mag Importance (Val BalAcc: 0.855): {'40': 0.2451304942369461, '100': 0.27696093916893005, '200': 0.23655788600444794, '400': 0.24135065078735352}
Epoch 05: Train: Loss 0.0695, Acc 0.694 | Val: Loss 0.0401, Acc 0.891, BalAcc 0.856, F1 0.926, AUC 0.951, Prec 0.921, Rec 0.932, Thresh 0.528 | LR: 0.000100
✅ New best validation balanced accuracy: 0.856, threshold: 0.528
📊 Mag Importance (Val BalAcc: 0.856): {'40': 0.2464296519756317, '100': 0.2771524488925934, '200': 0.23428186774253845, '400': 0.24213600158691406}
Epoch 06: Train: Loss 0.0724, Acc 0.691 | Val: Loss 0.0363, Acc 0.932, BalAcc 0.902, F1 0.954, AUC 0.971, Prec 0.942, Rec 0.966, Thresh 0.504 | LR: 0.000100
✅ New best validation balanced accuracy: 0.902, threshold: 0.504
📊 Mag Importance (Val BalAcc: 0.902): {'40': 0.24475200474262238, '100': 0.2801079750061035, '200': 0.2331848442554474, '400': 0.2419552057981491}
Epoch 07: Train: Loss 0.0671, Acc 0.707 | Val: Loss 0.0367, Acc 0.932, BalAcc 0.913, F1 0.953, AUC 0.966, Prec 0.953, Rec 0.953, Thresh 0.472 | LR: 0.000100
✅ New best validation balanced accuracy: 0.913, threshold: 0.472
📊 Mag Importance (Val BalAcc: 0.913): {'40': 0.2426203042268753, '100': 0.2818008065223694, '200': 0.23374418914318085, '400': 0.24183467030525208}
Epoch 08: Train: Loss 0.0690, Acc 0.694 | Val: Loss 0.0427, Acc 0.919, BalAcc 0.893, F1 0.945, AUC 0.947, Prec 0.941, Rec 0.949, Thresh 0.520 | LR: 0.000100
Epoch 09: Train: Loss 0.0659, Acc 0.702 | Val: Loss 0.0448, Acc 0.935, BalAcc 0.937, F1 0.954, AUC 0.976, Prec 0.978, Rec 0.932, Thresh 0.553 | LR: 0.000100
✅ New best validation balanced accuracy: 0.937, threshold: 0.553
📊 Mag Importance (Val BalAcc: 0.937): {'40': 0.24165110290050507, '100': 0.28181222081184387, '200': 0.23601968586444855, '400': 0.2405170202255249}
Epoch 10: Train: Loss 0.0682, Acc 0.719 | Val: Loss 0.0430, Acc 0.898, BalAcc 0.841, F1 0.932, AUC 0.933, Prec 0.904, Rec 0.962, Thresh 0.561 | LR: 0.000100
Epoch 11: Train: Loss 0.0651, Acc 0.695 | Val: Loss 0.0353, Acc 0.941, BalAcc 0.934, F1 0.959, AUC 0.975, Prec 0.970, Rec 0.949, Thresh 0.569 | LR: 0.000100
Epoch 12: Train: Loss 0.0656, Acc 0.716 | Val: Loss 0.0423, Acc 0.950, BalAcc 0.948, F1 0.966, AUC 0.973, Prec 0.978, Rec 0.953, Thresh 0.520 | LR: 0.000100
✅ New best validation balanced accuracy: 0.948, threshold: 0.520
📊 Mag Importance (Val BalAcc: 0.948): {'40': 0.24188664555549622, '100': 0.2804350256919861, '200': 0.23657581210136414, '400': 0.24110254645347595}
Epoch 13: Train: Loss 0.0655, Acc 0.713 | Val: Loss 0.0412, Acc 0.947, BalAcc 0.942, F1 0.964, AUC 0.976, Prec 0.974, Rec 0.953, Thresh 0.504 | LR: 0.000100
Epoch 14: Train: Loss 0.0632, Acc 0.694 | Val: Loss 0.0387, Acc 0.935, BalAcc 0.926, F1 0.955, AUC 0.973, Prec 0.965, Rec 0.945, Thresh 0.488 | LR: 0.000100
Epoch 15: Train: Loss 0.0611, Acc 0.721 | Val: Loss 0.0382, Acc 0.950, BalAcc 0.948, F1 0.966, AUC 0.979, Prec 0.978, Rec 0.953, Thresh 0.480 | LR: 0.000100
Epoch 16: Train: Loss 0.0589, Acc 0.709 | Val: Loss 0.0355, Acc 0.941, BalAcc 0.934, F1 0.959, AUC 0.975, Prec 0.970, Rec 0.949, Thresh 0.512 | LR: 0.000050
Epoch 17: Train: Loss 0.0583, Acc 0.721 | Val: Loss 0.0386, Acc 0.916, BalAcc 0.910, F1 0.942, AUC 0.965, Prec 0.960, Rec 0.924, Thresh 0.528 | LR: 0.000050
Epoch 18: Train: Loss 0.0604, Acc 0.718 | Val: Loss 0.0376, Acc 0.938, BalAcc 0.950, F1 0.956, AUC 0.977, Prec 0.991, Rec 0.924, Thresh 0.496 | LR: 0.000050
✅ New best validation balanced accuracy: 0.950, threshold: 0.496
📊 Mag Importance (Val BalAcc: 0.950): {'40': 0.24162939190864563, '100': 0.279721736907959, '200': 0.2393660545349121, '400': 0.23928287625312805}
Epoch 19: Train: Loss 0.0514, Acc 0.715 | Val: Loss 0.0422, Acc 0.932, BalAcc 0.942, F1 0.952, AUC 0.974, Prec 0.986, Rec 0.919, Thresh 0.488 | LR: 0.000050
Epoch 20: Train: Loss 0.0576, Acc 0.728 | Val: Loss 0.0372, Acc 0.938, BalAcc 0.947, F1 0.956, AUC 0.977, Prec 0.986, Rec 0.928, Thresh 0.512 | LR: 0.000050
Epoch 21: Train: Loss 0.0538, Acc 0.726 | Val: Loss 0.0437, Acc 0.916, BalAcc 0.880, F1 0.944, AUC 0.972, Prec 0.930, Rec 0.958, Thresh 0.439 | LR: 0.000050
Epoch 22: Train: Loss 0.0595, Acc 0.727 | Val: Loss 0.0419, Acc 0.932, BalAcc 0.924, F1 0.953, AUC 0.971, Prec 0.965, Rec 0.941, Thresh 0.447 | LR: 0.000025
Epoch 23: Train: Loss 0.0553, Acc 0.728 | Val: Loss 0.0384, Acc 0.932, BalAcc 0.939, F1 0.952, AUC 0.974, Prec 0.982, Rec 0.924, Thresh 0.504 | LR: 0.000025
Epoch 24: Train: Loss 0.0616, Acc 0.731 | Val: Loss 0.0402, Acc 0.916, BalAcc 0.910, F1 0.942, AUC 0.965, Prec 0.960, Rec 0.924, Thresh 0.504 | LR: 0.000025
Epoch 25: Train: Loss 0.0556, Acc 0.718 | Val: Loss 0.0451, Acc 0.925, BalAcc 0.938, F1 0.947, AUC 0.961, Prec 0.986, Rec 0.911, Thresh 0.512 | LR: 0.000025
⚠️ Early stopping after 25 epochs (no improvement for 7 epochs)
✅ Best model saved: ./output/models/best_model_fold_1.pth (Val BalAcc: 0.950)
⚡️ Test Results: Acc 0.864, BalAcc 0.847, F1 0.898, AUC 0.896, Precision 0.900, Recall 0.896 (threshold: 0.496)
📊 Confusion Matrix (Fold 1):
   [[TN:  95, FP:  24]
    [FN:  25, TP: 215]]
⚡ Avg Inference Time: 0.0099s per sample
📌 Final Magnification Importance (Fold 1): {'40': 0.2346128672361374, '100': 0.28646188974380493, '200': 0.24589788913726807, '400': 0.2330273687839508}
💾 Results saved to: ./output/results/fold_1_results.json

📊 Generating GradCAM visualizations for fold 1...
✅ Generated 5 GradCAM visualizations for fold 1

===== Fold 2 =====
Train patients: 49, Val Patients: 17, Test patients: 16
Training samples per epoch: {'total_samples_per_epoch': 2031, 'class_distribution': {0: 606, 1: 1425}, 'oversampling_factor': 2.0}
Validation samples: 340, Test samples: 293
Patients with full 4 mags: 49
Inner training samples: 2031, batch size: 16
📊 Class weights (balanced): Benign=1.75, Malignant=0.70
   Weight ratio: 2.5x (was 16.5x)
Epoch 01: Train: Loss 0.0869, Acc 0.656 | Val: Loss 0.0592, Acc 0.824, BalAcc 0.659, F1 0.894, AUC 0.809, Prec 0.826, Rec 0.973, Thresh 0.447 | LR: 0.000100
✅ New best validation balanced accuracy: 0.659, threshold: 0.447
📊 Mag Importance (Val BalAcc: 0.659): {'40': 0.2517901360988617, '100': 0.2527613341808319, '200': 0.23177233338356018, '400': 0.2636761963367462}
Epoch 02: Train: Loss 0.0814, Acc 0.675 | Val: Loss 0.0547, Acc 0.841, BalAcc 0.696, F1 0.903, AUC 0.869, Prec 0.843, Rec 0.973, Thresh 0.391 | LR: 0.000100
✅ New best validation balanced accuracy: 0.696, threshold: 0.391
📊 Mag Importance (Val BalAcc: 0.696): {'40': 0.25403329730033875, '100': 0.25138869881629944, '200': 0.22888240218162537, '400': 0.26569557189941406}
Epoch 03: Train: Loss 0.0730, Acc 0.686 | Val: Loss 0.0403, Acc 0.868, BalAcc 0.756, F1 0.918, AUC 0.921, Prec 0.872, Rec 0.969, Thresh 0.456 | LR: 0.000100
✅ New best validation balanced accuracy: 0.756, threshold: 0.456
📊 Mag Importance (Val BalAcc: 0.756): {'40': 0.25699156522750854, '100': 0.2460688352584839, '200': 0.23015284538269043, '400': 0.26678675413131714}
Epoch 04: Train: Loss 0.0699, Acc 0.716 | Val: Loss 0.0430, Acc 0.912, BalAcc 0.874, F1 0.942, AUC 0.942, Prec 0.939, Rec 0.946, Thresh 0.480 | LR: 0.000100
✅ New best validation balanced accuracy: 0.874, threshold: 0.480
📊 Mag Importance (Val BalAcc: 0.874): {'40': 0.25376027822494507, '100': 0.24827012419700623, '200': 0.23087537288665771, '400': 0.2670941948890686}
Epoch 05: Train: Loss 0.0710, Acc 0.706 | Val: Loss 0.0374, Acc 0.921, BalAcc 0.867, F1 0.949, AUC 0.953, Prec 0.930, Rec 0.969, Thresh 0.480 | LR: 0.000100
Epoch 06: Train: Loss 0.0661, Acc 0.726 | Val: Loss 0.0375, Acc 0.909, BalAcc 0.843, F1 0.942, AUC 0.940, Prec 0.916, Rec 0.969, Thresh 0.464 | LR: 0.000100
Epoch 07: Train: Loss 0.0656, Acc 0.724 | Val: Loss 0.0334, Acc 0.929, BalAcc 0.860, F1 0.955, AUC 0.964, Prec 0.921, Rec 0.992, Thresh 0.472 | LR: 0.000100
Epoch 08: Train: Loss 0.0601, Acc 0.740 | Val: Loss 0.0295, Acc 0.959, BalAcc 0.939, F1 0.973, AUC 0.983, Prec 0.969, Rec 0.977, Thresh 0.528 | LR: 0.000100
✅ New best validation balanced accuracy: 0.939, threshold: 0.528
📊 Mag Importance (Val BalAcc: 0.939): {'40': 0.258442223072052, '100': 0.24445576965808868, '200': 0.22796306014060974, '400': 0.26913899183273315}
Epoch 09: Train: Loss 0.0595, Acc 0.741 | Val: Loss 0.0323, Acc 0.947, BalAcc 0.889, F1 0.966, AUC 0.976, Prec 0.935, Rec 1.000, Thresh 0.512 | LR: 0.000100
Epoch 10: Train: Loss 0.0665, Acc 0.726 | Val: Loss 0.0318, Acc 0.962, BalAcc 0.932, F1 0.975, AUC 0.988, Prec 0.962, Rec 0.988, Thresh 0.488 | LR: 0.000100
Epoch 11: Train: Loss 0.0574, Acc 0.749 | Val: Loss 0.0298, Acc 0.971, BalAcc 0.943, F1 0.981, AUC 0.978, Prec 0.966, Rec 0.996, Thresh 0.512 | LR: 0.000100
✅ New best validation balanced accuracy: 0.943, threshold: 0.512
📊 Mag Importance (Val BalAcc: 0.943): {'40': 0.25927263498306274, '100': 0.24557192623615265, '200': 0.22727032005786896, '400': 0.26788514852523804}
Epoch 12: Train: Loss 0.0555, Acc 0.733 | Val: Loss 0.0289, Acc 0.959, BalAcc 0.931, F1 0.973, AUC 0.979, Prec 0.962, Rec 0.985, Thresh 0.512 | LR: 0.000100
Epoch 13: Train: Loss 0.0573, Acc 0.765 | Val: Loss 0.0233, Acc 0.976, BalAcc 0.951, F1 0.985, AUC 0.991, Prec 0.970, Rec 1.000, Thresh 0.488 | LR: 0.000100
✅ New best validation balanced accuracy: 0.951, threshold: 0.488
📊 Mag Importance (Val BalAcc: 0.951): {'40': 0.25486040115356445, '100': 0.2504126727581024, '200': 0.22495433688163757, '400': 0.26977258920669556}
Epoch 14: Train: Loss 0.0607, Acc 0.741 | Val: Loss 0.0302, Acc 0.965, BalAcc 0.930, F1 0.977, AUC 0.977, Prec 0.959, Rec 0.996, Thresh 0.536 | LR: 0.000100
Epoch 15: Train: Loss 0.0558, Acc 0.720 | Val: Loss 0.0296, Acc 0.956, BalAcc 0.924, F1 0.971, AUC 0.983, Prec 0.959, Rec 0.985, Thresh 0.528 | LR: 0.000100
Epoch 16: Train: Loss 0.0603, Acc 0.752 | Val: Loss 0.0276, Acc 0.979, BalAcc 0.970, F1 0.987, AUC 0.997, Prec 0.985, Rec 0.988, Thresh 0.504 | LR: 0.000100
✅ New best validation balanced accuracy: 0.970, threshold: 0.504
📊 Mag Importance (Val BalAcc: 0.970): {'40': 0.25556468963623047, '100': 0.2519229054450989, '200': 0.22332623600959778, '400': 0.2691861689090729}
Epoch 17: Train: Loss 0.0564, Acc 0.752 | Val: Loss 0.0268, Acc 0.988, BalAcc 0.980, F1 0.992, AUC 0.998, Prec 0.989, Rec 0.996, Thresh 0.504 | LR: 0.000100
✅ New best validation balanced accuracy: 0.980, threshold: 0.504
📊 Mag Importance (Val BalAcc: 0.980): {'40': 0.2555437982082367, '100': 0.25107279419898987, '200': 0.22514665126800537, '400': 0.26823675632476807}
Epoch 18: Train: Loss 0.0620, Acc 0.749 | Val: Loss 0.0283, Acc 0.971, BalAcc 0.951, F1 0.981, AUC 0.996, Prec 0.973, Rec 0.988, Thresh 0.472 | LR: 0.000100
Epoch 19: Train: Loss 0.0539, Acc 0.775 | Val: Loss 0.0266, Acc 0.974, BalAcc 0.957, F1 0.983, AUC 0.995, Prec 0.977, Rec 0.988, Thresh 0.544 | LR: 0.000100
Epoch 20: Train: Loss 0.0527, Acc 0.775 | Val: Loss 0.0304, Acc 0.976, BalAcc 0.959, F1 0.985, AUC 0.993, Prec 0.977, Rec 0.992, Thresh 0.391 | LR: 0.000100
Epoch 21: Train: Loss 0.0565, Acc 0.739 | Val: Loss 0.0296, Acc 0.971, BalAcc 0.943, F1 0.981, AUC 0.990, Prec 0.966, Rec 0.996, Thresh 0.512 | LR: 0.000050
Epoch 22: Train: Loss 0.0538, Acc 0.793 | Val: Loss 0.0288, Acc 0.971, BalAcc 0.943, F1 0.981, AUC 0.990, Prec 0.966, Rec 0.996, Thresh 0.504 | LR: 0.000050
Epoch 23: Train: Loss 0.0533, Acc 0.753 | Val: Loss 0.0279, Acc 0.968, BalAcc 0.936, F1 0.979, AUC 0.993, Prec 0.963, Rec 0.996, Thresh 0.512 | LR: 0.000050
Epoch 24: Train: Loss 0.0553, Acc 0.754 | Val: Loss 0.0311, Acc 0.953, BalAcc 0.914, F1 0.970, AUC 0.974, Prec 0.952, Rec 0.988, Thresh 0.512 | LR: 0.000050
⚠️ Early stopping after 24 epochs (no improvement for 7 epochs)
✅ Best model saved: ./output/models/best_model_fold_2.pth (Val BalAcc: 0.980)
⚡️ Test Results: Acc 0.863, BalAcc 0.825, F1 0.903, AUC 0.910, Precision 0.877, Recall 0.930 (threshold: 0.504)
📊 Confusion Matrix (Fold 2):
   [[TN:  67, FP:  26]
    [FN:  14, TP: 186]]
⚡ Avg Inference Time: 0.0112s per sample
📌 Final Magnification Importance (Fold 2): {'40': 0.2613515555858612, '100': 0.2542078495025635, '200': 0.22650304436683655, '400': 0.257937490940094}
💾 Results saved to: ./output/results/fold_2_results.json

📊 Generating GradCAM visualizations for fold 2...
✅ Generated 5 GradCAM visualizations for fold 2

===== Fold 3 =====
Train patients: 49, Val Patients: 17, Test patients: 16
Training samples per epoch: {'total_samples_per_epoch': 2085, 'class_distribution': {0: 603, 1: 1482}, 'oversampling_factor': 2.1}
Validation samples: 301, Test samples: 354
Patients with full 4 mags: 49
Inner training samples: 2085, batch size: 16
📊 Class weights (balanced): Benign=1.75, Malignant=0.70
   Weight ratio: 2.5x (was 16.5x)