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
Class weights: Benign=1.73, Malignant=0.70
Epoch 01: Train: Loss 0.0984, Acc 0.573 | Val: Loss 0.0868, Acc 0.722, BalAcc 0.592, F1 0.822, AUC 0.649, Prec 0.738, Rec 0.928, Thresh 0.312 | LR: 0.000100
✅ New best validation balanced accuracy: 0.592, threshold: 0.312
📊 Mag Importance (Val BalAcc: 0.592): {'40': 0.726329505443573, '100': 0.048996713012456894, '200': 0.04003071039915085, '400': 0.18464311957359314}
Epoch 02: Train: Loss 0.0660, Acc 0.686 | Val: Loss 0.0687, Acc 0.726, BalAcc 0.584, F1 0.828, AUC 0.701, Prec 0.733, Rec 0.952, Thresh 0.370 | LR: 0.000100
Epoch 03: Train: Loss 0.0603, Acc 0.713 | Val: Loss 0.0416, Acc 0.817, BalAcc 0.748, F1 0.876, AUC 0.819, Prec 0.829, Rec 0.928, Thresh 0.398 | LR: 0.000100
✅ New best validation balanced accuracy: 0.748, threshold: 0.398
📊 Mag Importance (Val BalAcc: 0.748): {'40': 0.7270306944847107, '100': 0.09832198917865753, '200': 0.0445745550096035, '400': 0.13007274270057678}
Epoch 04: Train: Loss 0.0456, Acc 0.712 | Val: Loss 0.0443, Acc 0.788, BalAcc 0.663, F1 0.866, AUC 0.797, Prec 0.771, Rec 0.988, Thresh 0.328 | LR: 0.000100
Epoch 05: Train: Loss 0.0406, Acc 0.756 | Val: Loss 0.0232, Acc 0.855, BalAcc 0.782, F1 0.903, AUC 0.931, Prec 0.844, Rec 0.970, Thresh 0.366 | LR: 0.000100
✅ New best validation balanced accuracy: 0.782, threshold: 0.366
📊 Mag Importance (Val BalAcc: 0.782): {'40': 0.6115072965621948, '100': 0.15708883106708527, '200': 0.0483182817697525, '400': 0.1830856055021286}
Epoch 06: Train: Loss 0.0397, Acc 0.756 | Val: Loss 0.0424, Acc 0.851, BalAcc 0.798, F1 0.897, AUC 0.875, Prec 0.862, Rec 0.934, Thresh 0.521 | LR: 0.000100
✅ New best validation balanced accuracy: 0.798, threshold: 0.521
📊 Mag Importance (Val BalAcc: 0.798): {'40': 0.5480503439903259, '100': 0.1905573010444641, '200': 0.07169603556394577, '400': 0.1896963119506836}
Epoch 07: Train: Loss 0.0313, Acc 0.760 | Val: Loss 0.0155, Acc 0.917, BalAcc 0.884, F1 0.942, AUC 0.958, Prec 0.915, Rec 0.970, Thresh 0.518 | LR: 0.000100
✅ New best validation balanced accuracy: 0.884, threshold: 0.518
📊 Mag Importance (Val BalAcc: 0.884): {'40': 0.5916043519973755, '100': 0.14619626104831696, '200': 0.06084340438246727, '400': 0.2013559639453888}
Epoch 08: Train: Loss 0.0302, Acc 0.774 | Val: Loss 0.0205, Acc 0.917, BalAcc 0.895, F1 0.941, AUC 0.944, Prec 0.930, Rec 0.952, Thresh 0.546 | LR: 0.000100
✅ New best validation balanced accuracy: 0.895, threshold: 0.546
📊 Mag Importance (Val BalAcc: 0.895): {'40': 0.4307594895362854, '100': 0.1744472235441208, '200': 0.10828790068626404, '400': 0.28650540113449097}
Epoch 09: Train: Loss 0.0305, Acc 0.769 | Val: Loss 0.1124, Acc 0.900, BalAcc 0.864, F1 0.930, AUC 0.849, Prec 0.904, Rec 0.958, Thresh 0.539 | LR: 0.000100
Epoch 10: Train: Loss 0.0244, Acc 0.777 | Val: Loss 0.0154, Acc 0.925, BalAcc 0.882, F1 0.949, AUC 0.966, Prec 0.907, Rec 0.994, Thresh 0.468 | LR: 0.000100
Epoch 11: Train: Loss 0.0316, Acc 0.778 | Val: Loss 0.0423, Acc 0.846, BalAcc 0.773, F1 0.897, AUC 0.867, Prec 0.839, Rec 0.964, Thresh 0.384 | LR: 0.000100
Epoch 12: Train: Loss 0.0366, Acc 0.770 | Val: Loss 0.0179, Acc 0.934, BalAcc 0.945, F1 0.950, AUC 0.977, Prec 0.987, Rec 0.916, Thresh 0.473 | LR: 0.000100
✅ New best validation balanced accuracy: 0.945, threshold: 0.473
📊 Mag Importance (Val BalAcc: 0.945): {'40': 0.3625689744949341, '100': 0.11968618631362915, '200': 0.2970033288002014, '400': 0.22074149549007416}
Epoch 13: Train: Loss 0.0337, Acc 0.759 | Val: Loss 0.0233, Acc 0.884, BalAcc 0.856, F1 0.917, AUC 0.918, Prec 0.906, Rec 0.928, Thresh 0.570 | LR: 0.000100
Epoch 14: Train: Loss 0.0373, Acc 0.767 | Val: Loss 0.0197, Acc 0.913, BalAcc 0.899, F1 0.937, AUC 0.948, Prec 0.940, Rec 0.934, Thresh 0.527 | LR: 0.000100
Epoch 15: Train: Loss 0.0351, Acc 0.760 | Val: Loss 0.0231, Acc 0.863, BalAcc 0.785, F1 0.909, AUC 0.924, Prec 0.842, Rec 0.988, Thresh 0.451 | LR: 0.000100
Epoch 16: Train: Loss 0.0313, Acc 0.803 | Val: Loss 0.0197, Acc 0.896, BalAcc 0.899, F1 0.923, AUC 0.943, Prec 0.955, Rec 0.892, Thresh 0.621 | LR: 0.000050
Epoch 17: Train: Loss 0.0299, Acc 0.778 | Val: Loss 0.0130, Acc 0.934, BalAcc 0.937, F1 0.951, AUC 0.976, Prec 0.975, Rec 0.928, Thresh 0.560 | LR: 0.000050
Epoch 18: Train: Loss 0.0256, Acc 0.790 | Val: Loss 0.0165, Acc 0.938, BalAcc 0.925, F1 0.955, AUC 0.964, Prec 0.952, Rec 0.958, Thresh 0.455 | LR: 0.000050
Epoch 19: Train: Loss 0.0236, Acc 0.816 | Val: Loss 0.0209, Acc 0.905, BalAcc 0.909, F1 0.929, AUC 0.957, Prec 0.962, Rec 0.898, Thresh 0.700 | LR: 0.000050
⚠️ Early stopping after 19 epochs (no improvement for 7 epochs)
✅ Best model saved: ./output/models/best_model_fold_0.pth (Val BalAcc: 0.945)
⚡️ Test Results: Acc 0.960, BalAcc 0.967, F1 0.967, AUC 0.999, Precision 0.995, Recall 0.941 (threshold: 0.473)
📊 Confusion Matrix (Fold 0):
   [[TN: 131, FP:   1]
    [FN:  13, TP: 208]]
⚡ Avg Inference Time: 0.0161s per sample
📌 Final Magnification Importance (Fold 0): {'40': 0.35181793570518494, '100': 0.13461515307426453, '200': 0.2945305407047272, '400': 0.21903635561466217}
💾 Results saved to: ./output/results/fold_0_results.json

📊 Generating GradCAM visualizations for fold 0...
✅ Generated 5 GradCAM visualizations for fold 0

===== Fold 1 =====
Train patients: 52, Val Patients: 13, Test patients: 17
Training samples per epoch: {'total_samples_per_epoch': 2169, 'class_distribution': {0: 639, 1: 1530}, 'oversampling_factor': 2.05}
Validation samples: 229, Test samples: 359
Patients with full 4 mags: 52
Inner training samples: 2169, batch size: 16
Class weights: Benign=1.73, Malignant=0.70
Epoch 01: Train: Loss 0.0971, Acc 0.610 | Val: Loss 0.0809, Acc 0.694, BalAcc 0.623, F1 0.785, AUC 0.660, Prec 0.757, Rec 0.815, Thresh 0.500 | LR: 0.000100
✅ New best validation balanced accuracy: 0.623, threshold: 0.500
📊 Mag Importance (Val BalAcc: 0.623): {'40': 0.3806096613407135, '100': 0.1638025939464569, '200': 0.345795214176178, '400': 0.10979249328374863}
Epoch 02: Train: Loss 0.0749, Acc 0.638 | Val: Loss 0.0412, Acc 0.742, BalAcc 0.613, F1 0.837, AUC 0.788, Prec 0.740, Rec 0.962, Thresh 0.363 | LR: 0.000100
Epoch 03: Train: Loss 0.0536, Acc 0.697 | Val: Loss 0.0459, Acc 0.773, BalAcc 0.688, F1 0.847, AUC 0.785, Prec 0.787, Rec 0.917, Thresh 0.458 | LR: 0.000100
✅ New best validation balanced accuracy: 0.688, threshold: 0.458
📊 Mag Importance (Val BalAcc: 0.688): {'40': 0.2771008312702179, '100': 0.17914924025535583, '200': 0.3857940137386322, '400': 0.15795595943927765}
Epoch 04: Train: Loss 0.0456, Acc 0.709 | Val: Loss 0.0276, Acc 0.847, BalAcc 0.776, F1 0.897, AUC 0.908, Prec 0.835, Rec 0.968, Thresh 0.413 | LR: 0.000100
✅ New best validation balanced accuracy: 0.776, threshold: 0.413
📊 Mag Importance (Val BalAcc: 0.776): {'40': 0.3063162565231323, '100': 0.18034802377223969, '200': 0.33170393109321594, '400': 0.18163177371025085}
Epoch 05: Train: Loss 0.0484, Acc 0.719 | Val: Loss 0.0322, Acc 0.843, BalAcc 0.773, F1 0.893, AUC 0.882, Prec 0.834, Rec 0.962, Thresh 0.399 | LR: 0.000100
Epoch 06: Train: Loss 0.0534, Acc 0.715 | Val: Loss 0.0770, Acc 0.869, BalAcc 0.856, F1 0.903, AUC 0.911, Prec 0.915, Rec 0.892, Thresh 0.526 | LR: 0.000100
✅ New best validation balanced accuracy: 0.856, threshold: 0.526
📊 Mag Importance (Val BalAcc: 0.856): {'40': 0.37052279710769653, '100': 0.13118800520896912, '200': 0.21974927186965942, '400': 0.27853989601135254}
Epoch 07: Train: Loss 0.0395, Acc 0.761 | Val: Loss 0.0158, Acc 0.939, BalAcc 0.937, F1 0.955, AUC 0.971, Prec 0.967, Rec 0.943, Thresh 0.501 | LR: 0.000100
✅ New best validation balanced accuracy: 0.937, threshold: 0.501
📊 Mag Importance (Val BalAcc: 0.937): {'40': 0.2371399700641632, '100': 0.13035158812999725, '200': 0.2667170464992523, '400': 0.365791380405426}
Epoch 08: Train: Loss 0.0372, Acc 0.762 | Val: Loss 0.0148, Acc 0.961, BalAcc 0.964, F1 0.971, AUC 0.979, Prec 0.987, Rec 0.955, Thresh 0.518 | LR: 0.000100
✅ New best validation balanced accuracy: 0.964, threshold: 0.518
📊 Mag Importance (Val BalAcc: 0.964): {'40': 0.25812286138534546, '100': 0.17622239887714386, '200': 0.2801312804222107, '400': 0.2855234742164612}
Epoch 09: Train: Loss 0.0320, Acc 0.788 | Val: Loss 0.0149, Acc 0.943, BalAcc 0.959, F1 0.957, AUC 0.973, Prec 1.000, Rec 0.917, Thresh 0.565 | LR: 0.000100
Epoch 10: Train: Loss 0.0299, Acc 0.757 | Val: Loss 0.0157, Acc 0.939, BalAcc 0.937, F1 0.955, AUC 0.969, Prec 0.967, Rec 0.943, Thresh 0.455 | LR: 0.000100
Epoch 11: Train: Loss 0.0301, Acc 0.774 | Val: Loss 0.0135, Acc 0.956, BalAcc 0.961, F1 0.968, AUC 0.974, Prec 0.987, Rec 0.949, Thresh 0.476 | LR: 0.000100
Epoch 12: Train: Loss 0.0346, Acc 0.775 | Val: Loss 0.0128, Acc 0.952, BalAcc 0.942, F1 0.965, AUC 0.978, Prec 0.962, Rec 0.968, Thresh 0.496 | LR: 0.000050
Epoch 13: Train: Loss 0.0265, Acc 0.801 | Val: Loss 0.0139, Acc 0.965, BalAcc 0.971, F1 0.974, AUC 0.980, Prec 0.993, Rec 0.955, Thresh 0.482 | LR: 0.000050
✅ New best validation balanced accuracy: 0.971, threshold: 0.482
📊 Mag Importance (Val BalAcc: 0.971): {'40': 0.1975025236606598, '100': 0.17017699778079987, '200': 0.2653329074382782, '400': 0.36698755621910095}
Epoch 14: Train: Loss 0.0252, Acc 0.817 | Val: Loss 0.0131, Acc 0.956, BalAcc 0.957, F1 0.968, AUC 0.986, Prec 0.980, Rec 0.955, Thresh 0.426 | LR: 0.000050
Epoch 15: Train: Loss 0.0236, Acc 0.802 | Val: Loss 0.0128, Acc 0.952, BalAcc 0.954, F1 0.964, AUC 0.979, Prec 0.980, Rec 0.949, Thresh 0.515 | LR: 0.000050
Epoch 16: Train: Loss 0.0236, Acc 0.792 | Val: Loss 0.0131, Acc 0.961, BalAcc 0.953, F1 0.971, AUC 0.989, Prec 0.968, Rec 0.975, Thresh 0.433 | LR: 0.000050
Epoch 17: Train: Loss 0.0236, Acc 0.753 | Val: Loss 0.0139, Acc 0.969, BalAcc 0.978, F1 0.977, AUC 0.987, Prec 1.000, Rec 0.955, Thresh 0.415 | LR: 0.000050
✅ New best validation balanced accuracy: 0.978, threshold: 0.415
📊 Mag Importance (Val BalAcc: 0.978): {'40': 0.21759995818138123, '100': 0.1439559906721115, '200': 0.2633320689201355, '400': 0.37511199712753296}
Epoch 18: Train: Loss 0.0284, Acc 0.786 | Val: Loss 0.0147, Acc 0.956, BalAcc 0.953, F1 0.968, AUC 0.979, Prec 0.974, Rec 0.962, Thresh 0.405 | LR: 0.000050
Epoch 19: Train: Loss 0.0256, Acc 0.769 | Val: Loss 0.0141, Acc 0.961, BalAcc 0.964, F1 0.971, AUC 0.980, Prec 0.987, Rec 0.955, Thresh 0.421 | LR: 0.000050
Epoch 20: Train: Loss 0.0228, Acc 0.766 | Val: Loss 0.0114, Acc 0.965, BalAcc 0.963, F1 0.974, AUC 0.988, Prec 0.981, Rec 0.968, Thresh 0.428 | LR: 0.000050
Epoch 21: Train: Loss 0.0214, Acc 0.801 | Val: Loss 0.0125, Acc 0.961, BalAcc 0.960, F1 0.971, AUC 0.985, Prec 0.981, Rec 0.962, Thresh 0.409 | LR: 0.000025
Epoch 22: Train: Loss 0.0232, Acc 0.800 | Val: Loss 0.0100, Acc 0.965, BalAcc 0.959, F1 0.975, AUC 0.991, Prec 0.975, Rec 0.975, Thresh 0.437 | LR: 0.000025
Epoch 23: Train: Loss 0.0222, Acc 0.799 | Val: Loss 0.0086, Acc 0.969, BalAcc 0.978, F1 0.977, AUC 0.991, Prec 1.000, Rec 0.955, Thresh 0.521 | LR: 0.000025
Epoch 24: Train: Loss 0.0213, Acc 0.799 | Val: Loss 0.0084, Acc 0.969, BalAcc 0.974, F1 0.977, AUC 0.994, Prec 0.993, Rec 0.962, Thresh 0.480 | LR: 0.000025
⚠️ Early stopping after 24 epochs (no improvement for 7 epochs)
✅ Best model saved: ./output/models/best_model_fold_1.pth (Val BalAcc: 0.978)
⚡️ Test Results: Acc 0.933, BalAcc 0.927, F1 0.950, AUC 0.987, Precision 0.954, Recall 0.946 (threshold: 0.415)
📊 Confusion Matrix (Fold 1):
   [[TN: 108, FP:  11]
    [FN:  13, TP: 227]]
⚡ Avg Inference Time: 0.0177s per sample
📌 Final Magnification Importance (Fold 1): {'40': 0.17206120491027832, '100': 0.16149601340293884, '200': 0.2586040794849396, '400': 0.40783873200416565}
💾 Results saved to: ./output/results/fold_1_results.json

📊 Generating GradCAM visualizations for fold 1...
✅ Generated 5 GradCAM visualizations for fold 1

===== Fold 2 =====
Train patients: 52, Val Patients: 14, Test patients: 16
Training samples per epoch: {'total_samples_per_epoch': 2160, 'class_distribution': {1: 1515, 0: 645}, 'oversampling_factor': 1.98}
Validation samples: 262, Test samples: 293
Patients with full 4 mags: 52
Inner training samples: 2160, batch size: 16
Class weights: Benign=1.73, Malignant=0.70
Epoch 01: Train: Loss 0.1080, Acc 0.563 | Val: Loss 0.0792, Acc 0.729, BalAcc 0.651, F1 0.817, AUC 0.717, Prec 0.823, Rec 0.810, Thresh 0.500 | LR: 0.000100
✅ New best validation balanced accuracy: 0.651, threshold: 0.500
📊 Mag Importance (Val BalAcc: 0.651): {'40': 0.3121504485607147, '100': 0.280524879693985, '200': 0.20139408111572266, '400': 0.20593060553073883}
Epoch 02: Train: Loss 0.0777, Acc 0.631 | Val: Loss 0.0309, Acc 0.863, BalAcc 0.795, F1 0.910, AUC 0.902, Prec 0.888, Rec 0.933, Thresh 0.409 | LR: 0.000100
✅ New best validation balanced accuracy: 0.795, threshold: 0.409
📊 Mag Importance (Val BalAcc: 0.795): {'40': 0.24458609521389008, '100': 0.23070961236953735, '200': 0.281056672334671, '400': 0.24364759027957916}
Epoch 03: Train: Loss 0.0576, Acc 0.672 | Val: Loss 0.0293, Acc 0.847, BalAcc 0.746, F1 0.903, AUC 0.906, Prec 0.857, Rec 0.954, Thresh 0.429 | LR: 0.000100
Epoch 04: Train: Loss 0.0443, Acc 0.718 | Val: Loss 0.0142, Acc 0.916, BalAcc 0.841, F1 0.946, AUC 0.967, Prec 0.902, Rec 0.995, Thresh 0.406 | LR: 0.000100
✅ New best validation balanced accuracy: 0.841, threshold: 0.406
📊 Mag Importance (Val BalAcc: 0.841): {'40': 0.18454229831695557, '100': 0.4436352252960205, '200': 0.22908581793308258, '400': 0.14273665845394135}
Epoch 05: Train: Loss 0.0382, Acc 0.751 | Val: Loss 0.0115, Acc 0.943, BalAcc 0.898, F1 0.963, AUC 0.985, Prec 0.937, Rec 0.990, Thresh 0.395 | LR: 0.000100
✅ New best validation balanced accuracy: 0.898, threshold: 0.395
📊 Mag Importance (Val BalAcc: 0.898): {'40': 0.10832701623439789, '100': 0.43843165040016174, '200': 0.23802010715007782, '400': 0.21522124111652374}
Epoch 06: Train: Loss 0.0422, Acc 0.725 | Val: Loss 0.0097, Acc 0.958, BalAcc 0.933, F1 0.972, AUC 0.990, Prec 0.960, Rec 0.985, Thresh 0.460 | LR: 0.000100
✅ New best validation balanced accuracy: 0.933, threshold: 0.460
📊 Mag Importance (Val BalAcc: 0.933): {'40': 0.10923625528812408, '100': 0.31962668895721436, '200': 0.31458017230033875, '400': 0.256556898355484}
Epoch 07: Train: Loss 0.0370, Acc 0.740 | Val: Loss 0.0135, Acc 0.943, BalAcc 0.893, F1 0.963, AUC 0.981, Prec 0.933, Rec 0.995, Thresh 0.463 | LR: 0.000100
Epoch 08: Train: Loss 0.0408, Acc 0.738 | Val: Loss 0.0093, Acc 0.973, BalAcc 0.962, F1 0.982, AUC 0.996, Prec 0.980, Rec 0.985, Thresh 0.534 | LR: 0.000100
✅ New best validation balanced accuracy: 0.962, threshold: 0.534
📊 Mag Importance (Val BalAcc: 0.962): {'40': 0.12296377867460251, '100': 0.3152485191822052, '200': 0.2595515847206116, '400': 0.3022361099720001}
Epoch 09: Train: Loss 0.0364, Acc 0.738 | Val: Loss 0.0089, Acc 0.985, BalAcc 0.980, F1 0.990, AUC 0.995, Prec 0.990, Rec 0.990, Thresh 0.513 | LR: 0.000100
✅ New best validation balanced accuracy: 0.980, threshold: 0.513
📊 Mag Importance (Val BalAcc: 0.980): {'40': 0.1935553401708603, '100': 0.34315934777259827, '200': 0.2539164125919342, '400': 0.20936891436576843}
Epoch 10: Train: Loss 0.0384, Acc 0.753 | Val: Loss 0.0150, Acc 0.969, BalAcc 0.945, F1 0.980, AUC 0.985, Prec 0.965, Rec 0.995, Thresh 0.591 | LR: 0.000100
Epoch 11: Train: Loss 0.0391, Acc 0.770 | Val: Loss 0.0082, Acc 0.992, BalAcc 0.990, F1 0.995, AUC 0.998, Prec 0.995, Rec 0.995, Thresh 0.574 | LR: 0.000100
✅ New best validation balanced accuracy: 0.990, threshold: 0.574
📊 Mag Importance (Val BalAcc: 0.990): {'40': 0.3084982633590698, '100': 0.1681634932756424, '200': 0.24046333134174347, '400': 0.28287485241889954}
Epoch 12: Train: Loss 0.0349, Acc 0.754 | Val: Loss 0.0054, Acc 0.992, BalAcc 0.995, F1 0.995, AUC 0.999, Prec 1.000, Rec 0.990, Thresh 0.490 | LR: 0.000100
✅ New best validation balanced accuracy: 0.995, threshold: 0.490
📊 Mag Importance (Val BalAcc: 0.995): {'40': 0.3847038149833679, '100': 0.22314836084842682, '200': 0.10606498271226883, '400': 0.2860828638076782}
Epoch 13: Train: Loss 0.0283, Acc 0.772 | Val: Loss 0.0067, Acc 0.992, BalAcc 0.990, F1 0.995, AUC 0.999, Prec 0.995, Rec 0.995, Thresh 0.563 | LR: 0.000100
Epoch 14: Train: Loss 0.0257, Acc 0.747 | Val: Loss 0.0061, Acc 0.992, BalAcc 0.990, F1 0.995, AUC 0.997, Prec 0.995, Rec 0.995, Thresh 0.567 | LR: 0.000100
Epoch 15: Train: Loss 0.0261, Acc 0.790 | Val: Loss 0.0037, Acc 1.000, BalAcc 1.000, F1 1.000, AUC 1.000, Prec 1.000, Rec 1.000, Thresh 0.531 | LR: 0.000100
✅ New best validation balanced accuracy: 1.000, threshold: 0.531
📊 Mag Importance (Val BalAcc: 1.000): {'40': 0.3803335130214691, '100': 0.15947940945625305, '200': 0.13211610913276672, '400': 0.3280710279941559}
Epoch 16: Train: Loss 0.0243, Acc 0.785 | Val: Loss 0.0048, Acc 1.000, BalAcc 1.000, F1 1.000, AUC 1.000, Prec 1.000, Rec 1.000, Thresh 0.446 | LR: 0.000100
Epoch 17: Train: Loss 0.0271, Acc 0.767 | Val: Loss 0.0047, Acc 1.000, BalAcc 1.000, F1 1.000, AUC 1.000, Prec 1.000, Rec 1.000, Thresh 0.609 | LR: 0.000100
Epoch 18: Train: Loss 0.0225, Acc 0.811 | Val: Loss 0.0046, Acc 1.000, BalAcc 1.000, F1 1.000, AUC 1.000, Prec 1.000, Rec 1.000, Thresh 0.553 | LR: 0.000100
Epoch 19: Train: Loss 0.0246, Acc 0.779 | Val: Loss 0.0044, Acc 1.000, BalAcc 1.000, F1 1.000, AUC 1.000, Prec 1.000, Rec 1.000, Thresh 0.649 | LR: 0.000050
Epoch 20: Train: Loss 0.0245, Acc 0.795 | Val: Loss 0.0035, Acc 1.000, BalAcc 1.000, F1 1.000, AUC 1.000, Prec 1.000, Rec 1.000, Thresh 0.604 | LR: 0.000050
Epoch 21: Train: Loss 0.0226, Acc 0.799 | Val: Loss 0.0044, Acc 1.000, BalAcc 1.000, F1 1.000, AUC 1.000, Prec 1.000, Rec 1.000, Thresh 0.630 | LR: 0.000050
Epoch 22: Train: Loss 0.0207, Acc 0.817 | Val: Loss 0.0030, Acc 1.000, BalAcc 1.000, F1 1.000, AUC 1.000, Prec 1.000, Rec 1.000, Thresh 0.682 | LR: 0.000050
⚠️ Early stopping after 22 epochs (no improvement for 7 epochs)
✅ Best model saved: ./output/models/best_model_fold_2.pth (Val BalAcc: 1.000)
⚡️ Test Results: Acc 0.846, BalAcc 0.816, F1 0.889, AUC 0.864, Precision 0.878, Recall 0.900 (threshold: 0.531)
📊 Confusion Matrix (Fold 2):
   [[TN:  68, FP:  25]
    [FN:  20, TP: 180]]
⚡ Avg Inference Time: 0.0176s per sample
📌 Final Magnification Importance (Fold 2): {'40': 0.17091935873031616, '100': 0.25113949179649353, '200': 0.16507850587368011, '400': 0.4128626585006714}
💾 Results saved to: ./output/results/fold_2_results.json

📊 Generating GradCAM visualizations for fold 2...
✅ Generated 5 GradCAM visualizations for fold 2

===== Fold 3 =====
Train patients: 52, Val Patients: 14, Test patients: 16
Training samples per epoch: {'total_samples_per_epoch': 2214, 'class_distribution': {1: 1572, 0: 642}, 'oversampling_factor': 2.1}
Validation samples: 238, Test samples: 354
Patients with full 4 mags: 52
Inner training samples: 2214, batch size: 16
Class weights: Benign=1.73, Malignant=0.70
Epoch 01: Train: Loss 0.0962, Acc 0.597 | Val: Loss 0.0998, Acc 0.647, BalAcc 0.607, F1 0.734, AUC 0.617, Prec 0.748, Rec 0.720, Thresh 0.500 | LR: 0.000100
✅ New best validation balanced accuracy: 0.607, threshold: 0.500
📊 Mag Importance (Val BalAcc: 0.607): {'40': 0.1542242020368576, '100': 0.7183847427368164, '200': 0.08282791078090668, '400': 0.044563110917806625}