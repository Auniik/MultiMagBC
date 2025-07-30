MNet - Multi-Magnification Network for Breast Cancer Classification
Using device: cuda
Batch size: 64
Learning rate: 0.0001
=== Fold-wise Dataset Fairness Summary ===

--- Fold 0 ---
Train: 50 patients (14/36 B/M, 28.0% benign)
Val:   15 patients (5/10 B/M, 33.3% benign)
Test:  17 patients (5/12 B/M, 29.4% benign)
Subtypes present (Train): adenosis, ductal_carcinoma, fibroadenoma, lobular_carcinoma, mucinous_carcinoma, papillary_carcinoma, phyllodes_tumor, tubular_adenoma
Subtypes present (Test):  adenosis, ductal_carcinoma, fibroadenoma, lobular_carcinoma, mucinous_carcinoma, papillary_carcinoma, phyllodes_tumor, tubular_adenoma
Magnification distribution (Train): {'100': 1221, '200': 1219, '40': 1201, '400': 1100}

--- Fold 1 ---
Train: 50 patients (14/36 B/M, 28.0% benign)
Val:   15 patients (5/10 B/M, 33.3% benign)
Test:  17 patients (5/12 B/M, 29.4% benign)
Subtypes present (Train): adenosis, ductal_carcinoma, fibroadenoma, lobular_carcinoma, mucinous_carcinoma, papillary_carcinoma, phyllodes_tumor, tubular_adenoma
Subtypes present (Test):  adenosis, ductal_carcinoma, fibroadenoma, lobular_carcinoma, mucinous_carcinoma, papillary_carcinoma, phyllodes_tumor, tubular_adenoma
Magnification distribution (Train): {'100': 1245, '200': 1190, '40': 1183, '400': 1048}

--- Fold 2 ---
Train: 50 patients (15/35 B/M, 30.0% benign)
Val:   16 patients (5/11 B/M, 31.2% benign)
Test:  16 patients (4/12 B/M, 25.0% benign)
Subtypes present (Train): adenosis, ductal_carcinoma, fibroadenoma, lobular_carcinoma, mucinous_carcinoma, papillary_carcinoma, phyllodes_tumor, tubular_adenoma
Subtypes present (Test):  ductal_carcinoma, fibroadenoma, lobular_carcinoma, mucinous_carcinoma, papillary_carcinoma, phyllodes_tumor, tubular_adenoma
Magnification distribution (Train): {'100': 1328, '200': 1317, '40': 1263, '400': 1153}

--- Fold 3 ---
Train: 51 patients (14/37 B/M, 27.5% benign)
Val:   15 patients (5/10 B/M, 33.3% benign)
Test:  16 patients (5/11 B/M, 31.2% benign)
Subtypes present (Train): adenosis, ductal_carcinoma, fibroadenoma, lobular_carcinoma, mucinous_carcinoma, papillary_carcinoma, phyllodes_tumor, tubular_adenoma
Subtypes present (Test):  adenosis, ductal_carcinoma, fibroadenoma, lobular_carcinoma, mucinous_carcinoma, papillary_carcinoma, tubular_adenoma
Magnification distribution (Train): {'100': 1347, '200': 1248, '40': 1264, '400': 1152}

--- Fold 4 ---
Train: 51 patients (14/37 B/M, 27.5% benign)
Val:   15 patients (5/10 B/M, 33.3% benign)
Test:  16 patients (5/11 B/M, 31.2% benign)
Subtypes present (Train): adenosis, ductal_carcinoma, fibroadenoma, lobular_carcinoma, mucinous_carcinoma, papillary_carcinoma, phyllodes_tumor, tubular_adenoma
Subtypes present (Test):  adenosis, ductal_carcinoma, fibroadenoma, lobular_carcinoma, mucinous_carcinoma, papillary_carcinoma, tubular_adenoma
Magnification distribution (Train): {'100': 1266, '200': 1238, '40': 1236, '400': 1118}

===== Fold 0 =====
Train patients: 50, Val Patients: 15, Test patients: 17
📊 [Sampling] Class weights: {1: 0.6944444444444444, 0: 1.7857142857142858}
📊 [Sampling] Subtype weights: {'lobular_carcinoma': 2.0833333333333335, 'mucinous_carcinoma': 1.0416666666666667, 'ductal_carcinoma': 0.2717391304347826, 'fibroadenoma': 1.0416666666666667, 'tubular_adenoma': 1.25, 'papillary_carcinoma': 1.5625, 'phyllodes_tumor': 6.25, 'adenosis': 3.125}
📊 [Sampling] Weight cap applied at 3.0× median (0.723).
Training samples per epoch: {'total_samples_per_epoch': 2091, 'class_distribution': {1: 1494, 0: 597}, 'subtype_distribution': {'lobular_carcinoma': 135, 'mucinous_carcinoma': 228, 'ductal_carcinoma': 960, 'fibroadenoma': 261, 'tubular_adenoma': 213, 'papillary_carcinoma': 171, 'phyllodes_tumor': 39, 'adenosis': 84}, 'oversampling_factor': 2.13}
Validation samples: 310, Test samples: 495
Patients with full 4 mags: 50
Inner training samples: 2091, batch size: 64
Epoch 01: Train: Loss 0.0863, Acc 0.522 | Val: Loss 0.0743, Acc 0.632, BalAcc 0.570, F1 0.739, AUC 0.574, Prec 0.654, Rec 0.847, Thresh 0.399 | LR: 0.000091
 ✅ New best validation loss: 0.0743 (BalAcc: 0.570, threshold: 0.399)
 📊 Mag Importance (Val Loss: 0.0743): {'40': 0.2457277923822403, '100': 0.2686363458633423, '200': 0.2545804977416992, '400': 0.231055349111557}
Epoch 02: Train: Loss 0.0709, Acc 0.539 | Val: Loss 0.0574, Acc 0.806, BalAcc 0.790, F1 0.845, AUC 0.846, Prec 0.828, Rec 0.863, Thresh 0.447 | LR: 0.000080
 ✅ New best validation loss: 0.0574 (BalAcc: 0.790, threshold: 0.447)
 📊 Mag Importance (Val Loss: 0.0574): {'40': 0.24977833032608032, '100': 0.26519912481307983, '200': 0.25252869725227356, '400': 0.23249389231204987}
Epoch 03: Train: Loss 0.0660, Acc 0.558 | Val: Loss 0.0554, Acc 0.768, BalAcc 0.755, F1 0.811, AUC 0.844, Prec 0.811, Rec 0.811, Thresh 0.415 | LR: 0.000066
 ✅ New best validation loss: 0.0554 (BalAcc: 0.755, threshold: 0.415)
 📊 Mag Importance (Val Loss: 0.0554): {'40': 0.25184115767478943, '100': 0.2634590268135071, '200': 0.2522929310798645, '400': 0.232406884431839}
Epoch 04: Train: Loss 0.0651, Acc 0.580 | Val: Loss 0.0580, Acc 0.758, BalAcc 0.707, F1 0.825, AUC 0.824, Prec 0.741, Rec 0.932, Thresh 0.367 | LR: 0.000051
Epoch 05: Train: Loss 0.0637, Acc 0.583 | Val: Loss 0.0503, Acc 0.823, BalAcc 0.808, F1 0.858, AUC 0.879, Prec 0.843, Rec 0.874, Thresh 0.407 | LR: 0.000036
 ✅ New best validation loss: 0.0503 (BalAcc: 0.808, threshold: 0.407)
 📊 Mag Importance (Val Loss: 0.0503): {'40': 0.25169897079467773, '100': 0.2643278241157532, '200': 0.252471923828125, '400': 0.2315012812614441}
Epoch 06: Train: Loss 0.0590, Acc 0.613 | Val: Loss 0.0512, Acc 0.823, BalAcc 0.800, F1 0.861, AUC 0.878, Prec 0.826, Rec 0.900, Thresh 0.431 | LR: 0.000022
Epoch 07: Train: Loss 0.0581, Acc 0.618 | Val: Loss 0.0491, Acc 0.832, BalAcc 0.832, F1 0.859, AUC 0.897, Prec 0.888, Rec 0.832, Thresh 0.464 | LR: 0.000011
 ✅ New best validation loss: 0.0491 (BalAcc: 0.832, threshold: 0.464)
 📊 Mag Importance (Val Loss: 0.0491): {'40': 0.25181010365486145, '100': 0.2647407352924347, '200': 0.25197654962539673, '400': 0.23147261142730713}
Epoch 08: Train: Loss 0.0601, Acc 0.573 | Val: Loss 0.0499, Acc 0.848, BalAcc 0.829, F1 0.881, AUC 0.890, Prec 0.849, Rec 0.916, Thresh 0.407 | LR: 0.000004
Epoch 09: Train: Loss 0.0574, Acc 0.615 | Val: Loss 0.0511, Acc 0.800, BalAcc 0.762, F1 0.851, AUC 0.866, Prec 0.783, Rec 0.932, Thresh 0.359 | LR: 0.000001
Epoch 10: Train: Loss 0.0576, Acc 0.643 | Val: Loss 0.0502, Acc 0.865, BalAcc 0.848, F1 0.893, AUC 0.895, Prec 0.866, Rec 0.921, Thresh 0.456 | LR: 0.000099
Epoch 11: Train: Loss 0.0529, Acc 0.643 | Val: Loss 0.0386, Acc 0.906, BalAcc 0.901, F1 0.924, AUC 0.943, Prec 0.921, Rec 0.926, Thresh 0.439 | LR: 0.000098
 ✅ New best validation loss: 0.0386 (BalAcc: 0.901, threshold: 0.439)
 📊 Mag Importance (Val Loss: 0.0386): {'40': 0.25368863344192505, '100': 0.26578545570373535, '200': 0.2514175772666931, '400': 0.22910822927951813}
Epoch 12: Train: Loss 0.0553, Acc 0.657 | Val: Loss 0.0434, Acc 0.890, BalAcc 0.872, F1 0.914, AUC 0.936, Prec 0.879, Rec 0.953, Thresh 0.423 | LR: 0.000095
Epoch 13: Train: Loss 0.0573, Acc 0.651 | Val: Loss 0.0442, Acc 0.890, BalAcc 0.884, F1 0.911, AUC 0.929, Prec 0.911, Rec 0.911, Thresh 0.439 | LR: 0.000091
Epoch 14: Train: Loss 0.0526, Acc 0.656 | Val: Loss 0.0462, Acc 0.839, BalAcc 0.822, F1 0.872, AUC 0.898, Prec 0.850, Rec 0.895, Thresh 0.391 | LR: 0.000086
Epoch 15: Train: Loss 0.0516, Acc 0.654 | Val: Loss 0.0409, Acc 0.887, BalAcc 0.868, F1 0.912, AUC 0.945, Prec 0.874, Rec 0.953, Thresh 0.439 | LR: 0.000080
Epoch 16: Train: Loss 0.0498, Acc 0.619 | Val: Loss 0.0455, Acc 0.884, BalAcc 0.858, F1 0.911, AUC 0.932, Prec 0.856, Rec 0.974, Thresh 0.407 | LR: 0.000073
Epoch 17: Train: Loss 0.0571, Acc 0.633 | Val: Loss 0.0352, Acc 0.900, BalAcc 0.898, F1 0.917, AUC 0.957, Prec 0.930, Rec 0.905, Thresh 0.423 | LR: 0.000066
 ✅ New best validation loss: 0.0352 (BalAcc: 0.898, threshold: 0.423)
 📊 Mag Importance (Val Loss: 0.0352): {'40': 0.25362586975097656, '100': 0.26451194286346436, '200': 0.2513135075569153, '400': 0.23054863512516022}
Epoch 18: Train: Loss 0.0462, Acc 0.655 | Val: Loss 0.0405, Acc 0.916, BalAcc 0.905, F1 0.933, AUC 0.945, Prec 0.914, Rec 0.953, Thresh 0.431 | LR: 0.000058
Epoch 19: Train: Loss 0.0514, Acc 0.646 | Val: Loss 0.0426, Acc 0.900, BalAcc 0.889, F1 0.920, AUC 0.940, Prec 0.904, Rec 0.937, Thresh 0.472 | LR: 0.000051
Epoch 20: Train: Loss 0.0498, Acc 0.675 | Val: Loss 0.0317, Acc 0.942, BalAcc 0.931, F1 0.954, AUC 0.978, Prec 0.930, Rec 0.979, Thresh 0.423 | LR: 0.000043
 ✅ New best validation loss: 0.0317 (BalAcc: 0.931, threshold: 0.423)
 📊 Mag Importance (Val Loss: 0.0317): {'40': 0.2540762722492218, '100': 0.26520583033561707, '200': 0.25020208954811096, '400': 0.2305157631635666}
Epoch 21: Train: Loss 0.0463, Acc 0.676 | Val: Loss 0.0345, Acc 0.910, BalAcc 0.893, F1 0.929, AUC 0.964, Prec 0.893, Rec 0.968, Thresh 0.439 | LR: 0.000035
Epoch 22: Train: Loss 0.0497, Acc 0.683 | Val: Loss 0.0393, Acc 0.900, BalAcc 0.892, F1 0.919, AUC 0.947, Prec 0.912, Rec 0.926, Thresh 0.480 | LR: 0.000028
Epoch 23: Train: Loss 0.0522, Acc 0.659 | Val: Loss 0.0327, Acc 0.935, BalAcc 0.934, F1 0.947, AUC 0.976, Prec 0.952, Rec 0.942, Thresh 0.480 | LR: 0.000022
Epoch 24: Train: Loss 0.0474, Acc 0.655 | Val: Loss 0.0355, Acc 0.923, BalAcc 0.925, F1 0.935, AUC 0.973, Prec 0.956, Rec 0.916, Thresh 0.472 | LR: 0.000016
Epoch 25: Train: Loss 0.0480, Acc 0.727 | Val: Loss 0.0336, Acc 0.942, BalAcc 0.936, F1 0.953, AUC 0.967, Prec 0.943, Rec 0.963, Thresh 0.439 | LR: 0.000011
Epoch 26: Train: Loss 0.0436, Acc 0.662 | Val: Loss 0.0418, Acc 0.890, BalAcc 0.883, F1 0.911, AUC 0.942, Prec 0.906, Rec 0.916, Thresh 0.464 | LR: 0.000007
Epoch 27: Train: Loss 0.0438, Acc 0.695 | Val: Loss 0.0464, Acc 0.890, BalAcc 0.880, F1 0.912, AUC 0.943, Prec 0.898, Rec 0.926, Thresh 0.456 | LR: 0.000003
 ⚠️ Early stopping after 27 epochs (no improvement for 7 epochs)
✅ Best model saved: ./output/models/best_model_fold_0.pth (Val Loss: 0.0317, BalAcc: 0.931)
⚡️ Test Results: Acc 0.721, BalAcc 0.690, F1 0.784, AUC 0.837, Precision 0.740, Recall 0.834 (threshold: 0.423)
📊 Confusion Matrix (Fold 0):
   [[TN: 106, FP:  88]
    [FN:  50, TP: 251]]
⚡ Avg Inference Time: 0.0129s per sample
📌 Final Magnification Importance (Fold 0): {'40': 0.25243091583251953, '100': 0.26600274443626404, '200': 0.24986763298511505, '400': 0.23169870674610138}
💾 Results saved to: ./output/results/fold_0_results.json

📊 Generating GradCAM visualizations for fold 0...
✅ Generated 5 GradCAM visualizations for fold 0

===== Fold 1 =====
Train patients: 50, Val Patients: 15, Test patients: 17
📊 [Sampling] Class weights: {1: 0.6944444444444444, 0: 1.7857142857142858}
📊 [Sampling] Subtype weights: {'lobular_carcinoma': 2.0833333333333335, 'ductal_carcinoma': 0.2717391304347826, 'fibroadenoma': 1.0416666666666667, 'adenosis': 3.125, 'tubular_adenoma': 1.25, 'mucinous_carcinoma': 1.0416666666666667, 'papillary_carcinoma': 1.5625, 'phyllodes_tumor': 6.25}
📊 [Sampling] Weight cap applied at 3.0× median (0.723).
Training samples per epoch: {'total_samples_per_epoch': 2028, 'class_distribution': {1: 1413, 0: 615}, 'subtype_distribution': {'lobular_carcinoma': 135, 'ductal_carcinoma': 876, 'fibroadenoma': 261, 'adenosis': 90, 'tubular_adenoma': 219, 'mucinous_carcinoma': 231, 'papillary_carcinoma': 171, 'phyllodes_tumor': 45}, 'oversampling_factor': 2.07}
Validation samples: 316, Test samples: 476
Patients with full 4 mags: 50
Inner training samples: 2028, batch size: 64
Epoch 01: Train: Loss 0.0826, Acc 0.444 | Val: Loss 0.0767, Acc 0.633, BalAcc 0.508, F1 0.773, AUC 0.510, Prec 0.631, Rec 1.000, Thresh 0.254 | LR: 0.000091
 ✅ New best validation loss: 0.0767 (BalAcc: 0.508, threshold: 0.254)
 📊 Mag Importance (Val Loss: 0.0767): {'40': 0.24561353027820587, '100': 0.23635168373584747, '200': 0.23323790729045868, '400': 0.2847968637943268}
Epoch 02: Train: Loss 0.0751, Acc 0.470 | Val: Loss 0.0714, Acc 0.684, BalAcc 0.634, F1 0.766, AUC 0.709, Prec 0.713, Rec 0.828, Thresh 0.407 | LR: 0.000080
 ✅ New best validation loss: 0.0714 (BalAcc: 0.634, threshold: 0.407)
 📊 Mag Importance (Val Loss: 0.0714): {'40': 0.2489488124847412, '100': 0.23325370252132416, '200': 0.22854676842689514, '400': 0.2892507314682007}
Epoch 03: Train: Loss 0.0722, Acc 0.528 | Val: Loss 0.0781, Acc 0.699, BalAcc 0.637, F1 0.787, AUC 0.730, Prec 0.709, Rec 0.884, Thresh 0.367 | LR: 0.000066
Epoch 04: Train: Loss 0.0670, Acc 0.550 | Val: Loss 0.0713, Acc 0.741, BalAcc 0.690, F1 0.811, AUC 0.766, Prec 0.746, Rec 0.889, Thresh 0.375 | LR: 0.000051
 ✅ New best validation loss: 0.0713 (BalAcc: 0.690, threshold: 0.375)
 📊 Mag Importance (Val Loss: 0.0713): {'40': 0.24811580777168274, '100': 0.23279882967472076, '200': 0.22957104444503784, '400': 0.28951433300971985}
Epoch 05: Train: Loss 0.0651, Acc 0.542 | Val: Loss 0.0714, Acc 0.731, BalAcc 0.674, F1 0.807, AUC 0.763, Prec 0.733, Rec 0.899, Thresh 0.367 | LR: 0.000036
Epoch 06: Train: Loss 0.0644, Acc 0.584 | Val: Loss 0.0674, Acc 0.731, BalAcc 0.720, F1 0.780, AUC 0.794, Prec 0.799, Rec 0.763, Thresh 0.488 | LR: 0.000022
 ✅ New best validation loss: 0.0674 (BalAcc: 0.720, threshold: 0.488)
 📊 Mag Importance (Val Loss: 0.0674): {'40': 0.24813741445541382, '100': 0.2327544242143631, '200': 0.2294357568025589, '400': 0.2896723747253418}
Epoch 07: Train: Loss 0.0660, Acc 0.557 | Val: Loss 0.0597, Acc 0.782, BalAcc 0.757, F1 0.830, AUC 0.828, Prec 0.809, Rec 0.854, Thresh 0.431 | LR: 0.000011
 ✅ New best validation loss: 0.0597 (BalAcc: 0.757, threshold: 0.431)
 📊 Mag Importance (Val Loss: 0.0597): {'40': 0.24865512549877167, '100': 0.23272284865379333, '200': 0.22859850525856018, '400': 0.2900235652923584}
Epoch 08: Train: Loss 0.0615, Acc 0.566 | Val: Loss 0.0708, Acc 0.718, BalAcc 0.688, F1 0.782, AUC 0.777, Prec 0.758, Rec 0.808, Thresh 0.447 | LR: 0.000004
Epoch 09: Train: Loss 0.0632, Acc 0.574 | Val: Loss 0.0726, Acc 0.734, BalAcc 0.721, F1 0.785, AUC 0.770, Prec 0.797, Rec 0.773, Thresh 0.496 | LR: 0.000001
Epoch 10: Train: Loss 0.0576, Acc 0.576 | Val: Loss 0.0621, Acc 0.753, BalAcc 0.731, F1 0.806, AUC 0.809, Prec 0.794, Rec 0.818, Thresh 0.431 | LR: 0.000099
Epoch 11: Train: Loss 0.0583, Acc 0.627 | Val: Loss 0.0613, Acc 0.794, BalAcc 0.761, F1 0.845, AUC 0.848, Prec 0.801, Rec 0.894, Thresh 0.456 | LR: 0.000098
Epoch 12: Train: Loss 0.0562, Acc 0.634 | Val: Loss 0.0661, Acc 0.769, BalAcc 0.756, F1 0.814, AUC 0.813, Prec 0.821, Rec 0.808, Thresh 0.480 | LR: 0.000095
Epoch 13: Train: Loss 0.0513, Acc 0.662 | Val: Loss 0.0567, Acc 0.807, BalAcc 0.795, F1 0.846, AUC 0.872, Prec 0.848, Rec 0.843, Thresh 0.504 | LR: 0.000091
 ✅ New best validation loss: 0.0567 (BalAcc: 0.795, threshold: 0.504)
 📊 Mag Importance (Val Loss: 0.0567): {'40': 0.25267523527145386, '100': 0.2315751016139984, '200': 0.22706912457942963, '400': 0.2886805534362793}
Epoch 14: Train: Loss 0.0502, Acc 0.662 | Val: Loss 0.0575, Acc 0.820, BalAcc 0.793, F1 0.862, AUC 0.878, Prec 0.828, Rec 0.899, Thresh 0.464 | LR: 0.000086
Epoch 15: Train: Loss 0.0525, Acc 0.662 | Val: Loss 0.0589, Acc 0.816, BalAcc 0.782, F1 0.863, AUC 0.874, Prec 0.812, Rec 0.919, Thresh 0.464 | LR: 0.000080
Epoch 16: Train: Loss 0.0463, Acc 0.690 | Val: Loss 0.0649, Acc 0.785, BalAcc 0.727, F1 0.848, AUC 0.851, Prec 0.762, Rec 0.955, Thresh 0.439 | LR: 0.000073
Epoch 17: Train: Loss 0.0495, Acc 0.684 | Val: Loss 0.0646, Acc 0.750, BalAcc 0.676, F1 0.829, AUC 0.831, Prec 0.725, Rec 0.970, Thresh 0.342 | LR: 0.000066
Epoch 18: Train: Loss 0.0512, Acc 0.702 | Val: Loss 0.0532, Acc 0.801, BalAcc 0.759, F1 0.853, AUC 0.872, Prec 0.792, Rec 0.924, Thresh 0.431 | LR: 0.000058
 ✅ New best validation loss: 0.0532 (BalAcc: 0.759, threshold: 0.431)
 📊 Mag Importance (Val Loss: 0.0532): {'40': 0.24811452627182007, '100': 0.23538623750209808, '200': 0.2310100495815277, '400': 0.28548920154571533}
Epoch 19: Train: Loss 0.0486, Acc 0.716 | Val: Loss 0.0472, Acc 0.842, BalAcc 0.797, F1 0.885, AUC 0.908, Prec 0.811, Rec 0.975, Thresh 0.367 | LR: 0.000051
 ✅ New best validation loss: 0.0472 (BalAcc: 0.797, threshold: 0.367)
 📊 Mag Importance (Val Loss: 0.0472): {'40': 0.2503051161766052, '100': 0.23494015634059906, '200': 0.22717860341072083, '400': 0.2875761091709137}
Epoch 20: Train: Loss 0.0457, Acc 0.710 | Val: Loss 0.0530, Acc 0.807, BalAcc 0.772, F1 0.855, AUC 0.885, Prec 0.807, Rec 0.909, Thresh 0.464 | LR: 0.000043
Epoch 21: Train: Loss 0.0487, Acc 0.702 | Val: Loss 0.0472, Acc 0.835, BalAcc 0.802, F1 0.877, AUC 0.904, Prec 0.826, Rec 0.934, Thresh 0.456 | LR: 0.000035
 ✅ New best validation loss: 0.0472 (BalAcc: 0.802, threshold: 0.456)
 📊 Mag Importance (Val Loss: 0.0472): {'40': 0.25043150782585144, '100': 0.23515990376472473, '200': 0.22649113833904266, '400': 0.28791746497154236}
Epoch 22: Train: Loss 0.0505, Acc 0.741 | Val: Loss 0.0520, Acc 0.794, BalAcc 0.752, F1 0.848, AUC 0.874, Prec 0.788, Rec 0.919, Thresh 0.391 | LR: 0.000028
Epoch 23: Train: Loss 0.0438, Acc 0.694 | Val: Loss 0.0564, Acc 0.788, BalAcc 0.757, F1 0.839, AUC 0.875, Prec 0.802, Rec 0.879, Thresh 0.496 | LR: 0.000022
Epoch 24: Train: Loss 0.0546, Acc 0.670 | Val: Loss 0.0536, Acc 0.832, BalAcc 0.815, F1 0.868, AUC 0.898, Prec 0.854, Rec 0.884, Thresh 0.528 | LR: 0.000016
Epoch 25: Train: Loss 0.0466, Acc 0.725 | Val: Loss 0.0528, Acc 0.816, BalAcc 0.771, F1 0.866, AUC 0.886, Prec 0.797, Rec 0.949, Thresh 0.431 | LR: 0.000011
Epoch 26: Train: Loss 0.0480, Acc 0.706 | Val: Loss 0.0513, Acc 0.820, BalAcc 0.791, F1 0.863, AUC 0.895, Prec 0.825, Rec 0.904, Thresh 0.496 | LR: 0.000007
Epoch 27: Train: Loss 0.0438, Acc 0.738 | Val: Loss 0.0519, Acc 0.810, BalAcc 0.780, F1 0.856, AUC 0.884, Prec 0.817, Rec 0.899, Thresh 0.456 | LR: 0.000004
Epoch 28: Train: Loss 0.0473, Acc 0.704 | Val: Loss 0.0554, Acc 0.835, BalAcc 0.812, F1 0.873, AUC 0.881, Prec 0.844, Rec 0.904, Thresh 0.496 | LR: 0.000002
 ⚠️ Early stopping after 28 epochs (no improvement for 7 epochs)
✅ Best model saved: ./output/models/best_model_fold_1.pth (Val Loss: 0.0472, BalAcc: 0.802)
⚡️ Test Results: Acc 0.947, BalAcc 0.888, F1 0.967, AUC 0.972, Precision 0.936, Recall 1.000 (threshold: 0.456)
📊 Confusion Matrix (Fold 1):
   [[TN:  87, FP:  25]
    [FN:   0, TP: 364]]
⚡ Avg Inference Time: 0.0134s per sample
📌 Final Magnification Importance (Fold 1): {'40': 0.2502753436565399, '100': 0.23327702283859253, '200': 0.2263369858264923, '400': 0.29011067748069763}
💾 Results saved to: ./output/results/fold_1_results.json

📊 Generating GradCAM visualizations for fold 1...
✅ Generated 5 GradCAM visualizations for fold 1

===== Fold 2 =====
Train patients: 50, Val Patients: 16, Test patients: 16
📊 [Sampling] Class weights: {1: 0.7142857142857143, 0: 1.6666666666666667}
📊 [Sampling] Subtype weights: {'lobular_carcinoma': 2.0833333333333335, 'mucinous_carcinoma': 1.0416666666666667, 'ductal_carcinoma': 0.2717391304347826, 'fibroadenoma': 1.0416666666666667, 'tubular_adenoma': 1.25, 'adenosis': 2.0833333333333335, 'papillary_carcinoma': 2.0833333333333335, 'phyllodes_tumor': 6.25}
📊 [Sampling] Weight cap applied at 3.0× median (0.744).
Training samples per epoch: {'total_samples_per_epoch': 2094, 'class_distribution': {1: 1455, 0: 639}, 'subtype_distribution': {'lobular_carcinoma': 135, 'mucinous_carcinoma': 246, 'ductal_carcinoma': 942, 'fibroadenoma': 258, 'tubular_adenoma': 213, 'adenosis': 129, 'papillary_carcinoma': 132, 'phyllodes_tumor': 39}, 'oversampling_factor': 2.1}
Validation samples: 373, Test samples: 408
Patients with full 4 mags: 50
Inner training samples: 2094, batch size: 64
Epoch 01: Train: Loss 0.0988, Acc 0.549 | Val: Loss 0.0916, Acc 0.571, BalAcc 0.525, F1 0.706, AUC 0.504, Prec 0.571, Rec 0.923, Thresh 0.464 | LR: 0.000091
 ✅ New best validation loss: 0.0916 (BalAcc: 0.525, threshold: 0.464)
 📊 Mag Importance (Val Loss: 0.0916): {'40': 0.2510715425014496, '100': 0.23490045964717865, '200': 0.24010194838047028, '400': 0.2739260196685791}
Epoch 02: Train: Loss 0.0835, Acc 0.574 | Val: Loss 0.0742, Acc 0.694, BalAcc 0.713, F1 0.669, AUC 0.743, Prec 0.846, Rec 0.553, Thresh 0.561 | LR: 0.000080
 ✅ New best validation loss: 0.0742 (BalAcc: 0.713, threshold: 0.561)
 📊 Mag Importance (Val Loss: 0.0742): {'40': 0.25391796231269836, '100': 0.2322879135608673, '200': 0.23866568505764008, '400': 0.27512845396995544}
Epoch 03: Train: Loss 0.0799, Acc 0.541 | Val: Loss 0.0725, Acc 0.740, BalAcc 0.744, F1 0.753, AUC 0.782, Prec 0.800, Rec 0.712, Thresh 0.504 | LR: 0.000066
 ✅ New best validation loss: 0.0725 (BalAcc: 0.744, threshold: 0.504)
 📊 Mag Importance (Val Loss: 0.0725): {'40': 0.2538382112979889, '100': 0.2325919270515442, '200': 0.23798783123493195, '400': 0.2755819857120514}
Epoch 04: Train: Loss 0.0682, Acc 0.559 | Val: Loss 0.0591, Acc 0.769, BalAcc 0.776, F1 0.776, AUC 0.846, Prec 0.847, Rec 0.716, Thresh 0.512 | LR: 0.000051
 ✅ New best validation loss: 0.0591 (BalAcc: 0.776, threshold: 0.512)
 📊 Mag Importance (Val Loss: 0.0591): {'40': 0.25449931621551514, '100': 0.2325333058834076, '200': 0.23772108554840088, '400': 0.2752462923526764}
Epoch 05: Train: Loss 0.0715, Acc 0.544 | Val: Loss 0.0538, Acc 0.812, BalAcc 0.812, F1 0.829, AUC 0.882, Prec 0.842, Rec 0.817, Thresh 0.488 | LR: 0.000036
 ✅ New best validation loss: 0.0538 (BalAcc: 0.812, threshold: 0.488)
 📊 Mag Importance (Val Loss: 0.0538): {'40': 0.2541920840740204, '100': 0.23211270570755005, '200': 0.23820148408412933, '400': 0.27549371123313904}
Epoch 06: Train: Loss 0.0718, Acc 0.585 | Val: Loss 0.0586, Acc 0.788, BalAcc 0.800, F1 0.786, AUC 0.843, Prec 0.901, Rec 0.697, Thresh 0.544 | LR: 0.000022
Epoch 07: Train: Loss 0.0644, Acc 0.562 | Val: Loss 0.0600, Acc 0.802, BalAcc 0.808, F1 0.808, AUC 0.872, Prec 0.876, Rec 0.750, Thresh 0.553 | LR: 0.000011
Epoch 08: Train: Loss 0.0686, Acc 0.565 | Val: Loss 0.0581, Acc 0.807, BalAcc 0.815, F1 0.812, AUC 0.870, Prec 0.891, Rec 0.745, Thresh 0.520 | LR: 0.000004
Epoch 09: Train: Loss 0.0686, Acc 0.592 | Val: Loss 0.0594, Acc 0.761, BalAcc 0.770, F1 0.764, AUC 0.841, Prec 0.852, Rec 0.692, Thresh 0.520 | LR: 0.000001
Epoch 10: Train: Loss 0.0675, Acc 0.595 | Val: Loss 0.0592, Acc 0.764, BalAcc 0.762, F1 0.786, AUC 0.839, Prec 0.794, Rec 0.779, Thresh 0.480 | LR: 0.000099
Epoch 11: Train: Loss 0.0646, Acc 0.583 | Val: Loss 0.0576, Acc 0.794, BalAcc 0.792, F1 0.814, AUC 0.878, Prec 0.820, Rec 0.808, Thresh 0.504 | LR: 0.000098
Epoch 12: Train: Loss 0.0593, Acc 0.597 | Val: Loss 0.0479, Acc 0.810, BalAcc 0.816, F1 0.817, AUC 0.900, Prec 0.883, Rec 0.760, Thresh 0.504 | LR: 0.000095
 ✅ New best validation loss: 0.0479 (BalAcc: 0.816, threshold: 0.504)
 📊 Mag Importance (Val Loss: 0.0479): {'40': 0.25304046273231506, '100': 0.23291465640068054, '200': 0.23868076503276825, '400': 0.27536413073539734}
Epoch 13: Train: Loss 0.0638, Acc 0.598 | Val: Loss 0.0538, Acc 0.786, BalAcc 0.797, F1 0.784, AUC 0.880, Prec 0.895, Rec 0.697, Thresh 0.544 | LR: 0.000091
Epoch 14: Train: Loss 0.0604, Acc 0.626 | Val: Loss 0.0511, Acc 0.815, BalAcc 0.813, F1 0.833, AUC 0.899, Prec 0.839, Rec 0.827, Thresh 0.496 | LR: 0.000086
Epoch 15: Train: Loss 0.0586, Acc 0.615 | Val: Loss 0.0491, Acc 0.828, BalAcc 0.819, F1 0.854, AUC 0.905, Prec 0.813, Rec 0.899, Thresh 0.456 | LR: 0.000080
Epoch 16: Train: Loss 0.0582, Acc 0.633 | Val: Loss 0.0543, Acc 0.815, BalAcc 0.807, F1 0.841, AUC 0.892, Prec 0.809, Rec 0.875, Thresh 0.488 | LR: 0.000073
Epoch 17: Train: Loss 0.0550, Acc 0.620 | Val: Loss 0.0539, Acc 0.812, BalAcc 0.802, F1 0.842, AUC 0.897, Prec 0.795, Rec 0.894, Thresh 0.488 | LR: 0.000066
Epoch 18: Train: Loss 0.0579, Acc 0.632 | Val: Loss 0.0440, Acc 0.858, BalAcc 0.855, F1 0.874, AUC 0.927, Prec 0.867, Rec 0.880, Thresh 0.480 | LR: 0.000058
 ✅ New best validation loss: 0.0440 (BalAcc: 0.855, threshold: 0.480)
 📊 Mag Importance (Val Loss: 0.0440): {'40': 0.25577476620674133, '100': 0.2292187511920929, '200': 0.23888759315013885, '400': 0.2761189043521881}
Epoch 19: Train: Loss 0.0505, Acc 0.659 | Val: Loss 0.0578, Acc 0.820, BalAcc 0.812, F1 0.846, AUC 0.902, Prec 0.811, Rec 0.885, Thresh 0.504 | LR: 0.000051
Epoch 20: Train: Loss 0.0582, Acc 0.643 | Val: Loss 0.0449, Acc 0.866, BalAcc 0.863, F1 0.881, AUC 0.931, Prec 0.873, Rec 0.889, Thresh 0.504 | LR: 0.000043
Epoch 21: Train: Loss 0.0561, Acc 0.666 | Val: Loss 0.0486, Acc 0.836, BalAcc 0.831, F1 0.856, AUC 0.912, Prec 0.839, Rec 0.875, Thresh 0.480 | LR: 0.000035
Epoch 22: Train: Loss 0.0502, Acc 0.610 | Val: Loss 0.0470, Acc 0.850, BalAcc 0.847, F1 0.867, AUC 0.922, Prec 0.858, Rec 0.875, Thresh 0.504 | LR: 0.000028
Epoch 23: Train: Loss 0.0521, Acc 0.660 | Val: Loss 0.0505, Acc 0.834, BalAcc 0.835, F1 0.847, AUC 0.903, Prec 0.872, Rec 0.822, Thresh 0.512 | LR: 0.000022
Epoch 24: Train: Loss 0.0490, Acc 0.660 | Val: Loss 0.0489, Acc 0.845, BalAcc 0.846, F1 0.856, AUC 0.923, Prec 0.883, Rec 0.832, Thresh 0.528 | LR: 0.000016
Epoch 25: Train: Loss 0.0503, Acc 0.617 | Val: Loss 0.0474, Acc 0.850, BalAcc 0.833, F1 0.879, AUC 0.916, Prec 0.799, Rec 0.976, Thresh 0.439 | LR: 0.000011
 ⚠️ Early stopping after 25 epochs (no improvement for 7 epochs)
✅ Best model saved: ./output/models/best_model_fold_2.pth (Val Loss: 0.0440, BalAcc: 0.855)
⚡️ Test Results: Acc 0.926, BalAcc 0.944, F1 0.947, AUC 0.986, Precision 0.993, Recall 0.905 (threshold: 0.480)
📊 Confusion Matrix (Fold 2):
   [[TN: 111, FP:   2]
    [FN:  28, TP: 267]]
⚡ Avg Inference Time: 0.0141s per sample
📌 Final Magnification Importance (Fold 2): {'40': 0.256552517414093, '100': 0.2300790548324585, '200': 0.23857782781124115, '400': 0.27479058504104614}
💾 Results saved to: ./output/results/fold_2_results.json

📊 Generating GradCAM visualizations for fold 2...
✅ Generated 5 GradCAM visualizations for fold 2

===== Fold 3 =====
Train patients: 51, Val Patients: 15, Test patients: 16
📊 [Sampling] Class weights: {1: 0.6891891891891891, 0: 1.8214285714285714}
📊 [Sampling] Subtype weights: {'mucinous_carcinoma': 1.0625, 'ductal_carcinoma': 0.265625, 'fibroadenoma': 1.0625, 'papillary_carcinoma': 1.59375, 'tubular_adenoma': 1.59375, 'adenosis': 3.1875, 'lobular_carcinoma': 2.125, 'phyllodes_tumor': 3.1875}
📊 [Sampling] Weight cap applied at 3.0× median (0.732).
Training samples per epoch: {'total_samples_per_epoch': 2136, 'class_distribution': {1: 1539, 0: 597}, 'subtype_distribution': {'mucinous_carcinoma': 237, 'ductal_carcinoma': 993, 'fibroadenoma': 249, 'papillary_carcinoma': 174, 'tubular_adenoma': 174, 'adenosis': 84, 'lobular_carcinoma': 135, 'phyllodes_tumor': 90}, 'oversampling_factor': 2.05}
Validation samples: 275, Test samples: 447
Patients with full 4 mags: 51
Inner training samples: 2136, batch size: 64
Epoch 01: Train: Loss 0.0930, Acc 0.478 | Val: Loss 0.0726, Acc 0.640, BalAcc 0.550, F1 0.766, AUC 0.559, Prec 0.633, Rec 0.970, Thresh 0.391 | LR: 0.000091
 ✅ New best validation loss: 0.0726 (BalAcc: 0.550, threshold: 0.391)
 📊 Mag Importance (Val Loss: 0.0726): {'40': 0.2666904926300049, '100': 0.23066169023513794, '200': 0.2595927119255066, '400': 0.24305514991283417}
Epoch 02: Train: Loss 0.0796, Acc 0.503 | Val: Loss 0.0652, Acc 0.738, BalAcc 0.694, F1 0.806, AUC 0.728, Prec 0.732, Rec 0.898, Thresh 0.383 | LR: 0.000080
 ✅ New best validation loss: 0.0652 (BalAcc: 0.694, threshold: 0.383)
 📊 Mag Importance (Val Loss: 0.0652): {'40': 0.26647046208381653, '100': 0.22991915047168732, '200': 0.26027366518974304, '400': 0.2433367371559143}
Epoch 03: Train: Loss 0.0738, Acc 0.529 | Val: Loss 0.0917, Acc 0.724, BalAcc 0.700, F1 0.780, AUC 0.721, Prec 0.754, Rec 0.808, Thresh 0.399 | LR: 0.000066
Epoch 04: Train: Loss 0.0678, Acc 0.540 | Val: Loss 0.0825, Acc 0.705, BalAcc 0.690, F1 0.758, AUC 0.710, Prec 0.756, Rec 0.760, Thresh 0.447 | LR: 0.000051
Epoch 05: Train: Loss 0.0678, Acc 0.596 | Val: Loss 0.0888, Acc 0.735, BalAcc 0.696, F1 0.800, AUC 0.735, Prec 0.737, Rec 0.874, Thresh 0.375 | LR: 0.000036
Epoch 06: Train: Loss 0.0601, Acc 0.598 | Val: Loss 0.0637, Acc 0.771, BalAcc 0.731, F1 0.829, AUC 0.779, Prec 0.757, Rec 0.916, Thresh 0.423 | LR: 0.000022
 ✅ New best validation loss: 0.0637 (BalAcc: 0.731, threshold: 0.423)
 📊 Mag Importance (Val Loss: 0.0637): {'40': 0.2655164897441864, '100': 0.22957096993923187, '200': 0.2606675624847412, '400': 0.2442449927330017}
Epoch 07: Train: Loss 0.0589, Acc 0.594 | Val: Loss 0.0645, Acc 0.782, BalAcc 0.755, F1 0.831, AUC 0.793, Prec 0.786, Rec 0.880, Thresh 0.447 | LR: 0.000011
Epoch 08: Train: Loss 0.0590, Acc 0.609 | Val: Loss 0.0736, Acc 0.753, BalAcc 0.736, F1 0.800, AUC 0.768, Prec 0.786, Rec 0.814, Thresh 0.472 | LR: 0.000004
Epoch 09: Train: Loss 0.0595, Acc 0.627 | Val: Loss 0.0697, Acc 0.785, BalAcc 0.768, F1 0.828, AUC 0.814, Prec 0.807, Rec 0.850, Thresh 0.456 | LR: 0.000001
Epoch 10: Train: Loss 0.0626, Acc 0.631 | Val: Loss 0.0990, Acc 0.745, BalAcc 0.728, F1 0.794, AUC 0.756, Prec 0.780, Rec 0.808, Thresh 0.512 | LR: 0.000099
Epoch 11: Train: Loss 0.0571, Acc 0.625 | Val: Loss 0.0785, Acc 0.738, BalAcc 0.693, F1 0.807, AUC 0.768, Prec 0.729, Rec 0.904, Thresh 0.415 | LR: 0.000098
Epoch 12: Train: Loss 0.0536, Acc 0.683 | Val: Loss 0.0661, Acc 0.775, BalAcc 0.741, F1 0.829, AUC 0.818, Prec 0.769, Rec 0.898, Thresh 0.496 | LR: 0.000095
Epoch 13: Train: Loss 0.0584, Acc 0.673 | Val: Loss 0.0615, Acc 0.782, BalAcc 0.748, F1 0.834, AUC 0.812, Prec 0.774, Rec 0.904, Thresh 0.472 | LR: 0.000091
 ✅ New best validation loss: 0.0615 (BalAcc: 0.748, threshold: 0.472)
 📊 Mag Importance (Val Loss: 0.0615): {'40': 0.26368579268455505, '100': 0.22844304144382477, '200': 0.25865113735198975, '400': 0.24921999871730804}
Epoch 14: Train: Loss 0.0552, Acc 0.672 | Val: Loss 0.0669, Acc 0.764, BalAcc 0.717, F1 0.828, AUC 0.792, Prec 0.743, Rec 0.934, Thresh 0.431 | LR: 0.000086
Epoch 15: Train: Loss 0.0543, Acc 0.662 | Val: Loss 0.0573, Acc 0.811, BalAcc 0.784, F1 0.854, AUC 0.852, Prec 0.804, Rec 0.910, Thresh 0.464 | LR: 0.000080
 ✅ New best validation loss: 0.0573 (BalAcc: 0.784, threshold: 0.464)
 📊 Mag Importance (Val Loss: 0.0573): {'40': 0.26396700739860535, '100': 0.22821666300296783, '200': 0.2589503824710846, '400': 0.24886593222618103}
Epoch 16: Train: Loss 0.0562, Acc 0.718 | Val: Loss 0.0730, Acc 0.789, BalAcc 0.746, F1 0.845, AUC 0.799, Prec 0.763, Rec 0.946, Thresh 0.472 | LR: 0.000073
Epoch 17: Train: Loss 0.0518, Acc 0.632 | Val: Loss 0.0786, Acc 0.756, BalAcc 0.708, F1 0.823, AUC 0.780, Prec 0.736, Rec 0.934, Thresh 0.456 | LR: 0.000066
Epoch 18: Train: Loss 0.0502, Acc 0.698 | Val: Loss 0.0609, Acc 0.818, BalAcc 0.778, F1 0.866, AUC 0.859, Prec 0.785, Rec 0.964, Thresh 0.480 | LR: 0.000058
Epoch 19: Train: Loss 0.0458, Acc 0.692 | Val: Loss 0.0712, Acc 0.789, BalAcc 0.759, F1 0.838, AUC 0.820, Prec 0.785, Rec 0.898, Thresh 0.512 | LR: 0.000051
Epoch 20: Train: Loss 0.0514, Acc 0.693 | Val: Loss 0.0679, Acc 0.804, BalAcc 0.755, F1 0.859, AUC 0.832, Prec 0.763, Rec 0.982, Thresh 0.456 | LR: 0.000043
Epoch 21: Train: Loss 0.0473, Acc 0.667 | Val: Loss 0.0810, Acc 0.767, BalAcc 0.727, F1 0.827, AUC 0.804, Prec 0.754, Rec 0.916, Thresh 0.528 | LR: 0.000035
Epoch 22: Train: Loss 0.0506, Acc 0.663 | Val: Loss 0.0707, Acc 0.796, BalAcc 0.752, F1 0.851, AUC 0.831, Prec 0.766, Rec 0.958, Thresh 0.496 | LR: 0.000028
 ⚠️ Early stopping after 22 epochs (no improvement for 7 epochs)
✅ Best model saved: ./output/models/best_model_fold_3.pth (Val Loss: 0.0573, BalAcc: 0.784)
⚡️ Test Results: Acc 0.826, BalAcc 0.807, F1 0.867, AUC 0.914, Precision 0.867, Recall 0.867 (threshold: 0.464)
📊 Confusion Matrix (Fold 3):
   [[TN: 115, FP:  39]
    [FN:  39, TP: 254]]
⚡ Avg Inference Time: 0.0139s per sample
📌 Final Magnification Importance (Fold 3): {'40': 0.26189228892326355, '100': 0.2298133224248886, '200': 0.25741419196128845, '400': 0.2508801817893982}
💾 Results saved to: ./output/results/fold_3_results.json

📊 Generating GradCAM visualizations for fold 3...
✅ Generated 5 GradCAM visualizations for fold 3

===== Fold 4 =====
Train patients: 51, Val Patients: 15, Test patients: 16
📊 [Sampling] Class weights: {1: 0.6891891891891891, 0: 1.8214285714285714}
📊 [Sampling] Subtype weights: {'mucinous_carcinoma': 1.0625, 'ductal_carcinoma': 0.265625, 'fibroadenoma': 1.0625, 'papillary_carcinoma': 1.59375, 'tubular_adenoma': 1.59375, 'adenosis': 3.1875, 'lobular_carcinoma': 2.125, 'phyllodes_tumor': 3.1875}
📊 [Sampling] Weight cap applied at 3.0× median (0.732).
Training samples per epoch: {'total_samples_per_epoch': 2043, 'class_distribution': {1: 1455, 0: 588}, 'subtype_distribution': {'mucinous_carcinoma': 249, 'ductal_carcinoma': 894, 'fibroadenoma': 249, 'papillary_carcinoma': 177, 'tubular_adenoma': 171, 'adenosis': 84, 'lobular_carcinoma': 135, 'phyllodes_tumor': 84}, 'oversampling_factor': 2.15}
Validation samples: 360, Test samples: 468
Patients with full 4 mags: 51
Inner training samples: 2043, batch size: 64
Epoch 01: Train: Loss 0.0902, Acc 0.425 | Val: Loss 0.0764, Acc 0.658, BalAcc 0.500, F1 0.794, AUC 0.539, Prec 0.658, Rec 1.000, Thresh 0.100 | LR: 0.000091
 ✅ New best validation loss: 0.0764 (BalAcc: 0.500, threshold: 0.100)
 📊 Mag Importance (Val Loss: 0.0764): {'40': 0.2597561180591583, '100': 0.23629143834114075, '200': 0.26006877422332764, '400': 0.24388372898101807}
Epoch 02: Train: Loss 0.0760, Acc 0.463 | Val: Loss 0.0678, Acc 0.719, BalAcc 0.646, F1 0.805, AUC 0.741, Prec 0.743, Rec 0.878, Thresh 0.342 | LR: 0.000080
 ✅ New best validation loss: 0.0678 (BalAcc: 0.646, threshold: 0.342)
 📊 Mag Importance (Val Loss: 0.0678): {'40': 0.26123908162117004, '100': 0.23890350759029388, '200': 0.2575424909591675, '400': 0.2423149049282074}
Epoch 03: Train: Loss 0.0681, Acc 0.526 | Val: Loss 0.0677, Acc 0.750, BalAcc 0.679, F1 0.826, AUC 0.774, Prec 0.762, Rec 0.903, Thresh 0.286 | LR: 0.000066
 ✅ New best validation loss: 0.0677 (BalAcc: 0.679, threshold: 0.286)
 📊 Mag Importance (Val Loss: 0.0677): {'40': 0.26232850551605225, '100': 0.23999805748462677, '200': 0.2558348476886749, '400': 0.24183858931064606}
Epoch 04: Train: Loss 0.0632, Acc 0.534 | Val: Loss 0.0601, Acc 0.769, BalAcc 0.729, F1 0.830, AUC 0.822, Prec 0.806, Rec 0.857, Thresh 0.334 | LR: 0.000051
 ✅ New best validation loss: 0.0601 (BalAcc: 0.729, threshold: 0.334)
 📊 Mag Importance (Val Loss: 0.0601): {'40': 0.2631933391094208, '100': 0.23983505368232727, '200': 0.2555546462535858, '400': 0.24141693115234375}
Epoch 05: Train: Loss 0.0616, Acc 0.562 | Val: Loss 0.0664, Acc 0.731, BalAcc 0.653, F1 0.815, AUC 0.769, Prec 0.745, Rec 0.899, Thresh 0.302 | LR: 0.000036
Epoch 06: Train: Loss 0.0578, Acc 0.577 | Val: Loss 0.0601, Acc 0.747, BalAcc 0.675, F1 0.825, AUC 0.815, Prec 0.759, Rec 0.903, Thresh 0.334 | LR: 0.000022
Epoch 07: Train: Loss 0.0600, Acc 0.593 | Val: Loss 0.0614, Acc 0.783, BalAcc 0.765, F1 0.833, AUC 0.824, Prec 0.844, Rec 0.823, Thresh 0.407 | LR: 0.000011
Epoch 08: Train: Loss 0.0563, Acc 0.615 | Val: Loss 0.0629, Acc 0.767, BalAcc 0.702, F1 0.837, AUC 0.820, Prec 0.776, Rec 0.907, Thresh 0.302 | LR: 0.000004
Epoch 09: Train: Loss 0.0587, Acc 0.589 | Val: Loss 0.0637, Acc 0.761, BalAcc 0.691, F1 0.834, AUC 0.817, Prec 0.769, Rec 0.911, Thresh 0.294 | LR: 0.000001
Epoch 10: Train: Loss 0.0576, Acc 0.617 | Val: Loss 0.0642, Acc 0.786, BalAcc 0.734, F1 0.847, AUC 0.830, Prec 0.801, Rec 0.899, Thresh 0.278 | LR: 0.000099
Epoch 11: Train: Loss 0.0559, Acc 0.669 | Val: Loss 0.0626, Acc 0.753, BalAcc 0.707, F1 0.819, AUC 0.809, Prec 0.789, Rec 0.852, Thresh 0.391 | LR: 0.000098
 ⚠️ Early stopping after 11 epochs (no improvement for 7 epochs)
✅ Best model saved: ./output/models/best_model_fold_4.pth (Val Loss: 0.0601, BalAcc: 0.729)
⚡️ Test Results: Acc 0.741, BalAcc 0.548, F1 0.845, AUC 0.633, Precision 0.762, Recall 0.948 (threshold: 0.334)
📊 Confusion Matrix (Fold 4):
   [[TN:  18, FP: 103]
    [FN:  18, TP: 329]]
⚡ Avg Inference Time: 0.0136s per sample
📌 Final Magnification Importance (Fold 4): {'40': 0.2662070691585541, '100': 0.24169866740703583, '200': 0.2516587972640991, '400': 0.2404354065656662}
💾 Results saved to: ./output/results/fold_4_results.json

📊 Generating GradCAM visualizations for fold 4...
✅ Generated 5 GradCAM visualizations for fold 4

=== Cross-Validation Results ===
Acc:      0.832 ± 0.092
BalAcc:   0.775 ± 0.142
F1:       0.882 ± 0.067
AUC:      0.869 ± 0.129
Precision: 0.859 ± 0.097
Recall:    0.911 ± 0.059