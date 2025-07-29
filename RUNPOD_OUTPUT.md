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
model.safetensors: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 21.4M/21.4M [00:00<00:00, 156MB/s]
Epoch 01: Train: Loss 0.0966, Acc 0.611 | Val: Loss 0.1040, Acc 0.660, BalAcc 0.555, F1 0.771, AUC 0.562, Prec 0.723, Rec 0.826, Thresh 0.500 | LR: 0.000100
✅ New best validation balanced accuracy: 0.555, threshold: 0.500
📊 Mag Importance (Val BalAcc: 0.555): {'40': 0.06619323045015335, '100': 0.13132813572883606, '200': 0.2964969575405121, '400': 0.5059817433357239}
Epoch 02: Train: Loss 0.0850, Acc 0.615 | Val: Loss 0.0882, Acc 0.631, BalAcc 0.553, F1 0.739, AUC 0.567, Prec 0.724, Rec 0.754, Thresh 0.500 | LR: 0.000100
Epoch 03: Train: Loss 0.0672, Acc 0.659 | Val: Loss 0.0851, Acc 0.656, BalAcc 0.548, F1 0.769, AUC 0.574, Prec 0.719, Rec 0.826, Thresh 0.500 | LR: 0.000100
Epoch 04: Train: Loss 0.0625, Acc 0.661 | Val: Loss 0.0736, Acc 0.747, BalAcc 0.610, F1 0.841, AUC 0.690, Prec 0.745, Rec 0.964, Thresh 0.421 | LR: 0.000100
✅ New best validation balanced accuracy: 0.610, threshold: 0.421
📊 Mag Importance (Val BalAcc: 0.610): {'40': 0.06547711789608002, '100': 0.09929078817367554, '200': 0.41737642884254456, '400': 0.41785573959350586}
Epoch 05: Train: Loss 0.0484, Acc 0.686 | Val: Loss 0.0697, Acc 0.751, BalAcc 0.613, F1 0.844, AUC 0.693, Prec 0.747, Rec 0.970, Thresh 0.363 | LR: 0.000100
✅ New best validation balanced accuracy: 0.613, threshold: 0.363
📊 Mag Importance (Val BalAcc: 0.613): {'40': 0.07761164754629135, '100': 0.12486991286277771, '200': 0.3793032169342041, '400': 0.41821521520614624}
Epoch 06: Train: Loss 0.0464, Acc 0.679 | Val: Loss 0.0568, Acc 0.784, BalAcc 0.652, F1 0.865, AUC 0.748, Prec 0.765, Rec 0.994, Thresh 0.374 | LR: 0.000100
✅ New best validation balanced accuracy: 0.652, threshold: 0.374
📊 Mag Importance (Val BalAcc: 0.652): {'40': 0.07867787033319473, '100': 0.14785540103912354, '200': 0.35483935475349426, '400': 0.4186273515224457}
Epoch 07: Train: Loss 0.0410, Acc 0.715 | Val: Loss 0.0614, Acc 0.788, BalAcc 0.686, F1 0.862, AUC 0.767, Prec 0.787, Rec 0.952, Thresh 0.490 | LR: 0.000100
✅ New best validation balanced accuracy: 0.686, threshold: 0.490
📊 Mag Importance (Val BalAcc: 0.686): {'40': 0.06287005543708801, '100': 0.12718608975410461, '200': 0.31928879022598267, '400': 0.4906550347805023}
Epoch 08: Train: Loss 0.0319, Acc 0.754 | Val: Loss 0.0456, Acc 0.826, BalAcc 0.769, F1 0.879, AUC 0.816, Prec 0.845, Rec 0.916, Thresh 0.545 | LR: 0.000100
✅ New best validation balanced accuracy: 0.769, threshold: 0.545
📊 Mag Importance (Val BalAcc: 0.769): {'40': 0.06373566389083862, '100': 0.10406415164470673, '200': 0.34079059958457947, '400': 0.49140965938568115}
Epoch 09: Train: Loss 0.0269, Acc 0.763 | Val: Loss 0.0370, Acc 0.846, BalAcc 0.788, F1 0.895, AUC 0.867, Prec 0.853, Rec 0.940, Thresh 0.512 | LR: 0.000100
✅ New best validation balanced accuracy: 0.788, threshold: 0.512
📊 Mag Importance (Val BalAcc: 0.788): {'40': 0.07012535631656647, '100': 0.13798829913139343, '200': 0.32216453552246094, '400': 0.46972185373306274}
Epoch 10: Train: Loss 0.0239, Acc 0.763 | Val: Loss 0.0272, Acc 0.867, BalAcc 0.810, F1 0.909, AUC 0.909, Prec 0.865, Rec 0.958, Thresh 0.515 | LR: 0.000100
✅ New best validation balanced accuracy: 0.810, threshold: 0.515
📊 Mag Importance (Val BalAcc: 0.810): {'40': 0.06204863637685776, '100': 0.12087423354387283, '200': 0.4119264781475067, '400': 0.4051506817340851}
Epoch 11: Train: Loss 0.0283, Acc 0.763 | Val: Loss 0.0331, Acc 0.867, BalAcc 0.788, F1 0.912, AUC 0.882, Prec 0.843, Rec 0.994, Thresh 0.505 | LR: 0.000100
Epoch 12: Train: Loss 0.0260, Acc 0.758 | Val: Loss 0.0254, Acc 0.871, BalAcc 0.794, F1 0.915, AUC 0.902, Prec 0.847, Rec 0.994, Thresh 0.443 | LR: 0.000100
Epoch 13: Train: Loss 0.0215, Acc 0.779 | Val: Loss 0.0356, Acc 0.888, BalAcc 0.836, F1 0.923, AUC 0.876, Prec 0.880, Rec 0.970, Thresh 0.458 | LR: 0.000100
✅ New best validation balanced accuracy: 0.836, threshold: 0.458
📊 Mag Importance (Val BalAcc: 0.836): {'40': 0.07226154208183289, '100': 0.13715799152851105, '200': 0.3829992413520813, '400': 0.4075812101364136}
Epoch 14: Train: Loss 0.0196, Acc 0.790 | Val: Loss 0.0250, Acc 0.888, BalAcc 0.840, F1 0.923, AUC 0.920, Prec 0.885, Rec 0.964, Thresh 0.496 | LR: 0.000100
✅ New best validation balanced accuracy: 0.840, threshold: 0.496
📊 Mag Importance (Val BalAcc: 0.840): {'40': 0.06721628457307816, '100': 0.11631422489881516, '200': 0.3964301347732544, '400': 0.4200392961502075}
Epoch 15: Train: Loss 0.0218, Acc 0.784 | Val: Loss 0.0226, Acc 0.884, BalAcc 0.815, F1 0.922, AUC 0.926, Prec 0.860, Rec 0.994, Thresh 0.472 | LR: 0.000100
Epoch 16: Train: Loss 0.0237, Acc 0.803 | Val: Loss 0.0258, Acc 0.884, BalAcc 0.841, F1 0.919, AUC 0.918, Prec 0.888, Rec 0.952, Thresh 0.490 | LR: 0.000100
✅ New best validation balanced accuracy: 0.841, threshold: 0.490
📊 Mag Importance (Val BalAcc: 0.841): {'40': 0.07738770544528961, '100': 0.11695147305727005, '200': 0.3417322635650635, '400': 0.46392855048179626}
Epoch 17: Train: Loss 0.0201, Acc 0.797 | Val: Loss 0.0242, Acc 0.896, BalAcc 0.872, F1 0.926, AUC 0.941, Prec 0.918, Rec 0.934, Thresh 0.589 | LR: 0.000100
✅ New best validation balanced accuracy: 0.872, threshold: 0.589
📊 Mag Importance (Val BalAcc: 0.872): {'40': 0.07525110989809036, '100': 0.11178193241357803, '200': 0.28866714239120483, '400': 0.5242998003959656}
Epoch 18: Train: Loss 0.0204, Acc 0.780 | Val: Loss 0.0201, Acc 0.896, BalAcc 0.850, F1 0.928, AUC 0.941, Prec 0.890, Rec 0.970, Thresh 0.460 | LR: 0.000100
Epoch 19: Train: Loss 0.0168, Acc 0.821 | Val: Loss 0.0174, Acc 0.892, BalAcc 0.858, F1 0.924, AUC 0.951, Prec 0.903, Rec 0.946, Thresh 0.525 | LR: 0.000100
Epoch 20: Train: Loss 0.0216, Acc 0.780 | Val: Loss 0.0189, Acc 0.888, BalAcc 0.863, F1 0.920, AUC 0.942, Prec 0.912, Rec 0.928, Thresh 0.532 | LR: 0.000100
Epoch 21: Train: Loss 0.0178, Acc 0.820 | Val: Loss 0.0280, Acc 0.892, BalAcc 0.843, F1 0.926, AUC 0.936, Prec 0.885, Rec 0.970, Thresh 0.569 | LR: 0.000050
Epoch 22: Train: Loss 0.0180, Acc 0.809 | Val: Loss 0.0245, Acc 0.871, BalAcc 0.809, F1 0.913, AUC 0.921, Prec 0.862, Rec 0.970, Thresh 0.442 | LR: 0.000050
Epoch 23: Train: Loss 0.0201, Acc 0.792 | Val: Loss 0.0213, Acc 0.884, BalAcc 0.852, F1 0.918, AUC 0.945, Prec 0.902, Rec 0.934, Thresh 0.526 | LR: 0.000050
Epoch 24: Train: Loss 0.0206, Acc 0.781 | Val: Loss 0.0236, Acc 0.884, BalAcc 0.811, F1 0.923, AUC 0.933, Prec 0.856, Rec 1.000, Thresh 0.434 | LR: 0.000050
⚠️ Early stopping after 24 epochs (no improvement for 7 epochs)
✅ Best model saved: ./output/models/best_model_fold_0.pth (Val BalAcc: 0.872)
⚡️ Test Results: Acc 0.856, BalAcc 0.885, F1 0.870, AUC 0.997, Precision 1.000, Recall 0.769 (threshold: 0.589)
📊 Confusion Matrix (Fold 0):
   [[TN: 132, FP:   0]
    [FN:  51, TP: 170]]
⚡ Avg Inference Time: 0.0081s per sample
📌 Final Magnification Importance (Fold 0): {'40': 0.0810704231262207, '100': 0.11381269246339798, '200': 0.24000217020511627, '400': 0.5651147961616516}
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
Epoch 01: Train: Loss 0.1030, Acc 0.570 | Val: Loss 0.0712, Acc 0.651, BalAcc 0.580, F1 0.752, AUC 0.630, Prec 0.733, Rec 0.771, Thresh 0.500 | LR: 0.000100
✅ New best validation balanced accuracy: 0.580, threshold: 0.500
📊 Mag Importance (Val BalAcc: 0.580): {'40': 0.3797079622745514, '100': 0.02676422707736492, '200': 0.11023443192243576, '400': 0.48329341411590576}
Epoch 02: Train: Loss 0.0779, Acc 0.629 | Val: Loss 0.0855, Acc 0.616, BalAcc 0.524, F1 0.733, AUC 0.594, Prec 0.699, Rec 0.771, Thresh 0.500 | LR: 0.000100
Epoch 03: Train: Loss 0.0797, Acc 0.636 | Val: Loss 0.0519, Acc 0.642, BalAcc 0.588, F1 0.737, AUC 0.714, Prec 0.742, Rec 0.732, Thresh 0.500 | LR: 0.000100
✅ New best validation balanced accuracy: 0.588, threshold: 0.500
📊 Mag Importance (Val BalAcc: 0.588): {'40': 0.3723514974117279, '100': 0.025014542043209076, '200': 0.15615631639957428, '400': 0.44647765159606934}
Epoch 04: Train: Loss 0.0618, Acc 0.668 | Val: Loss 0.0467, Acc 0.734, BalAcc 0.622, F1 0.826, AUC 0.752, Prec 0.747, Rec 0.924, Thresh 0.428 | LR: 0.000100
✅ New best validation balanced accuracy: 0.622, threshold: 0.428
📊 Mag Importance (Val BalAcc: 0.622): {'40': 0.37934428453445435, '100': 0.03070245496928692, '200': 0.1754145622253418, '400': 0.4145386815071106}
Epoch 05: Train: Loss 0.0562, Acc 0.662 | Val: Loss 0.0545, Acc 0.721, BalAcc 0.571, F1 0.827, AUC 0.751, Prec 0.718, Rec 0.975, Thresh 0.303 | LR: 0.000100
Epoch 06: Train: Loss 0.0468, Acc 0.686 | Val: Loss 0.0361, Acc 0.795, BalAcc 0.707, F1 0.863, AUC 0.847, Prec 0.796, Rec 0.943, Thresh 0.381 | LR: 0.000100
✅ New best validation balanced accuracy: 0.707, threshold: 0.381
📊 Mag Importance (Val BalAcc: 0.707): {'40': 0.36973440647125244, '100': 0.0382988303899765, '200': 0.12990359961986542, '400': 0.46206316351890564}
Epoch 07: Train: Loss 0.0401, Acc 0.721 | Val: Loss 0.0329, Acc 0.830, BalAcc 0.778, F1 0.881, AUC 0.872, Prec 0.847, Rec 0.917, Thresh 0.471 | LR: 0.000100
✅ New best validation balanced accuracy: 0.778, threshold: 0.471
📊 Mag Importance (Val BalAcc: 0.778): {'40': 0.34511885046958923, '100': 0.045089855790138245, '200': 0.16312414407730103, '400': 0.4466671049594879}
Epoch 08: Train: Loss 0.0348, Acc 0.741 | Val: Loss 0.0261, Acc 0.834, BalAcc 0.811, F1 0.878, AUC 0.903, Prec 0.884, Rec 0.873, Thresh 0.491 | LR: 0.000100
✅ New best validation balanced accuracy: 0.811, threshold: 0.491
📊 Mag Importance (Val BalAcc: 0.811): {'40': 0.3961542546749115, '100': 0.03892076760530472, '200': 0.16134434938430786, '400': 0.4035806357860565}
Epoch 09: Train: Loss 0.0336, Acc 0.759 | Val: Loss 0.0179, Acc 0.891, BalAcc 0.894, F1 0.917, AUC 0.952, Prec 0.952, Rec 0.885, Thresh 0.521 | LR: 0.000100
✅ New best validation balanced accuracy: 0.894, threshold: 0.521
📊 Mag Importance (Val BalAcc: 0.894): {'40': 0.40894660353660583, '100': 0.05366530641913414, '200': 0.12310615926980972, '400': 0.4142819941043854}
Epoch 10: Train: Loss 0.0278, Acc 0.753 | Val: Loss 0.0187, Acc 0.886, BalAcc 0.853, F1 0.919, AUC 0.948, Prec 0.897, Rec 0.943, Thresh 0.463 | LR: 0.000100
Epoch 11: Train: Loss 0.0250, Acc 0.785 | Val: Loss 0.0163, Acc 0.934, BalAcc 0.915, F1 0.953, AUC 0.973, Prec 0.938, Rec 0.968, Thresh 0.436 | LR: 0.000100
✅ New best validation balanced accuracy: 0.915, threshold: 0.436
📊 Mag Importance (Val BalAcc: 0.915): {'40': 0.36382877826690674, '100': 0.06000328063964844, '200': 0.16971540451049805, '400': 0.40645262598991394}
Epoch 12: Train: Loss 0.0277, Acc 0.737 | Val: Loss 0.0214, Acc 0.895, BalAcc 0.860, F1 0.926, AUC 0.959, Prec 0.898, Rec 0.955, Thresh 0.394 | LR: 0.000100
Epoch 13: Train: Loss 0.0232, Acc 0.781 | Val: Loss 0.0141, Acc 0.934, BalAcc 0.915, F1 0.953, AUC 0.983, Prec 0.938, Rec 0.968, Thresh 0.459 | LR: 0.000100
Epoch 14: Train: Loss 0.0225, Acc 0.790 | Val: Loss 0.0141, Acc 0.948, BalAcc 0.943, F1 0.962, AUC 0.983, Prec 0.968, Rec 0.955, Thresh 0.464 | LR: 0.000100
✅ New best validation balanced accuracy: 0.943, threshold: 0.464
📊 Mag Importance (Val BalAcc: 0.943): {'40': 0.41201698780059814, '100': 0.060217153280973434, '200': 0.17302024364471436, '400': 0.3547455966472626}
Epoch 15: Train: Loss 0.0211, Acc 0.742 | Val: Loss 0.0152, Acc 0.943, BalAcc 0.936, F1 0.958, AUC 0.983, Prec 0.962, Rec 0.955, Thresh 0.436 | LR: 0.000100
Epoch 16: Train: Loss 0.0192, Acc 0.803 | Val: Loss 0.0105, Acc 0.969, BalAcc 0.974, F1 0.977, AUC 0.992, Prec 0.993, Rec 0.962, Thresh 0.473 | LR: 0.000100
✅ New best validation balanced accuracy: 0.974, threshold: 0.473
📊 Mag Importance (Val BalAcc: 0.974): {'40': 0.3858686685562134, '100': 0.07816915214061737, '200': 0.1549164056777954, '400': 0.3810458481311798}
Epoch 17: Train: Loss 0.0190, Acc 0.806 | Val: Loss 0.0117, Acc 0.965, BalAcc 0.959, F1 0.975, AUC 0.993, Prec 0.975, Rec 0.975, Thresh 0.449 | LR: 0.000100
Epoch 18: Train: Loss 0.0192, Acc 0.799 | Val: Loss 0.0101, Acc 0.969, BalAcc 0.959, F1 0.978, AUC 0.996, Prec 0.969, Rec 0.987, Thresh 0.424 | LR: 0.000100
Epoch 19: Train: Loss 0.0188, Acc 0.801 | Val: Loss 0.0086, Acc 0.983, BalAcc 0.980, F1 0.987, AUC 0.996, Prec 0.987, Rec 0.987, Thresh 0.521 | LR: 0.000100
✅ New best validation balanced accuracy: 0.980, threshold: 0.521
📊 Mag Importance (Val BalAcc: 0.980): {'40': 0.35024166107177734, '100': 0.06888667494058609, '200': 0.155826136469841, '400': 0.42504560947418213}
Epoch 20: Train: Loss 0.0191, Acc 0.774 | Val: Loss 0.0114, Acc 0.965, BalAcc 0.971, F1 0.974, AUC 0.993, Prec 0.993, Rec 0.955, Thresh 0.468 | LR: 0.000100
Epoch 21: Train: Loss 0.0223, Acc 0.815 | Val: Loss 0.0150, Acc 0.956, BalAcc 0.949, F1 0.968, AUC 0.985, Prec 0.968, Rec 0.968, Thresh 0.402 | LR: 0.000100
Epoch 22: Train: Loss 0.0197, Acc 0.807 | Val: Loss 0.0102, Acc 0.969, BalAcc 0.963, F1 0.978, AUC 0.994, Prec 0.975, Rec 0.981, Thresh 0.429 | LR: 0.000100
Epoch 23: Train: Loss 0.0159, Acc 0.797 | Val: Loss 0.0101, Acc 0.956, BalAcc 0.946, F1 0.968, AUC 0.988, Prec 0.962, Rec 0.975, Thresh 0.408 | LR: 0.000050
Epoch 24: Train: Loss 0.0172, Acc 0.832 | Val: Loss 0.0123, Acc 0.969, BalAcc 0.966, F1 0.978, AUC 0.993, Prec 0.981, Rec 0.975, Thresh 0.390 | LR: 0.000050
Epoch 25: Train: Loss 0.0181, Acc 0.787 | Val: Loss 0.0139, Acc 0.952, BalAcc 0.950, F1 0.965, AUC 0.981, Prec 0.974, Rec 0.955, Thresh 0.441 | LR: 0.000050
✅ Best model saved: ./output/models/best_model_fold_1.pth (Val BalAcc: 0.980)
⚡️ Test Results: Acc 0.825, BalAcc 0.867, F1 0.850, AUC 0.976, Precision 0.994, Recall 0.742 (threshold: 0.521)
📊 Confusion Matrix (Fold 1):
   [[TN: 118, FP:   1]
    [FN:  62, TP: 178]]
⚡ Avg Inference Time: 0.0080s per sample
📌 Final Magnification Importance (Fold 1): {'40': 0.3331405818462372, '100': 0.0758974701166153, '200': 0.13156306743621826, '400': 0.45939886569976807}
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
Epoch 01: Train: Loss 0.1058, Acc 0.551 | Val: Loss 0.0901, Acc 0.641, BalAcc 0.524, F1 0.760, AUC 0.557, Prec 0.756, Rec 0.764, Thresh 0.500 | LR: 0.000100
✅ New best validation balanced accuracy: 0.524, threshold: 0.500
📊 Mag Importance (Val BalAcc: 0.524): {'40': 0.17826373875141144, '100': 0.3256174325942993, '200': 0.1366681158542633, '400': 0.35945069789886475}
Epoch 02: Train: Loss 0.0837, Acc 0.606 | Val: Loss 0.0562, Acc 0.676, BalAcc 0.586, F1 0.779, AUC 0.672, Prec 0.789, Rec 0.769, Thresh 0.500 | LR: 0.000100
✅ New best validation balanced accuracy: 0.586, threshold: 0.500
📊 Mag Importance (Val BalAcc: 0.586): {'40': 0.16819435358047485, '100': 0.32572823762893677, '200': 0.14930859208106995, '400': 0.3567688465118408}