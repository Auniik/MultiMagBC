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
Epoch 01: Train: Loss 0.0961, Acc 0.570 | Val: Loss 0.0979, Acc 0.734, BalAcc 0.598, F1 0.832, AUC 0.710, Prec 0.740, Rec 0.952, Thresh 0.313 | LR: 0.000100
✅ New best validation balanced accuracy: 0.598, threshold: 0.313
📊 Mag Importance (Val BalAcc: 0.598): {'40': 0.7886618375778198, '100': 0.05241680145263672, '200': 0.01457051932811737, '400': 0.1443508416414261}
Epoch 02: Train: Loss 0.0618, Acc 0.688 | Val: Loss 0.0781, Acc 0.801, BalAcc 0.725, F1 0.865, AUC 0.791, Prec 0.815, Rec 0.922, Thresh 0.450 | LR: 0.000100
✅ New best validation balanced accuracy: 0.725, threshold: 0.450
📊 Mag Importance (Val BalAcc: 0.725): {'40': 0.7314907312393188, '100': 0.07717277854681015, '200': 0.08339222520589828, '400': 0.10794424265623093}
Val :   6%|█████████▏                                                                                                                                        | 1/16 [00:00<00:11,  1.30it/s]⚠️ NaN detected in batch 2 probs. Replacing with 0.5.
Val :  38%|██████████████████████████████████████████████████████▊                                                                                           | 6/16 [00:01<00:01,  7.75it/s]⚠️ NaN detected in batch 8 probs. Replacing with 0.5.
Epoch 03: Train: Loss 0.0522, Acc 0.736 | Val: Loss nan, Acc 0.830, BalAcc 0.761, F1 0.885, AUC 0.827, Prec 0.835, Rec 0.940, Thresh 0.423 | LR: 0.000100
✅ New best validation balanced accuracy: 0.761, threshold: 0.423
📊 Mag Importance (Val BalAcc: 0.761): {'40': 0.7769832611083984, '100': 0.07336964458227158, '200': 0.051577214151620865, '400': 0.09806984663009644}
Val :   6%|█████████▏                                                                                                                                        | 1/16 [00:00<00:10,  1.38it/s]⚠️ NaN detected in batch 2 probs. Replacing with 0.5.
Val :  38%|██████████████████████████████████████████████████████▊                                                                                           | 6/16 [00:01<00:01,  8.24it/s]⚠️ NaN detected in batch 8 probs. Replacing with 0.5.
Epoch 04: Train: Loss 0.0404, Acc 0.718 | Val: Loss nan, Acc 0.851, BalAcc 0.794, F1 0.897, AUC 0.872, Prec 0.858, Rec 0.940, Thresh 0.443 | LR: 0.000100
✅ New best validation balanced accuracy: 0.794, threshold: 0.443
📊 Mag Importance (Val BalAcc: 0.794): {'40': 0.8032637238502502, '100': 0.05627302825450897, '200': 0.04874110594391823, '400': 0.0917220339179039}
Epoch 05: Train: Loss 0.0377, Acc 0.760 | Val: Loss 0.0345, Acc 0.871, BalAcc 0.851, F1 0.907, AUC 0.906, Prec 0.910, Rec 0.904, Thresh 0.536 | LR: 0.000100
✅ New best validation balanced accuracy: 0.851, threshold: 0.536
📊 Mag Importance (Val BalAcc: 0.851): {'40': 0.7927157878875732, '100': 0.06776010245084763, '200': 0.05009440332651138, '400': 0.08942969143390656}
Epoch 06: Train: Loss 0.0366, Acc 0.739 | Val: Loss 0.0351, Acc 0.851, BalAcc 0.794, F1 0.897, AUC 0.884, Prec 0.858, Rec 0.940, Thresh 0.502 | LR: 0.000100
Epoch 07: Train: Loss 0.0341, Acc 0.763 | Val: Loss 0.0361, Acc 0.871, BalAcc 0.813, F1 0.912, AUC 0.890, Prec 0.866, Rec 0.964, Thresh 0.495 | LR: 0.000100
Epoch 08: Train: Loss 0.0300, Acc 0.780 | Val: Loss 0.0364, Acc 0.855, BalAcc 0.786, F1 0.902, AUC 0.890, Prec 0.847, Rec 0.964, Thresh 0.436 | LR: 0.000100
Epoch 09: Train: Loss 0.0276, Acc 0.762 | Val: Loss 0.0180, Acc 0.913, BalAcc 0.888, F1 0.938, AUC 0.960, Prec 0.924, Rec 0.952, Thresh 0.541 | LR: 0.000100
✅ New best validation balanced accuracy: 0.888, threshold: 0.541
📊 Mag Importance (Val BalAcc: 0.888): {'40': 0.6898437738418579, '100': 0.1129608079791069, '200': 0.06467420607805252, '400': 0.13252118229866028}
Epoch 10: Train: Loss 0.0237, Acc 0.788 | Val: Loss 0.0198, Acc 0.896, BalAcc 0.869, F1 0.926, AUC 0.950, Prec 0.913, Rec 0.940, Thresh 0.516 | LR: 0.000100
Epoch 11: Train: Loss 0.0292, Acc 0.790 | Val: Loss 0.0204, Acc 0.888, BalAcc 0.859, F1 0.920, AUC 0.945, Prec 0.907, Rec 0.934, Thresh 0.494 | LR: 0.000100
Epoch 12: Train: Loss 0.0324, Acc 0.780 | Val: Loss 0.0176, Acc 0.909, BalAcc 0.897, F1 0.934, AUC 0.955, Prec 0.939, Rec 0.928, Thresh 0.549 | LR: 0.000100
✅ New best validation balanced accuracy: 0.897, threshold: 0.549
📊 Mag Importance (Val BalAcc: 0.897): {'40': 0.6628205180168152, '100': 0.1461772471666336, '200': 0.09153512865304947, '400': 0.09946711361408234}
Epoch 13: Train: Loss 0.0235, Acc 0.780 | Val: Loss 0.0149, Acc 0.942, BalAcc 0.936, F1 0.958, AUC 0.971, Prec 0.964, Rec 0.952, Thresh 0.473 | LR: 0.000100
✅ New best validation balanced accuracy: 0.936, threshold: 0.473
📊 Mag Importance (Val BalAcc: 0.936): {'40': 0.6760176420211792, '100': 0.12652643024921417, '200': 0.0905834287405014, '400': 0.10687259584665298}
Epoch 14: Train: Loss 0.0246, Acc 0.785 | Val: Loss 0.0344, Acc 0.921, BalAcc 0.894, F1 0.944, AUC 0.928, Prec 0.925, Rec 0.964, Thresh 0.457 | LR: 0.000100
Epoch 15: Train: Loss 0.0283, Acc 0.776 | Val: Loss 0.0511, Acc 0.884, BalAcc 0.818, F1 0.922, AUC 0.873, Prec 0.864, Rec 0.988, Thresh 0.407 | LR: 0.000100
Epoch 16: Train: Loss 0.0258, Acc 0.828 | Val: Loss 0.0392, Acc 0.851, BalAcc 0.764, F1 0.902, AUC 0.897, Prec 0.829, Rec 0.988, Thresh 0.426 | LR: 0.000100
Epoch 17: Train: Loss 0.0287, Acc 0.783 | Val: Loss 0.0209, Acc 0.929, BalAcc 0.904, F1 0.950, AUC 0.954, Prec 0.931, Rec 0.970, Thresh 0.555 | LR: 0.000050
Epoch 18: Train: Loss 0.0256, Acc 0.798 | Val: Loss 0.0309, Acc 0.867, BalAcc 0.806, F1 0.910, AUC 0.899, Prec 0.861, Rec 0.964, Thresh 0.384 | LR: 0.000050
Epoch 19: Train: Loss 0.0226, Acc 0.801 | Val: Loss 0.0324, Acc 0.876, BalAcc 0.824, F1 0.914, AUC 0.915, Prec 0.874, Rec 0.958, Thresh 0.456 | LR: 0.000050
Epoch 20: Train: Loss 0.0258, Acc 0.799 | Val: Loss 0.1570, Acc 0.884, BalAcc 0.830, F1 0.920, AUC 0.814, Prec 0.876, Rec 0.970, Thresh 0.471 | LR: 0.000050
⚠️ Early stopping after 20 epochs (no improvement for 7 epochs)
✅ Best model saved: ./output/models/best_model_fold_0.pth (Val BalAcc: 0.936)
⚡️ Test Results: Acc 0.935, BalAcc 0.945, F1 0.946, AUC 0.989, Precision 0.990, Recall 0.905 (threshold: 0.473)
📊 Confusion Matrix (Fold 0):
   [[TN: 130, FP:   2]
    [FN:  21, TP: 200]]
⚡ Avg Inference Time: 0.0086s per sample
📌 Final Magnification Importance (Fold 0): {'40': 0.6460239887237549, '100': 0.21022509038448334, '200': 0.08899907767772675, '400': 0.05475180968642235}
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
Epoch 01: Train: Loss 0.1073, Acc 0.569 | Val: Loss 0.0711, Acc 0.721, BalAcc 0.574, F1 0.826, AUC 0.731, Prec 0.720, Rec 0.968, Thresh 0.357 | LR: 0.000100
✅ New best validation balanced accuracy: 0.574, threshold: 0.357
📊 Mag Importance (Val BalAcc: 0.574): {'40': 0.3538089990615845, '100': 0.13462717831134796, '200': 0.4824298024177551, '400': 0.02913404069840908}
Epoch 02: Train: Loss 0.0708, Acc 0.687 | Val: Loss 0.0464, Acc 0.817, BalAcc 0.742, F1 0.876, AUC 0.826, Prec 0.818, Rec 0.943, Thresh 0.434 | LR: 0.000100
✅ New best validation balanced accuracy: 0.742, threshold: 0.434
📊 Mag Importance (Val BalAcc: 0.742): {'40': 0.5533182621002197, '100': 0.15573453903198242, '200': 0.2543138265609741, '400': 0.03663339093327522}
Epoch 03: Train: Loss 0.0537, Acc 0.712 | Val: Loss 0.0294, Acc 0.873, BalAcc 0.825, F1 0.912, AUC 0.910, Prec 0.872, Rec 0.955, Thresh 0.382 | LR: 0.000100
✅ New best validation balanced accuracy: 0.825, threshold: 0.382
📊 Mag Importance (Val BalAcc: 0.825): {'40': 0.47166404128074646, '100': 0.12323801964521408, '200': 0.3571182191371918, '400': 0.047979772090911865}
Epoch 04: Train: Loss 0.0475, Acc 0.708 | Val: Loss 0.0218, Acc 0.904, BalAcc 0.892, F1 0.929, AUC 0.956, Prec 0.935, Rec 0.924, Thresh 0.450 | LR: 0.000100
✅ New best validation balanced accuracy: 0.892, threshold: 0.450
📊 Mag Importance (Val BalAcc: 0.892): {'40': 0.42795777320861816, '100': 0.12618109583854675, '200': 0.4008091986179352, '400': 0.045051928609609604}
Epoch 05: Train: Loss 0.0406, Acc 0.727 | Val: Loss 0.0185, Acc 0.921, BalAcc 0.905, F1 0.943, AUC 0.963, Prec 0.937, Rec 0.949, Thresh 0.405 | LR: 0.000100
✅ New best validation balanced accuracy: 0.905, threshold: 0.405
📊 Mag Importance (Val BalAcc: 0.905): {'40': 0.42993682622909546, '100': 0.12361675500869751, '200': 0.3997326195240021, '400': 0.04671372100710869}
Epoch 06: Train: Loss 0.0331, Acc 0.763 | Val: Loss 0.0240, Acc 0.878, BalAcc 0.836, F1 0.914, AUC 0.936, Prec 0.882, Rec 0.949, Thresh 0.360 | LR: 0.000100
Epoch 07: Train: Loss 0.0363, Acc 0.760 | Val: Loss 0.0335, Acc 0.904, BalAcc 0.900, F1 0.929, AUC 0.949, Prec 0.947, Rec 0.911, Thresh 0.393 | LR: 0.000100
Epoch 08: Train: Loss 0.0322, Acc 0.758 | Val: Loss 0.0197, Acc 0.900, BalAcc 0.852, F1 0.931, AUC 0.958, Prec 0.885, Rec 0.981, Thresh 0.338 | LR: 0.000100
Epoch 09: Train: Loss 0.0282, Acc 0.750 | Val: Loss 0.0119, Acc 0.948, BalAcc 0.924, F1 0.963, AUC 0.988, Prec 0.939, Rec 0.987, Thresh 0.384 | LR: 0.000100
✅ New best validation balanced accuracy: 0.924, threshold: 0.384
📊 Mag Importance (Val BalAcc: 0.924): {'40': 0.4043266475200653, '100': 0.11442188918590546, '200': 0.4152531623840332, '400': 0.06599835306406021}
Epoch 10: Train: Loss 0.0280, Acc 0.796 | Val: Loss 0.0128, Acc 0.965, BalAcc 0.959, F1 0.975, AUC 0.991, Prec 0.975, Rec 0.975, Thresh 0.437 | LR: 0.000100
✅ New best validation balanced accuracy: 0.959, threshold: 0.437
📊 Mag Importance (Val BalAcc: 0.959): {'40': 0.3831024467945099, '100': 0.13650648295879364, '200': 0.4220331609249115, '400': 0.05835794284939766}
Epoch 11: Train: Loss 0.0319, Acc 0.775 | Val: Loss 0.0143, Acc 0.952, BalAcc 0.954, F1 0.964, AUC 0.982, Prec 0.980, Rec 0.949, Thresh 0.462 | LR: 0.000100
Epoch 12: Train: Loss 0.0274, Acc 0.788 | Val: Loss 0.0080, Acc 0.974, BalAcc 0.962, F1 0.981, AUC 0.997, Prec 0.969, Rec 0.994, Thresh 0.403 | LR: 0.000100
✅ New best validation balanced accuracy: 0.962, threshold: 0.403
📊 Mag Importance (Val BalAcc: 0.962): {'40': 0.5026488900184631, '100': 0.11895547062158585, '200': 0.29547378420829773, '400': 0.08292187005281448}
Epoch 13: Train: Loss 0.0296, Acc 0.777 | Val: Loss 0.0105, Acc 0.961, BalAcc 0.968, F1 0.971, AUC 0.990, Prec 0.993, Rec 0.949, Thresh 0.502 | LR: 0.000100
✅ New best validation balanced accuracy: 0.968, threshold: 0.502
📊 Mag Importance (Val BalAcc: 0.968): {'40': 0.5056548714637756, '100': 0.12982772290706635, '200': 0.2612549960613251, '400': 0.10326237976551056}
Epoch 14: Train: Loss 0.0230, Acc 0.775 | Val: Loss 0.0132, Acc 0.969, BalAcc 0.978, F1 0.977, AUC 0.979, Prec 1.000, Rec 0.955, Thresh 0.459 | LR: 0.000100
✅ New best validation balanced accuracy: 0.978, threshold: 0.459
📊 Mag Importance (Val BalAcc: 0.978): {'40': 0.39140698313713074, '100': 0.21857976913452148, '200': 0.30160579085350037, '400': 0.0884074866771698}
Epoch 15: Train: Loss 0.0258, Acc 0.783 | Val: Loss 0.0099, Acc 0.983, BalAcc 0.984, F1 0.987, AUC 0.997, Prec 0.994, Rec 0.981, Thresh 0.443 | LR: 0.000100
✅ New best validation balanced accuracy: 0.984, threshold: 0.443
📊 Mag Importance (Val BalAcc: 0.984): {'40': 0.35117220878601074, '100': 0.17831780016422272, '200': 0.3634212911128998, '400': 0.10708872228860855}
Epoch 16: Train: Loss 0.0257, Acc 0.738 | Val: Loss 0.0141, Acc 0.939, BalAcc 0.944, F1 0.954, AUC 0.979, Prec 0.980, Rec 0.930, Thresh 0.500 | LR: 0.000100
Epoch 17: Train: Loss 0.0291, Acc 0.772 | Val: Loss 0.0087, Acc 0.996, BalAcc 0.993, F1 0.997, AUC 1.000, Prec 0.994, Rec 1.000, Thresh 0.452 | LR: 0.000100
✅ New best validation balanced accuracy: 0.993, threshold: 0.452
📊 Mag Importance (Val BalAcc: 0.993): {'40': 0.38479962944984436, '100': 0.21355853974819183, '200': 0.3051704466342926, '400': 0.09647134691476822}
Epoch 18: Train: Loss 0.0234, Acc 0.796 | Val: Loss 0.0109, Acc 0.974, BalAcc 0.977, F1 0.981, AUC 0.994, Prec 0.993, Rec 0.968, Thresh 0.404 | LR: 0.000100
Epoch 19: Train: Loss 0.0221, Acc 0.790 | Val: Loss 0.0080, Acc 0.987, BalAcc 0.987, F1 0.990, AUC 0.997, Prec 0.994, Rec 0.987, Thresh 0.453 | LR: 0.000100
Epoch 20: Train: Loss 0.0191, Acc 0.798 | Val: Loss 0.0072, Acc 0.987, BalAcc 0.990, F1 0.990, AUC 0.997, Prec 1.000, Rec 0.981, Thresh 0.507 | LR: 0.000100
Epoch 21: Train: Loss 0.0190, Acc 0.799 | Val: Loss 0.0128, Acc 0.969, BalAcc 0.966, F1 0.978, AUC 0.992, Prec 0.981, Rec 0.975, Thresh 0.385 | LR: 0.000050
Epoch 22: Train: Loss 0.0200, Acc 0.810 | Val: Loss 0.0113, Acc 0.978, BalAcc 0.980, F1 0.984, AUC 0.992, Prec 0.994, Rec 0.975, Thresh 0.384 | LR: 0.000050
Epoch 23: Train: Loss 0.0217, Acc 0.790 | Val: Loss 0.0099, Acc 0.952, BalAcc 0.965, F1 0.964, AUC 0.996, Prec 1.000, Rec 0.930, Thresh 0.500 | LR: 0.000050
Epoch 24: Train: Loss 0.0193, Acc 0.787 | Val: Loss 0.0081, Acc 0.983, BalAcc 0.984, F1 0.987, AUC 0.996, Prec 0.994, Rec 0.981, Thresh 0.403 | LR: 0.000050
⚠️ Early stopping after 24 epochs (no improvement for 7 epochs)
✅ Best model saved: ./output/models/best_model_fold_1.pth (Val BalAcc: 0.993)
⚡️ Test Results: Acc 0.897, BalAcc 0.887, F1 0.922, AUC 0.971, Precision 0.928, Recall 0.917 (threshold: 0.452)
📊 Confusion Matrix (Fold 1):
   [[TN: 102, FP:  17]
    [FN:  20, TP: 220]]
⚡ Avg Inference Time: 0.0098s per sample
📌 Final Magnification Importance (Fold 1): {'40': 0.4482637047767639, '100': 0.23789772391319275, '200': 0.21660171449184418, '400': 0.09723693132400513}
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
Epoch 01: Train: Loss 0.1239, Acc 0.544 | Val: Loss 0.0898, Acc 0.813, BalAcc 0.669, F1 0.885, AUC 0.689, Prec 0.817, Rec 0.964, Thresh 0.341 | LR: 0.000100
✅ New best validation balanced accuracy: 0.669, threshold: 0.341
📊 Mag Importance (Val BalAcc: 0.669): {'40': 0.10476332157850266, '100': 0.4176919758319855, '200': 0.3618573844432831, '400': 0.11568740010261536}
Epoch 02: Train: Loss 0.0783, Acc 0.636 | Val: Loss 0.0261, Acc 0.859, BalAcc 0.758, F1 0.910, AUC 0.883, Prec 0.862, Rec 0.964, Thresh 0.414 | LR: 0.000100
✅ New best validation balanced accuracy: 0.758, threshold: 0.414
📊 Mag Importance (Val BalAcc: 0.758): {'40': 0.13379113376140594, '100': 0.4350714683532715, '200': 0.33118709921836853, '400': 0.09995034337043762}
Epoch 03: Train: Loss 0.0613, Acc 0.682 | Val: Loss 0.0541, Acc 0.859, BalAcc 0.758, F1 0.910, AUC 0.877, Prec 0.862, Rec 0.964, Thresh 0.331 | LR: 0.000100
Epoch 04: Train: Loss 0.0538, Acc 0.733 | Val: Loss 0.0383, Acc 0.969, BalAcc 0.965, F1 0.979, AUC 0.968, Prec 0.984, Rec 0.974, Thresh 0.490 | LR: 0.000100
✅ New best validation balanced accuracy: 0.965, threshold: 0.490
📊 Mag Importance (Val BalAcc: 0.965): {'40': 0.30711859464645386, '100': 0.3320145905017853, '200': 0.23267219960689545, '400': 0.12819457054138184}
Epoch 05: Train: Loss 0.0495, Acc 0.718 | Val: Loss 0.0629, Acc 0.927, BalAcc 0.878, F1 0.953, AUC 0.925, Prec 0.927, Rec 0.979, Thresh 0.499 | LR: 0.000100
Epoch 06: Train: Loss 0.0415, Acc 0.756 | Val: Loss 0.0259, Acc 0.969, BalAcc 0.945, F1 0.980, AUC 0.971, Prec 0.965, Rec 0.995, Thresh 0.505 | LR: 0.000100
Epoch 07: Train: Loss 0.0373, Acc 0.750 | Val: Loss 0.0740, Acc 0.935, BalAcc 0.883, F1 0.958, AUC 0.886, Prec 0.928, Rec 0.990, Thresh 0.534 | LR: 0.000100
Epoch 08: Train: Loss 0.0362, Acc 0.754 | Val: Loss 0.0107, Acc 0.962, BalAcc 0.940, F1 0.975, AUC 0.985, Prec 0.965, Rec 0.985, Thresh 0.528 | LR: 0.000050
Epoch 09: Train: Loss 0.0367, Acc 0.774 | Val: Loss 0.0394, Acc 0.954, BalAcc 0.920, F1 0.970, AUC 0.942, Prec 0.951, Rec 0.990, Thresh 0.503 | LR: 0.000050
Epoch 10: Train: Loss 0.0285, Acc 0.773 | Val: Loss 0.0394, Acc 0.977, BalAcc 0.955, F1 0.985, AUC 0.967, Prec 0.970, Rec 1.000, Thresh 0.565 | LR: 0.000050
Epoch 11: Train: Loss 0.0343, Acc 0.737 | Val: Loss 0.0122, Acc 0.973, BalAcc 0.953, F1 0.982, AUC 0.968, Prec 0.970, Rec 0.995, Thresh 0.464 | LR: 0.000050
⚠️ Early stopping after 11 epochs (no improvement for 7 epochs)
✅ Best model saved: ./output/models/best_model_fold_2.pth (Val BalAcc: 0.965)
⚡️ Test Results: Acc 0.775, BalAcc 0.763, F1 0.828, AUC 0.816, Precision 0.864, Recall 0.795 (threshold: 0.490)
📊 Confusion Matrix (Fold 2):
   [[TN:  68, FP:  25]
    [FN:  41, TP: 159]]
⚡ Avg Inference Time: 0.0097s per sample
📌 Final Magnification Importance (Fold 2): {'40': 0.2633240222930908, '100': 0.3964027762413025, '200': 0.21865037083625793, '400': 0.12162286043167114}
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
Val :  60%|███████████████████████████████████████████████████████████████████████████████████████▌                                                          | 9/15 [00:01<00:00,  7.34it/s]⚠️ NaN detected in batch 10 probs. Replacing with 0.5.
Epoch 01: Train: Loss 0.0959, Acc 0.599 | Val: Loss nan, Acc 0.651, BalAcc 0.590, F1 0.748, AUC 0.679, Prec 0.732, Rec 0.764, Thresh 0.500 | LR: 0.000100
✅ New best validation balanced accuracy: 0.590, threshold: 0.500
📊 Mag Importance (Val BalAcc: 0.590): {'40': 0.6919451355934143, '100': 0.10335712134838104, '200': 0.17076514661312103, '400': 0.0339326336979866}
Epoch 02: Train: Loss 0.0588, Acc 0.656 | Val: Loss 0.0642, Acc 0.660, BalAcc 0.633, F1 0.738, AUC 0.704, Prec 0.770, Rec 0.708, Thresh 0.500 | LR: 0.000100
✅ New best validation balanced accuracy: 0.633, threshold: 0.500
📊 Mag Importance (Val BalAcc: 0.633): {'40': 0.5789467692375183, '100': 0.2230224907398224, '200': 0.16102388501167297, '400': 0.03700682520866394}
Epoch 03: Train: Loss 0.0486, Acc 0.709 | Val: Loss 0.0314, Acc 0.798, BalAcc 0.715, F1 0.864, AUC 0.881, Prec 0.793, Rec 0.950, Thresh 0.359 | LR: 0.000100
✅ New best validation balanced accuracy: 0.715, threshold: 0.359
📊 Mag Importance (Val BalAcc: 0.715): {'40': 0.560987651348114, '100': 0.18785959482192993, '200': 0.20039625465869904, '400': 0.050756484270095825}
Epoch 04: Train: Loss 0.0418, Acc 0.723 | Val: Loss 0.0383, Acc 0.832, BalAcc 0.828, F1 0.871, AUC 0.890, Prec 0.906, Rec 0.839, Thresh 0.442 | LR: 0.000100
✅ New best validation balanced accuracy: 0.828, threshold: 0.442
📊 Mag Importance (Val BalAcc: 0.828): {'40': 0.4893672466278076, '100': 0.15992148220539093, '200': 0.31431376934051514, '400': 0.036397505551576614}
Epoch 05: Train: Loss 0.0410, Acc 0.731 | Val: Loss 0.0394, Acc 0.874, BalAcc 0.859, F1 0.906, AUC 0.926, Prec 0.912, Rec 0.901, Thresh 0.436 | LR: 0.000100
✅ New best validation balanced accuracy: 0.859, threshold: 0.436
📊 Mag Importance (Val BalAcc: 0.859): {'40': 0.6728293299674988, '100': 0.16543036699295044, '200': 0.1288747936487198, '400': 0.03286558762192726}
Val :   7%|█████████▋                                                                                                                                        | 1/15 [00:00<00:13,  1.06it/s]⚠️ NaN detected in batch 2 probs. Replacing with 0.5.
Val :  33%|████████████████████████████████████████████████▋                                                                                                 | 5/15 [00:01<00:01,  6.07it/s]⚠️ NaN detected in batch 7 probs. Replacing with 0.5.
Val :  60%|███████████████████████████████████████████████████████████████████████████████████████▌                                                          | 9/15 [00:01<00:00,  7.58it/s]⚠️ NaN detected in batch 9 probs. Replacing with 0.5.
Epoch 06: Train: Loss 0.0377, Acc 0.759 | Val: Loss nan, Acc 0.891, BalAcc 0.882, F1 0.918, AUC 0.950, Prec 0.930, Rec 0.907, Thresh 0.477 | LR: 0.000100
✅ New best validation balanced accuracy: 0.882, threshold: 0.477
📊 Mag Importance (Val BalAcc: 0.882): {'40': 0.48825299739837646, '100': 0.25401055812835693, '200': 0.16760915517807007, '400': 0.09012731909751892}
Val :  60%|███████████████████████████████████████████████████████████████████████████████████████▌                                                          | 9/15 [00:01<00:00,  7.58it/s]⚠️ NaN detected in batch 10 probs. Replacing with 0.5.
Epoch 07: Train: Loss 0.0345, Acc 0.756 | Val: Loss nan, Acc 0.954, BalAcc 0.949, F1 0.966, AUC 0.981, Prec 0.969, Rec 0.963, Thresh 0.421 | LR: 0.000100
✅ New best validation balanced accuracy: 0.949, threshold: 0.421
📊 Mag Importance (Val BalAcc: 0.949): {'40': 0.6178871393203735, '100': 0.18805143237113953, '200': 0.10428650677204132, '400': 0.08977489173412323}
Val :  60%|███████████████████████████████████████████████████████████████████████████████████████▌                                                          | 9/15 [00:01<00:00,  7.13it/s]⚠️ NaN detected in batch 10 probs. Replacing with 0.5.
Epoch 08: Train: Loss 0.0320, Acc 0.770 | Val: Loss nan, Acc 0.929, BalAcc 0.923, F1 0.947, AUC 0.967, Prec 0.956, Rec 0.938, Thresh 0.442 | LR: 0.000100
Epoch 09: Train: Loss 0.0289, Acc 0.774 | Val: Loss 0.0130, Acc 0.950, BalAcc 0.949, F1 0.962, AUC 0.987, Prec 0.975, Rec 0.950, Thresh 0.450 | LR: 0.000100
✅ New best validation balanced accuracy: 0.949, threshold: 0.450
📊 Mag Importance (Val BalAcc: 0.949): {'40': 0.7341946959495544, '100': 0.12541121244430542, '200': 0.07737477123737335, '400': 0.06301940977573395}
Epoch 10: Train: Loss 0.0289, Acc 0.781 | Val: Loss 0.0215, Acc 0.929, BalAcc 0.930, F1 0.946, AUC 0.968, Prec 0.968, Rec 0.925, Thresh 0.443 | LR: 0.000100
Epoch 11: Train: Loss 0.0279, Acc 0.794 | Val: Loss 0.0124, Acc 0.966, BalAcc 0.962, F1 0.975, AUC 0.994, Prec 0.975, Rec 0.975, Thresh 0.444 | LR: 0.000100
✅ New best validation balanced accuracy: 0.962, threshold: 0.444
📊 Mag Importance (Val BalAcc: 0.962): {'40': 0.6618638038635254, '100': 0.18473394215106964, '200': 0.07856626808643341, '400': 0.07483602315187454}
Epoch 12: Train: Loss 0.0209, Acc 0.793 | Val: Loss 0.0122, Acc 0.971, BalAcc 0.961, F1 0.978, AUC 0.992, Prec 0.970, Rec 0.988, Thresh 0.433 | LR: 0.000100
Epoch 13: Train: Loss 0.0262, Acc 0.757 | Val: Loss 0.0154, Acc 0.966, BalAcc 0.962, F1 0.975, AUC 0.992, Prec 0.975, Rec 0.975, Thresh 0.390 | LR: 0.000100
Epoch 14: Train: Loss 0.0233, Acc 0.755 | Val: Loss 0.0140, Acc 0.941, BalAcc 0.912, F1 0.958, AUC 0.986, Prec 0.925, Rec 0.994, Thresh 0.431 | LR: 0.000100
Epoch 15: Train: Loss 0.0253, Acc 0.764 | Val: Loss 0.0126, Acc 0.975, BalAcc 0.975, F1 0.981, AUC 0.995, Prec 0.987, Rec 0.975, Thresh 0.419 | LR: 0.000100
✅ New best validation balanced accuracy: 0.975, threshold: 0.419
📊 Mag Importance (Val BalAcc: 0.975): {'40': 0.25483763217926025, '100': 0.35516357421875, '200': 0.20544372498989105, '400': 0.1845550388097763}
Epoch 16: Train: Loss 0.0250, Acc 0.747 | Val: Loss 0.0103, Acc 0.971, BalAcc 0.965, F1 0.978, AUC 0.993, Prec 0.975, Rec 0.981, Thresh 0.498 | LR: 0.000100
Epoch 17: Train: Loss 0.0243, Acc 0.751 | Val: Loss 0.0156, Acc 0.937, BalAcc 0.933, F1 0.953, AUC 0.981, Prec 0.962, Rec 0.944, Thresh 0.408 | LR: 0.000100
Epoch 18: Train: Loss 0.0254, Acc 0.764 | Val: Loss 0.0103, Acc 0.962, BalAcc 0.945, F1 0.973, AUC 0.994, Prec 0.952, Rec 0.994, Thresh 0.461 | LR: 0.000100
Epoch 19: Train: Loss 0.0241, Acc 0.743 | Val: Loss 0.0089, Acc 0.966, BalAcc 0.955, F1 0.975, AUC 0.993, Prec 0.964, Rec 0.988, Thresh 0.497 | LR: 0.000050
Epoch 20: Train: Loss 0.0250, Acc 0.783 | Val: Loss 0.0135, Acc 0.975, BalAcc 0.971, F1 0.981, AUC 0.989, Prec 0.981, Rec 0.981, Thresh 0.406 | LR: 0.000050
Epoch 21: Train: Loss 0.0194, Acc 0.785 | Val: Loss 0.0069, Acc 0.996, BalAcc 0.994, F1 0.997, AUC 1.000, Prec 0.994, Rec 1.000, Thresh 0.460 | LR: 0.000050
✅ New best validation balanced accuracy: 0.994, threshold: 0.460
📊 Mag Importance (Val BalAcc: 0.994): {'40': 0.3036110997200012, '100': 0.38126495480537415, '200': 0.19751963019371033, '400': 0.11760429292917252}
Epoch 22: Train: Loss 0.0219, Acc 0.775 | Val: Loss 0.0089, Acc 0.983, BalAcc 0.981, F1 0.988, AUC 0.999, Prec 0.988, Rec 0.988, Thresh 0.446 | LR: 0.000050
Epoch 23: Train: Loss 0.0224, Acc 0.768 | Val: Loss 0.0068, Acc 0.996, BalAcc 0.997, F1 0.997, AUC 1.000, Prec 1.000, Rec 0.994, Thresh 0.428 | LR: 0.000050
✅ New best validation balanced accuracy: 0.997, threshold: 0.428
📊 Mag Importance (Val BalAcc: 0.997): {'40': 0.33451154828071594, '100': 0.3515428602695465, '200': 0.18454039096832275, '400': 0.1294052004814148}
Epoch 24: Train: Loss 0.0214, Acc 0.791 | Val: Loss 0.0062, Acc 1.000, BalAcc 1.000, F1 1.000, AUC 1.000, Prec 1.000, Rec 1.000, Thresh 0.440 | LR: 0.000050
✅ New best validation balanced accuracy: 1.000, threshold: 0.440
📊 Mag Importance (Val BalAcc: 1.000): {'40': 0.33018121123313904, '100': 0.36346980929374695, '200': 0.1791878342628479, '400': 0.1271611601114273}
Epoch 25: Train: Loss 0.0219, Acc 0.803 | Val: Loss 0.0070, Acc 0.987, BalAcc 0.987, F1 0.991, AUC 0.999, Prec 0.994, Rec 0.988, Thresh 0.435 | LR: 0.000050
✅ Best model saved: ./output/models/best_model_fold_3.pth (Val BalAcc: 1.000)
⚡️ Test Results: Acc 0.873, BalAcc 0.839, F1 0.906, AUC 0.958, Precision 0.862, Recall 0.956 (threshold: 0.440)
📊 Confusion Matrix (Fold 3):
   [[TN:  91, FP:  35]
    [FN:  10, TP: 218]]
⚡ Avg Inference Time: 0.0086s per sample
📌 Final Magnification Importance (Fold 3): {'40': 0.31513580679893494, '100': 0.37484821677207947, '200': 0.1780359148979187, '400': 0.13198009133338928}
💾 Results saved to: ./output/results/fold_3_results.json

📊 Generating GradCAM visualizations for fold 3...
✅ Generated 5 GradCAM visualizations for fold 3

===== Fold 4 =====
Train patients: 52, Val Patients: 14, Test patients: 16
Training samples per epoch: {'total_samples_per_epoch': 2253, 'class_distribution': {0: 693, 1: 1560}, 'oversampling_factor': 2.04}
Validation samples: 255, Test samples: 287
Patients with full 4 mags: 52
Inner training samples: 2253, batch size: 16
Class weights: Benign=1.62, Malignant=0.72
Epoch 01: Train: Loss 0.1110, Acc 0.568 | Val: Loss 0.0903, Acc 0.651, BalAcc 0.579, F1 0.752, AUC 0.602, Prec 0.730, Rec 0.776, Thresh 0.500 | LR: 0.000100
✅ New best validation balanced accuracy: 0.579, threshold: 0.500
📊 Mag Importance (Val BalAcc: 0.579): {'40': 0.0905289500951767, '100': 0.2828667461872101, '200': 0.34035998582839966, '400': 0.28624433279037476}
Epoch 02: Train: Loss 0.0830, Acc 0.620 | Val: Loss 0.0783, Acc 0.631, BalAcc 0.601, F1 0.717, AUC 0.653, Prec 0.753, Rec 0.684, Thresh 0.500 | LR: 0.000100
✅ New best validation balanced accuracy: 0.601, threshold: 0.500
📊 Mag Importance (Val BalAcc: 0.601): {'40': 0.0716475322842598, '100': 0.20162637531757355, '200': 0.33868831396102905, '400': 0.388037770986557}
Epoch 03: Train: Loss 0.0635, Acc 0.685 | Val: Loss 0.0540, Acc 0.804, BalAcc 0.728, F1 0.867, AUC 0.787, Prec 0.807, Rec 0.937, Thresh 0.496 | LR: 0.000100
✅ New best validation balanced accuracy: 0.728, threshold: 0.496
📊 Mag Importance (Val BalAcc: 0.728): {'40': 0.09922683238983154, '100': 0.21792322397232056, '200': 0.31570279598236084, '400': 0.36714720726013184}
Val :   0%|                                                                                                                                                          | 0/16 [00:00<?, ?it/s]⚠️ NaN detected in batch 0 probs. Replacing with 0.5.
Epoch 04: Train: Loss 0.0453, Acc 0.713 | Val: Loss nan, Acc 0.839, BalAcc 0.793, F1 0.886, AUC 0.853, Prec 0.856, Rec 0.920, Thresh 0.506 | LR: 0.000100
✅ New best validation balanced accuracy: 0.793, threshold: 0.506
📊 Mag Importance (Val BalAcc: 0.793): {'40': 0.09096241742372513, '100': 0.23574793338775635, '200': 0.32867273688316345, '400': 0.34461697936058044}
Epoch 05: Train: Loss 0.0403, Acc 0.713 | Val: Loss 0.0375, Acc 0.824, BalAcc 0.739, F1 0.883, AUC 0.885, Prec 0.809, Rec 0.971, Thresh 0.465 | LR: 0.000100
Epoch 06: Train: Loss 0.0350, Acc 0.761 | Val: Loss 0.0353, Acc 0.878, BalAcc 0.838, F1 0.914, AUC 0.914, Prec 0.882, Rec 0.948, Thresh 0.539 | LR: 0.000100
✅ New best validation balanced accuracy: 0.838, threshold: 0.539
📊 Mag Importance (Val BalAcc: 0.838): {'40': 0.17013627290725708, '100': 0.2879641652107239, '200': 0.24537107348442078, '400': 0.29652851819992065}
Epoch 07: Train: Loss 0.0330, Acc 0.754 | Val: Loss 0.0403, Acc 0.871, BalAcc 0.836, F1 0.908, AUC 0.888, Prec 0.885, Rec 0.931, Thresh 0.610 | LR: 0.000100
Epoch 08: Train: Loss 0.0332, Acc 0.771 | Val: Loss 0.0486, Acc 0.831, BalAcc 0.761, F1 0.885, AUC 0.843, Prec 0.826, Rec 0.954, Thresh 0.548 | LR: 0.000100
Epoch 09: Train: Loss 0.0351, Acc 0.773 | Val: Loss 0.0323, Acc 0.839, BalAcc 0.767, F1 0.891, AUC 0.858, Prec 0.828, Rec 0.966, Thresh 0.482 | LR: 0.000100
Epoch 10: Train: Loss 0.0262, Acc 0.758 | Val: Loss 0.0324, Acc 0.863, BalAcc 0.797, F1 0.907, AUC 0.889, Prec 0.846, Rec 0.977, Thresh 0.537 | LR: 0.000050
Epoch 11: Train: Loss 0.0266, Acc 0.787 | Val: Loss 0.0435, Acc 0.882, BalAcc 0.825, F1 0.919, AUC 0.888, Prec 0.864, Rec 0.983, Thresh 0.550 | LR: 0.000050
Epoch 12: Train: Loss 0.0251, Acc 0.769 | Val: Loss 0.0305, Acc 0.847, BalAcc 0.799, F1 0.893, AUC 0.890, Prec 0.857, Rec 0.931, Thresh 0.542 | LR: 0.000050
Epoch 13: Train: Loss 0.0260, Acc 0.772 | Val: Loss 0.0303, Acc 0.863, BalAcc 0.784, F1 0.909, AUC 0.900, Prec 0.833, Rec 1.000, Thresh 0.454 | LR: 0.000050
⚠️ Early stopping after 13 epochs (no improvement for 7 epochs)
✅ Best model saved: ./output/models/best_model_fold_4.pth (Val BalAcc: 0.838)
⚡️ Test Results: Acc 0.909, BalAcc 0.925, F1 0.939, AUC 0.989, Precision 0.985, Recall 0.897 (threshold: 0.539)
📊 Confusion Matrix (Fold 4):
   [[TN:  60, FP:   3]
    [FN:  23, TP: 201]]
⚡ Avg Inference Time: 0.0105s per sample
📌 Final Magnification Importance (Fold 4): {'40': 0.21149152517318726, '100': 0.22279541194438934, '200': 0.24736270308494568, '400': 0.31835034489631653}
💾 Results saved to: ./output/results/fold_4_results.json

📊 Generating GradCAM visualizations for fold 4...
✅ Generated 5 GradCAM visualizations for fold 4

=== Cross-Validation Results ===
Acc:      0.878 ± 0.055
BalAcc:   0.872 ± 0.065
F1:       0.908 ± 0.042
AUC:      0.945 ± 0.065
Precision: 0.926 ± 0.056
Recall:    0.894 ± 0.053