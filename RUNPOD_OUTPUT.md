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
Epoch 01: Train: Loss 0.0268, Acc 0.555 | Val: Loss 0.0322, Acc 0.739, BalAcc 0.623, F1 0.830, AUC 0.705, Prec 0.755, Rec 0.922, Thresh 0.391 | LR: 0.000100
✅ New best validation balanced accuracy: 0.623, threshold: 0.391
📊 Mag Importance (Val BalAcc: 0.623): {'40': 0.25387638807296753, '100': 0.26216810941696167, '200': 0.25178179144859314, '400': 0.23217371106147766}
Epoch 02: Train: Loss 0.0247, Acc 0.620 | Val: Loss 0.0376, Acc 0.739, BalAcc 0.586, F1 0.839, AUC 0.637, Prec 0.732, Rec 0.982, Thresh 0.334 | LR: 0.000100
Epoch 03: Train: Loss 0.0213, Acc 0.640 | Val: Loss 0.0272, Acc 0.784, BalAcc 0.679, F1 0.859, AUC 0.741, Prec 0.783, Rec 0.952, Thresh 0.375 | LR: 0.000100
✅ New best validation balanced accuracy: 0.679, threshold: 0.375
📊 Mag Importance (Val BalAcc: 0.679): {'40': 0.2520669102668762, '100': 0.262614369392395, '200': 0.2530537247657776, '400': 0.2322649508714676}
Epoch 04: Train: Loss 0.0190, Acc 0.663 | Val: Loss 0.0470, Acc 0.768, BalAcc 0.689, F1 0.842, AUC 0.719, Prec 0.797, Rec 0.892, Thresh 0.456 | LR: 0.000100
✅ New best validation balanced accuracy: 0.689, threshold: 0.456
📊 Mag Importance (Val BalAcc: 0.689): {'40': 0.25321584939956665, '100': 0.2626064717769623, '200': 0.25133973360061646, '400': 0.23283791542053223}
Epoch 05: Train: Loss 0.0170, Acc 0.644 | Val: Loss 0.0378, Acc 0.817, BalAcc 0.778, F1 0.870, AUC 0.743, Prec 0.860, Rec 0.880, Thresh 0.464 | LR: 0.000100
✅ New best validation balanced accuracy: 0.778, threshold: 0.464
📊 Mag Importance (Val BalAcc: 0.778): {'40': 0.25190117955207825, '100': 0.26383909583091736, '200': 0.2502812445163727, '400': 0.2339785397052765}
Epoch 06: Train: Loss 0.0190, Acc 0.673 | Val: Loss 0.0325, Acc 0.797, BalAcc 0.699, F1 0.866, AUC 0.747, Prec 0.795, Rec 0.952, Thresh 0.383 | LR: 0.000100
Epoch 07: Train: Loss 0.0175, Acc 0.691 | Val: Loss 0.0321, Acc 0.813, BalAcc 0.734, F1 0.875, AUC 0.731, Prec 0.818, Rec 0.940, Thresh 0.423 | LR: 0.000100
Epoch 08: Train: Loss 0.0166, Acc 0.709 | Val: Loss 0.0584, Acc 0.826, BalAcc 0.791, F1 0.875, AUC 0.750, Prec 0.870, Rec 0.880, Thresh 0.504 | LR: 0.000100
✅ New best validation balanced accuracy: 0.791, threshold: 0.504
📊 Mag Importance (Val BalAcc: 0.791): {'40': 0.2503712475299835, '100': 0.266275018453598, '200': 0.24956543743610382, '400': 0.23378832638263702}
Epoch 09: Train: Loss 0.0148, Acc 0.677 | Val: Loss 0.0653, Acc 0.822, BalAcc 0.762, F1 0.877, AUC 0.700, Prec 0.841, Rec 0.916, Thresh 0.456 | LR: 0.000100
Epoch 10: Train: Loss 0.0177, Acc 0.687 | Val: Loss 0.0350, Acc 0.817, BalAcc 0.740, F1 0.877, AUC 0.697, Prec 0.822, Rec 0.940, Thresh 0.456 | LR: 0.000100
Epoch 11: Train: Loss 0.0133, Acc 0.693 | Val: Loss 0.0551, Acc 0.830, BalAcc 0.779, F1 0.881, AUC 0.730, Prec 0.854, Rec 0.910, Thresh 0.472 | LR: 0.000100
Epoch 12: Train: Loss 0.0143, Acc 0.695 | Val: Loss 0.0281, Acc 0.851, BalAcc 0.787, F1 0.898, AUC 0.767, Prec 0.850, Rec 0.952, Thresh 0.431 | LR: 0.000050
Epoch 13: Train: Loss 0.0147, Acc 0.709 | Val: Loss 0.0243, Acc 0.842, BalAcc 0.796, F1 0.890, AUC 0.811, Prec 0.864, Rec 0.916, Thresh 0.496 | LR: 0.000050
✅ New best validation balanced accuracy: 0.796, threshold: 0.496
📊 Mag Importance (Val BalAcc: 0.796): {'40': 0.24981561303138733, '100': 0.2665737271308899, '200': 0.25275513529777527, '400': 0.23085549473762512}
Epoch 14: Train: Loss 0.0137, Acc 0.728 | Val: Loss 0.0290, Acc 0.842, BalAcc 0.785, F1 0.891, AUC 0.765, Prec 0.852, Rec 0.934, Thresh 0.431 | LR: 0.000050
Epoch 15: Train: Loss 0.0134, Acc 0.696 | Val: Loss 0.0197, Acc 0.846, BalAcc 0.776, F1 0.896, AUC 0.795, Prec 0.842, Rec 0.958, Thresh 0.423 | LR: 0.000050
Epoch 16: Train: Loss 0.0157, Acc 0.726 | Val: Loss 0.0217, Acc 0.830, BalAcc 0.783, F1 0.880, AUC 0.787, Prec 0.858, Rec 0.904, Thresh 0.480 | LR: 0.000050
Epoch 17: Train: Loss 0.0122, Acc 0.724 | Val: Loss 0.0214, Acc 0.842, BalAcc 0.788, F1 0.891, AUC 0.812, Prec 0.856, Rec 0.928, Thresh 0.480 | LR: 0.000025
Epoch 18: Train: Loss 0.0116, Acc 0.724 | Val: Loss 0.0229, Acc 0.842, BalAcc 0.796, F1 0.890, AUC 0.824, Prec 0.864, Rec 0.916, Thresh 0.472 | LR: 0.000025
Epoch 19: Train: Loss 0.0108, Acc 0.732 | Val: Loss 0.0261, Acc 0.838, BalAcc 0.774, F1 0.890, AUC 0.798, Prec 0.844, Rec 0.940, Thresh 0.447 | LR: 0.000025
Epoch 20: Train: Loss 0.0139, Acc 0.735 | Val: Loss 0.0207, Acc 0.855, BalAcc 0.794, F1 0.901, AUC 0.811, Prec 0.855, Rec 0.952, Thresh 0.439 | LR: 0.000025
⚠️ Early stopping after 20 epochs (no improvement for 7 epochs)
✅ Best model saved: ./output/models/best_model_fold_0.pth (Val BalAcc: 0.796)
⚡️ Test Results: Acc 0.918, BalAcc 0.931, F1 0.930, AUC 0.992, Precision 0.990, Recall 0.878 (threshold: 0.496)
📊 Confusion Matrix (Fold 0):
   [[TN: 130, FP:   2]
    [FN:  27, TP: 194]]
⚡ Avg Inference Time: 0.0085s per sample
Traceback (most recent call last):
  File "/workspace/MultiMagBC/main.py", line 343, in <module>
    main()
  File "/workspace/MultiMagBC/main.py", line 260, in main
    save_as_json(fold_results, json_path)
  File "/workspace/MultiMagBC/utils/stats.py", line 6, in save_as_json
    json.dump(data, f, indent=2)
  File "/usr/lib/python3.11/json/__init__.py", line 179, in dump
    for chunk in iterable:
  File "/usr/lib/python3.11/json/encoder.py", line 432, in _iterencode
    yield from _iterencode_dict(o, _current_indent_level)
  File "/usr/lib/python3.11/json/encoder.py", line 406, in _iterencode_dict
    yield from chunks
  File "/usr/lib/python3.11/json/encoder.py", line 439, in _iterencode
    o = _default(o)
        ^^^^^^^^^^^
  File "/usr/lib/python3.11/json/encoder.py", line 180, in default
    raise TypeError(f'Object of type {o.__class__.__name__} '
TypeError: Object of type ndarray is not JSON serializable
root@0b7d713d4452:/workspace/MultiMagBC# git pull
remote: Enumerating objects: 6, done.
remote: Counting objects: 100% (6/6), done.
remote: Compressing objects: 100% (3/3), done.
remote: Total 6 (delta 3), reused 6 (delta 3), pack-reused 0 (from 0)
Unpacking objects: 100% (6/6), 3.30 KiB | 41.00 KiB/s, done.
From https://github.com/Auniik/MultiMagBC
   cc46121..24da52b  lightweight -> origin/lightweight
Updating cc46121..24da52b
error: Your local changes to the following files would be overwritten by merge:
	RUNPOD_OUTPUT.md
	utils/stats.py
Please commit your changes or stash them before you merge.
Aborting
root@0b7d713d4452:/workspace/MultiMagBC# git stash
Saved working directory and index state WIP on lightweight: cc46121 Updated baseline with lightweight 3 with fix
root@0b7d713d4452:/workspace/MultiMagBC# git pull
Updating cc46121..24da52b
Fast-forward
 RUNPOD_OUTPUT.md | 110 ++++++++++++++++++++++++++++++++++++++++++++++++++++++--------------------------------------------------------
 test_json_fix.py |  73 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 utils/stats.py   |  16 +++++++++++++++-
 3 files changed, 142 insertions(+), 57 deletions(-)
 create mode 100644 test_json_fix.py
root@0b7d713d4452:/workspace/MultiMagBC# python main.py
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
Epoch 01: Train: Loss 0.0272, Acc 0.563 | Val: Loss 0.0275, Acc 0.759, BalAcc 0.653, F1 0.842, AUC 0.719, Prec 0.771, Rec 0.928, Thresh 0.399 | LR: 0.000100
✅ New best validation balanced accuracy: 0.653, threshold: 0.399
📊 Mag Importance (Val BalAcc: 0.653): {'40': 0.25323840975761414, '100': 0.26283150911331177, '200': 0.251819908618927, '400': 0.2321101427078247}
Epoch 02: Train: Loss 0.0232, Acc 0.616 | Val: Loss 0.0329, Acc 0.743, BalAcc 0.592, F1 0.841, AUC 0.698, Prec 0.735, Rec 0.982, Thresh 0.342 | LR: 0.000100
Epoch 03: Train: Loss 0.0206, Acc 0.643 | Val: Loss 0.0288, Acc 0.768, BalAcc 0.671, F1 0.846, AUC 0.718, Prec 0.782, Rec 0.922, Thresh 0.391 | LR: 0.000100
✅ New best validation balanced accuracy: 0.671, threshold: 0.391
📊 Mag Importance (Val BalAcc: 0.671): {'40': 0.2527458667755127, '100': 0.2630433440208435, '200': 0.2535671591758728, '400': 0.2306436002254486}
Epoch 04: Train: Loss 0.0200, Acc 0.639 | Val: Loss 0.0361, Acc 0.776, BalAcc 0.695, F1 0.848, AUC 0.702, Prec 0.799, Rec 0.904, Thresh 0.447 | LR: 0.000100
✅ New best validation balanced accuracy: 0.695, threshold: 0.447
📊 Mag Importance (Val BalAcc: 0.695): {'40': 0.25466758012771606, '100': 0.26224058866500854, '200': 0.2521589398384094, '400': 0.23093286156654358}
Epoch 05: Train: Loss 0.0173, Acc 0.643 | Val: Loss 0.0242, Acc 0.801, BalAcc 0.728, F1 0.864, AUC 0.767, Prec 0.818, Rec 0.916, Thresh 0.423 | LR: 0.000100
✅ New best validation balanced accuracy: 0.728, threshold: 0.423
📊 Mag Importance (Val BalAcc: 0.728): {'40': 0.254907488822937, '100': 0.26373201608657837, '200': 0.25101298093795776, '400': 0.23034751415252686}
Epoch 06: Train: Loss 0.0189, Acc 0.657 | Val: Loss 0.0213, Acc 0.817, BalAcc 0.718, F1 0.881, AUC 0.781, Prec 0.803, Rec 0.976, Thresh 0.391 | LR: 0.000100
Epoch 07: Train: Loss 0.0174, Acc 0.664 | Val: Loss 0.0307, Acc 0.797, BalAcc 0.680, F1 0.870, AUC 0.738, Prec 0.781, Rec 0.982, Thresh 0.375 | LR: 0.000100
Epoch 08: Train: Loss 0.0165, Acc 0.693 | Val: Loss 0.0282, Acc 0.813, BalAcc 0.726, F1 0.876, AUC 0.756, Prec 0.811, Rec 0.952, Thresh 0.456 | LR: 0.000100
Epoch 09: Train: Loss 0.0143, Acc 0.694 | Val: Loss 0.0458, Acc 0.846, BalAcc 0.788, F1 0.895, AUC 0.706, Prec 0.853, Rec 0.940, Thresh 0.439 | LR: 0.000100
✅ New best validation balanced accuracy: 0.788, threshold: 0.439
📊 Mag Importance (Val BalAcc: 0.788): {'40': 0.2545764446258545, '100': 0.2632032632827759, '200': 0.25597748160362244, '400': 0.22624275088310242}
Epoch 10: Train: Loss 0.0167, Acc 0.690 | Val: Loss 0.0299, Acc 0.834, BalAcc 0.771, F1 0.886, AUC 0.729, Prec 0.843, Rec 0.934, Thresh 0.464 | LR: 0.000100
Epoch 11: Train: Loss 0.0136, Acc 0.711 | Val: Loss 0.0442, Acc 0.846, BalAcc 0.803, F1 0.892, AUC 0.728, Prec 0.869, Rec 0.916, Thresh 0.480 | LR: 0.000100
✅ New best validation balanced accuracy: 0.803, threshold: 0.480
📊 Mag Importance (Val BalAcc: 0.803): {'40': 0.25466758012771606, '100': 0.2642529606819153, '200': 0.2515111565589905, '400': 0.2295682728290558}
Epoch 12: Train: Loss 0.0139, Acc 0.702 | Val: Loss 0.0268, Acc 0.851, BalAcc 0.787, F1 0.898, AUC 0.753, Prec 0.850, Rec 0.952, Thresh 0.431 | LR: 0.000100
Epoch 13: Train: Loss 0.0143, Acc 0.695 | Val: Loss 0.0212, Acc 0.834, BalAcc 0.790, F1 0.883, AUC 0.796, Prec 0.863, Rec 0.904, Thresh 0.464 | LR: 0.000100
Epoch 14: Train: Loss 0.0130, Acc 0.717 | Val: Loss 0.0361, Acc 0.851, BalAcc 0.802, F1 0.896, AUC 0.739, Prec 0.866, Rec 0.928, Thresh 0.456 | LR: 0.000100
Epoch 15: Train: Loss 0.0130, Acc 0.700 | Val: Loss 0.0205, Acc 0.838, BalAcc 0.763, F1 0.891, AUC 0.772, Prec 0.833, Rec 0.958, Thresh 0.431 | LR: 0.000050
Epoch 16: Train: Loss 0.0144, Acc 0.724 | Val: Loss 0.0227, Acc 0.830, BalAcc 0.787, F1 0.880, AUC 0.777, Prec 0.862, Rec 0.898, Thresh 0.496 | LR: 0.000050
Epoch 17: Train: Loss 0.0117, Acc 0.731 | Val: Loss 0.0221, Acc 0.830, BalAcc 0.764, F1 0.884, AUC 0.798, Prec 0.839, Rec 0.934, Thresh 0.456 | LR: 0.000050
Epoch 18: Train: Loss 0.0113, Acc 0.722 | Val: Loss 0.0232, Acc 0.838, BalAcc 0.763, F1 0.891, AUC 0.818, Prec 0.833, Rec 0.958, Thresh 0.423 | LR: 0.000050
⚠️ Early stopping after 18 epochs (no improvement for 7 epochs)
✅ Best model saved: ./output/models/best_model_fold_0.pth (Val BalAcc: 0.803)
⚡️ Test Results: Acc 0.941, BalAcc 0.948, F1 0.951, AUC 0.994, Precision 0.985, Recall 0.919 (threshold: 0.480)
📊 Confusion Matrix (Fold 0):
   [[TN: 129, FP:   3]
    [FN:  18, TP: 203]]
⚡ Avg Inference Time: 0.0086s per sample
📌 Final Magnification Importance (Fold 0): {'40': 0.24513955414295197, '100': 0.2621263563632965, '200': 0.2661844789981842, '400': 0.22654959559440613}
💾 Results saved to: ./output/results/fold_0_results.json

📊 Generating GradCAM visualizations for fold 0...
✅ Generated 5 GradCAM visualizations for fold 0

===== Fold 1 =====
Train patients: 52, Val Patients: 13, Test patients: 17
Training samples per epoch: {'total_samples_per_epoch': 2169, 'class_distribution': {0: 639, 1: 1530}, 'oversampling_factor': 2.05}
Validation samples: 229, Test samples: 359
Patients with full 4 mags: 52
Inner training samples: 2169, batch size: 16
Class weights: Benign=3.47, Malignant=0.21
Epoch 01: Train: Loss 0.0269, Acc 0.538 | Val: Loss 0.0129, Acc 0.808, BalAcc 0.773, F1 0.861, AUC 0.855, Prec 0.855, Rec 0.866, Thresh 0.407 | LR: 0.000100
✅ New best validation balanced accuracy: 0.773, threshold: 0.407
📊 Mag Importance (Val BalAcc: 0.773): {'40': 0.25281721353530884, '100': 0.23390769958496094, '200': 0.23401407897472382, '400': 0.27926105260849}
Epoch 02: Train: Loss 0.0223, Acc 0.582 | Val: Loss 0.0111, Acc 0.834, BalAcc 0.766, F1 0.887, AUC 0.889, Prec 0.832, Rec 0.949, Thresh 0.375 | LR: 0.000100
Epoch 03: Train: Loss 0.0198, Acc 0.622 | Val: Loss 0.0095, Acc 0.882, BalAcc 0.858, F1 0.915, AUC 0.929, Prec 0.906, Rec 0.924, Thresh 0.431 | LR: 0.000100
✅ New best validation balanced accuracy: 0.858, threshold: 0.431
📊 Mag Importance (Val BalAcc: 0.858): {'40': 0.25583741068840027, '100': 0.2319726049900055, '200': 0.23258507251739502, '400': 0.2796049118041992}
Epoch 04: Train: Loss 0.0168, Acc 0.648 | Val: Loss 0.0098, Acc 0.886, BalAcc 0.876, F1 0.916, AUC 0.940, Prec 0.928, Rec 0.904, Thresh 0.439 | LR: 0.000100
✅ New best validation balanced accuracy: 0.876, threshold: 0.439
📊 Mag Importance (Val BalAcc: 0.876): {'40': 0.2567233741283417, '100': 0.23025481402873993, '200': 0.2320982962846756, '400': 0.2809234857559204}
Epoch 05: Train: Loss 0.0177, Acc 0.642 | Val: Loss 0.0099, Acc 0.917, BalAcc 0.906, F1 0.939, AUC 0.949, Prec 0.942, Rec 0.936, Thresh 0.431 | LR: 0.000100
✅ New best validation balanced accuracy: 0.906, threshold: 0.431
📊 Mag Importance (Val BalAcc: 0.906): {'40': 0.25869640707969666, '100': 0.22825288772583008, '200': 0.23320475220680237, '400': 0.2798459529876709}
Epoch 06: Train: Loss 0.0162, Acc 0.649 | Val: Loss 0.0094, Acc 0.930, BalAcc 0.923, F1 0.949, AUC 0.951, Prec 0.955, Rec 0.943, Thresh 0.431 | LR: 0.000100
✅ New best validation balanced accuracy: 0.923, threshold: 0.431
📊 Mag Importance (Val BalAcc: 0.923): {'40': 0.2602885961532593, '100': 0.22531965374946594, '200': 0.23157624900341034, '400': 0.28281551599502563}
Epoch 07: Train: Loss 0.0157, Acc 0.677 | Val: Loss 0.0079, Acc 0.917, BalAcc 0.909, F1 0.939, AUC 0.973, Prec 0.948, Rec 0.930, Thresh 0.447 | LR: 0.000100
Epoch 08: Train: Loss 0.0132, Acc 0.728 | Val: Loss 0.0072, Acc 0.939, BalAcc 0.952, F1 0.954, AUC 0.969, Prec 0.993, Rec 0.917, Thresh 0.464 | LR: 0.000100
✅ New best validation balanced accuracy: 0.952, threshold: 0.464
📊 Mag Importance (Val BalAcc: 0.952): {'40': 0.25918248295783997, '100': 0.2294633388519287, '200': 0.23288984596729279, '400': 0.27846425771713257}
Epoch 09: Train: Loss 0.0139, Acc 0.713 | Val: Loss 0.0103, Acc 0.917, BalAcc 0.917, F1 0.938, AUC 0.951, Prec 0.960, Rec 0.917, Thresh 0.431 | LR: 0.000100
Epoch 10: Train: Loss 0.0132, Acc 0.734 | Val: Loss 0.0110, Acc 0.921, BalAcc 0.916, F1 0.942, AUC 0.960, Prec 0.954, Rec 0.930, Thresh 0.391 | LR: 0.000100
Epoch 11: Train: Loss 0.0128, Acc 0.692 | Val: Loss 0.0069, Acc 0.930, BalAcc 0.930, F1 0.948, AUC 0.971, Prec 0.967, Rec 0.930, Thresh 0.439 | LR: 0.000100
Epoch 12: Train: Loss 0.0120, Acc 0.741 | Val: Loss 0.0061, Acc 0.948, BalAcc 0.962, F1 0.960, AUC 0.968, Prec 1.000, Rec 0.924, Thresh 0.472 | LR: 0.000100
✅ New best validation balanced accuracy: 0.962, threshold: 0.472
📊 Mag Importance (Val BalAcc: 0.962): {'40': 0.2565760910511017, '100': 0.2285422384738922, '200': 0.23823609948158264, '400': 0.27664560079574585}
Epoch 13: Train: Loss 0.0127, Acc 0.723 | Val: Loss 0.0059, Acc 0.943, BalAcc 0.959, F1 0.957, AUC 0.979, Prec 1.000, Rec 0.917, Thresh 0.464 | LR: 0.000100
Epoch 14: Train: Loss 0.0119, Acc 0.758 | Val: Loss 0.0058, Acc 0.939, BalAcc 0.937, F1 0.955, AUC 0.983, Prec 0.967, Rec 0.943, Thresh 0.415 | LR: 0.000100
Epoch 15: Train: Loss 0.0129, Acc 0.736 | Val: Loss 0.0070, Acc 0.939, BalAcc 0.955, F1 0.953, AUC 0.966, Prec 1.000, Rec 0.911, Thresh 0.480 | LR: 0.000100
Epoch 16: Train: Loss 0.0127, Acc 0.733 | Val: Loss 0.0064, Acc 0.939, BalAcc 0.944, F1 0.954, AUC 0.974, Prec 0.980, Rec 0.930, Thresh 0.423 | LR: 0.000050
Epoch 17: Train: Loss 0.0115, Acc 0.750 | Val: Loss 0.0060, Acc 0.934, BalAcc 0.937, F1 0.951, AUC 0.976, Prec 0.973, Rec 0.930, Thresh 0.439 | LR: 0.000050
Epoch 18: Train: Loss 0.0123, Acc 0.716 | Val: Loss 0.0113, Acc 0.939, BalAcc 0.948, F1 0.954, AUC 0.967, Prec 0.986, Rec 0.924, Thresh 0.351 | LR: 0.000050
Epoch 19: Train: Loss 0.0113, Acc 0.754 | Val: Loss 0.0067, Acc 0.943, BalAcc 0.955, F1 0.957, AUC 0.969, Prec 0.993, Rec 0.924, Thresh 0.464 | LR: 0.000050
⚠️ Early stopping after 19 epochs (no improvement for 7 epochs)
✅ Best model saved: ./output/models/best_model_fold_1.pth (Val BalAcc: 0.962)
⚡️ Test Results: Acc 0.852, BalAcc 0.839, F1 0.888, AUC 0.910, Precision 0.898, Recall 0.879 (threshold: 0.472)
📊 Confusion Matrix (Fold 1):
   [[TN:  95, FP:  24]
    [FN:  29, TP: 211]]
⚡ Avg Inference Time: 0.0086s per sample
📌 Final Magnification Importance (Fold 1): {'40': 0.2545374035835266, '100': 0.22567477822303772, '200': 0.23727183043956757, '400': 0.2825160026550293}
💾 Results saved to: ./output/results/fold_1_results.json

📊 Generating GradCAM visualizations for fold 1...
✅ Generated 5 GradCAM visualizations for fold 1

===== Fold 2 =====
Train patients: 52, Val Patients: 14, Test patients: 16
Training samples per epoch: {'total_samples_per_epoch': 2160, 'class_distribution': {1: 1515, 0: 645}, 'oversampling_factor': 1.98}
Validation samples: 262, Test samples: 293
Patients with full 4 mags: 52
Inner training samples: 2160, batch size: 16
Class weights: Benign=3.47, Malignant=0.21
Epoch 01: Train: Loss 0.0355, Acc 0.557 | Val: Loss 0.0167, Acc 0.809, BalAcc 0.700, F1 0.878, AUC 0.866, Prec 0.837, Rec 0.923, Thresh 0.334 | LR: 0.000100
✅ New best validation balanced accuracy: 0.700, threshold: 0.334
📊 Mag Importance (Val BalAcc: 0.700): {'40': 0.2508813142776489, '100': 0.2390602082014084, '200': 0.2632010877132416, '400': 0.2468574047088623}
Epoch 02: Train: Loss 0.0270, Acc 0.586 | Val: Loss 0.0162, Acc 0.844, BalAcc 0.733, F1 0.901, AUC 0.887, Prec 0.850, Rec 0.959, Thresh 0.318 | LR: 0.000100
✅ New best validation balanced accuracy: 0.733, threshold: 0.318
📊 Mag Importance (Val BalAcc: 0.733): {'40': 0.25167447328567505, '100': 0.23789441585540771, '200': 0.26378870010375977, '400': 0.24664244055747986}
Epoch 03: Train: Loss 0.0244, Acc 0.628 | Val: Loss 0.0104, Acc 0.889, BalAcc 0.823, F1 0.928, AUC 0.921, Prec 0.899, Rec 0.959, Thresh 0.399 | LR: 0.000100
✅ New best validation balanced accuracy: 0.823, threshold: 0.399
📊 Mag Importance (Val BalAcc: 0.823): {'40': 0.25262993574142456, '100': 0.2356185019016266, '200': 0.26579245924949646, '400': 0.2459591180086136}
Epoch 04: Train: Loss 0.0203, Acc 0.635 | Val: Loss 0.0096, Acc 0.908, BalAcc 0.855, F1 0.940, AUC 0.936, Prec 0.917, Rec 0.964, Thresh 0.423 | LR: 0.000100
✅ New best validation balanced accuracy: 0.855, threshold: 0.423
📊 Mag Importance (Val BalAcc: 0.855): {'40': 0.2518474757671356, '100': 0.2364361733198166, '200': 0.26653432846069336, '400': 0.24518203735351562}
Epoch 05: Train: Loss 0.0193, Acc 0.643 | Val: Loss 0.0059, Acc 0.962, BalAcc 0.940, F1 0.975, AUC 0.983, Prec 0.965, Rec 0.985, Thresh 0.456 | LR: 0.000100
✅ New best validation balanced accuracy: 0.940, threshold: 0.456
📊 Mag Importance (Val BalAcc: 0.940): {'40': 0.25071609020233154, '100': 0.23586341738700867, '200': 0.2685011029243469, '400': 0.2449193298816681}
Epoch 06: Train: Loss 0.0172, Acc 0.660 | Val: Loss 0.0063, Acc 0.950, BalAcc 0.932, F1 0.967, AUC 0.984, Prec 0.964, Rec 0.969, Thresh 0.431 | LR: 0.000100
Epoch 07: Train: Loss 0.0159, Acc 0.653 | Val: Loss 0.0087, Acc 0.905, BalAcc 0.892, F1 0.935, AUC 0.954, Prec 0.952, Rec 0.918, Thresh 0.480 | LR: 0.000100
Epoch 08: Train: Loss 0.0159, Acc 0.686 | Val: Loss 0.0063, Acc 0.966, BalAcc 0.948, F1 0.977, AUC 0.987, Prec 0.970, Rec 0.985, Thresh 0.464 | LR: 0.000100
✅ New best validation balanced accuracy: 0.948, threshold: 0.464
📊 Mag Importance (Val BalAcc: 0.948): {'40': 0.2552563548088074, '100': 0.23298723995685577, '200': 0.26835811138153076, '400': 0.2433982789516449}
Epoch 09: Train: Loss 0.0153, Acc 0.675 | Val: Loss 0.0061, Acc 0.962, BalAcc 0.940, F1 0.975, AUC 0.983, Prec 0.965, Rec 0.985, Thresh 0.431 | LR: 0.000100
Epoch 10: Train: Loss 0.0137, Acc 0.703 | Val: Loss 0.0047, Acc 0.962, BalAcc 0.925, F1 0.975, AUC 0.992, Prec 0.951, Rec 1.000, Thresh 0.383 | LR: 0.000100
Epoch 11: Train: Loss 0.0133, Acc 0.715 | Val: Loss 0.0051, Acc 0.969, BalAcc 0.975, F1 0.979, AUC 0.985, Prec 0.995, Rec 0.964, Thresh 0.472 | LR: 0.000100
✅ New best validation balanced accuracy: 0.975, threshold: 0.472
📊 Mag Importance (Val BalAcc: 0.975): {'40': 0.24974223971366882, '100': 0.23598361015319824, '200': 0.270175576210022, '400': 0.2440985143184662}
Epoch 12: Train: Loss 0.0145, Acc 0.673 | Val: Loss 0.0048, Acc 0.969, BalAcc 0.955, F1 0.980, AUC 0.994, Prec 0.975, Rec 0.985, Thresh 0.423 | LR: 0.000100
Epoch 13: Train: Loss 0.0130, Acc 0.707 | Val: Loss 0.0042, Acc 0.989, BalAcc 0.983, F1 0.992, AUC 0.998, Prec 0.990, Rec 0.995, Thresh 0.456 | LR: 0.000100
✅ New best validation balanced accuracy: 0.983, threshold: 0.456
📊 Mag Importance (Val BalAcc: 0.983): {'40': 0.2507482171058655, '100': 0.2346663922071457, '200': 0.2722240686416626, '400': 0.24236135184764862}
Epoch 14: Train: Loss 0.0128, Acc 0.717 | Val: Loss 0.0049, Acc 0.977, BalAcc 0.955, F1 0.985, AUC 0.989, Prec 0.970, Rec 1.000, Thresh 0.431 | LR: 0.000100
Epoch 15: Train: Loss 0.0124, Acc 0.707 | Val: Loss 0.0044, Acc 0.989, BalAcc 0.978, F1 0.992, AUC 0.999, Prec 0.985, Rec 1.000, Thresh 0.464 | LR: 0.000100
Epoch 16: Train: Loss 0.0123, Acc 0.727 | Val: Loss 0.0045, Acc 0.969, BalAcc 0.975, F1 0.979, AUC 0.996, Prec 0.995, Rec 0.964, Thresh 0.480 | LR: 0.000100
Epoch 17: Train: Loss 0.0118, Acc 0.744 | Val: Loss 0.0039, Acc 0.985, BalAcc 0.970, F1 0.990, AUC 0.999, Prec 0.980, Rec 1.000, Thresh 0.423 | LR: 0.000050
Epoch 18: Train: Loss 0.0105, Acc 0.731 | Val: Loss 0.0030, Acc 0.992, BalAcc 0.985, F1 0.995, AUC 1.000, Prec 0.990, Rec 1.000, Thresh 0.439 | LR: 0.000050
✅ New best validation balanced accuracy: 0.985, threshold: 0.439
📊 Mag Importance (Val BalAcc: 0.985): {'40': 0.2510261535644531, '100': 0.23110704123973846, '200': 0.27096930146217346, '400': 0.24689750373363495}
Epoch 19: Train: Loss 0.0114, Acc 0.739 | Val: Loss 0.0046, Acc 0.985, BalAcc 0.980, F1 0.990, AUC 0.999, Prec 0.990, Rec 0.990, Thresh 0.520 | LR: 0.000050
Epoch 20: Train: Loss 0.0104, Acc 0.725 | Val: Loss 0.0039, Acc 0.981, BalAcc 0.977, F1 0.987, AUC 0.998, Prec 0.990, Rec 0.985, Thresh 0.439 | LR: 0.000050
Epoch 21: Train: Loss 0.0109, Acc 0.757 | Val: Loss 0.0031, Acc 1.000, BalAcc 1.000, F1 1.000, AUC 1.000, Prec 1.000, Rec 1.000, Thresh 0.439 | LR: 0.000050
✅ New best validation balanced accuracy: 1.000, threshold: 0.439
📊 Mag Importance (Val BalAcc: 1.000): {'40': 0.25100177526474, '100': 0.23422464728355408, '200': 0.2684800624847412, '400': 0.24629350006580353}
Epoch 22: Train: Loss 0.0101, Acc 0.740 | Val: Loss 0.0035, Acc 1.000, BalAcc 1.000, F1 1.000, AUC 1.000, Prec 1.000, Rec 1.000, Thresh 0.407 | LR: 0.000050
Epoch 23: Train: Loss 0.0119, Acc 0.729 | Val: Loss 0.0034, Acc 0.985, BalAcc 0.975, F1 0.990, AUC 0.999, Prec 0.985, Rec 0.995, Thresh 0.415 | LR: 0.000050
Epoch 24: Train: Loss 0.0111, Acc 0.762 | Val: Loss 0.0037, Acc 0.981, BalAcc 0.977, F1 0.987, AUC 0.997, Prec 0.990, Rec 0.985, Thresh 0.375 | LR: 0.000050
Epoch 25: Train: Loss 0.0106, Acc 0.761 | Val: Loss 0.0047, Acc 0.962, BalAcc 0.969, F1 0.974, AUC 0.993, Prec 0.995, Rec 0.954, Thresh 0.383 | LR: 0.000025
✅ Best model saved: ./output/models/best_model_fold_2.pth (Val BalAcc: 1.000)
⚡️ Test Results: Acc 0.758, BalAcc 0.753, F1 0.812, AUC 0.817, Precision 0.864, Recall 0.765 (threshold: 0.439)
📊 Confusion Matrix (Fold 2):
   [[TN:  69, FP:  24]
    [FN:  47, TP: 153]]
⚡ Avg Inference Time: 0.0091s per sample
📌 Final Magnification Importance (Fold 2): {'40': 0.25108397006988525, '100': 0.23364976048469543, '200': 0.27657991647720337, '400': 0.23868636786937714}
💾 Results saved to: ./output/results/fold_2_results.json

📊 Generating GradCAM visualizations for fold 2...
✅ Generated 5 GradCAM visualizations for fold 2

===== Fold 3 =====
Train patients: 52, Val Patients: 14, Test patients: 16
Training samples per epoch: {'total_samples_per_epoch': 2214, 'class_distribution': {1: 1572, 0: 642}, 'oversampling_factor': 2.1}
Validation samples: 238, Test samples: 354
Patients with full 4 mags: 52
Inner training samples: 2214, batch size: 16
Class weights: Benign=3.47, Malignant=0.21
Epoch 01: Train: Loss 0.0360, Acc 0.571 | Val: Loss 0.0155, Acc 0.761, BalAcc 0.735, F1 0.820, AUC 0.814, Prec 0.833, Rec 0.807, Thresh 0.472 | LR: 0.000100
✅ New best validation balanced accuracy: 0.735, threshold: 0.472
📊 Mag Importance (Val BalAcc: 0.735): {'40': 0.24670033156871796, '100': 0.2824554443359375, '200': 0.22805090248584747, '400': 0.24279335141181946}
Epoch 02: Train: Loss 0.0267, Acc 0.613 | Val: Loss 0.0185, Acc 0.752, BalAcc 0.732, F1 0.812, AUC 0.798, Prec 0.836, Rec 0.789, Thresh 0.431 | LR: 0.000100
Epoch 03: Train: Loss 0.0222, Acc 0.622 | Val: Loss 0.0136, Acc 0.845, BalAcc 0.807, F1 0.888, AUC 0.888, Prec 0.865, Rec 0.913, Thresh 0.456 | LR: 0.000100
✅ New best validation balanced accuracy: 0.807, threshold: 0.456
📊 Mag Importance (Val BalAcc: 0.807): {'40': 0.2449355125427246, '100': 0.2824684977531433, '200': 0.23028044402599335, '400': 0.24231556057929993}
Epoch 04: Train: Loss 0.0203, Acc 0.657 | Val: Loss 0.0112, Acc 0.857, BalAcc 0.830, F1 0.896, AUC 0.896, Prec 0.885, Rec 0.907, Thresh 0.456 | LR: 0.000100
✅ New best validation balanced accuracy: 0.830, threshold: 0.456
📊 Mag Importance (Val BalAcc: 0.830): {'40': 0.24567405879497528, '100': 0.2819755971431732, '200': 0.22943374514579773, '400': 0.2429165542125702}
Epoch 05: Train: Loss 0.0167, Acc 0.659 | Val: Loss 0.0086, Acc 0.908, BalAcc 0.891, F1 0.932, AUC 0.950, Prec 0.926, Rec 0.938, Thresh 0.472 | LR: 0.000100
✅ New best validation balanced accuracy: 0.891, threshold: 0.472
📊 Mag Importance (Val BalAcc: 0.891): {'40': 0.24393925070762634, '100': 0.27974510192871094, '200': 0.22894060611724854, '400': 0.2473749816417694}
Epoch 06: Train: Loss 0.0162, Acc 0.672 | Val: Loss 0.0103, Acc 0.866, BalAcc 0.829, F1 0.904, AUC 0.919, Prec 0.877, Rec 0.932, Thresh 0.456 | LR: 0.000100
Epoch 07: Train: Loss 0.0174, Acc 0.671 | Val: Loss 0.0092, Acc 0.878, BalAcc 0.832, F1 0.914, AUC 0.936, Prec 0.871, Rec 0.963, Thresh 0.447 | LR: 0.000100
Epoch 08: Train: Loss 0.0162, Acc 0.654 | Val: Loss 0.0088, Acc 0.920, BalAcc 0.883, F1 0.944, AUC 0.951, Prec 0.903, Rec 0.988, Thresh 0.439 | LR: 0.000100
Epoch 09: Train: Loss 0.0148, Acc 0.678 | Val: Loss 0.0109, Acc 0.887, BalAcc 0.852, F1 0.919, AUC 0.936, Prec 0.890, Rec 0.950, Thresh 0.480 | LR: 0.000050
Epoch 10: Train: Loss 0.0131, Acc 0.684 | Val: Loss 0.0110, Acc 0.899, BalAcc 0.851, F1 0.930, AUC 0.932, Prec 0.878, Rec 0.988, Thresh 0.480 | LR: 0.000050
Epoch 11: Train: Loss 0.0125, Acc 0.708 | Val: Loss 0.0078, Acc 0.945, BalAcc 0.916, F1 0.961, AUC 0.966, Prec 0.925, Rec 1.000, Thresh 0.464 | LR: 0.000050
✅ New best validation balanced accuracy: 0.916, threshold: 0.464
📊 Mag Importance (Val BalAcc: 0.916): {'40': 0.24792127311229706, '100': 0.28013795614242554, '200': 0.2293572872877121, '400': 0.24258343875408173}
Epoch 12: Train: Loss 0.0134, Acc 0.687 | Val: Loss 0.0126, Acc 0.857, BalAcc 0.827, F1 0.896, AUC 0.906, Prec 0.880, Rec 0.913, Thresh 0.504 | LR: 0.000050
Epoch 13: Train: Loss 0.0128, Acc 0.723 | Val: Loss 0.0090, Acc 0.916, BalAcc 0.887, F1 0.940, AUC 0.951, Prec 0.912, Rec 0.969, Thresh 0.472 | LR: 0.000050
Epoch 14: Train: Loss 0.0118, Acc 0.708 | Val: Loss 0.0087, Acc 0.903, BalAcc 0.871, F1 0.931, AUC 0.953, Prec 0.901, Rec 0.963, Thresh 0.464 | LR: 0.000050
Epoch 15: Train: Loss 0.0123, Acc 0.730 | Val: Loss 0.0126, Acc 0.874, BalAcc 0.842, F1 0.909, AUC 0.922, Prec 0.888, Rec 0.932, Thresh 0.520 | LR: 0.000025
Epoch 16: Train: Loss 0.0124, Acc 0.688 | Val: Loss 0.0091, Acc 0.908, BalAcc 0.898, F1 0.931, AUC 0.952, Prec 0.937, Rec 0.925, Thresh 0.512 | LR: 0.000025
Epoch 17: Train: Loss 0.0127, Acc 0.731 | Val: Loss 0.0075, Acc 0.929, BalAcc 0.903, F1 0.949, AUC 0.969, Prec 0.924, Rec 0.975, Thresh 0.447 | LR: 0.000025
Epoch 18: Train: Loss 0.0130, Acc 0.726 | Val: Loss 0.0090, Acc 0.891, BalAcc 0.885, F1 0.918, AUC 0.957, Prec 0.935, Rec 0.901, Thresh 0.504 | LR: 0.000025
⚠️ Early stopping after 18 epochs (no improvement for 7 epochs)
✅ Best model saved: ./output/models/best_model_fold_3.pth (Val BalAcc: 0.916)
⚡️ Test Results: Acc 0.782, BalAcc 0.746, F1 0.838, AUC 0.823, Precision 0.806, Recall 0.873 (threshold: 0.464)
📊 Confusion Matrix (Fold 3):
   [[TN:  78, FP:  48]
    [FN:  29, TP: 199]]
⚡ Avg Inference Time: 0.0088s per sample
📌 Final Magnification Importance (Fold 3): {'40': 0.2502613663673401, '100': 0.27578970789909363, '200': 0.22918696701526642, '400': 0.24476191401481628}
💾 Results saved to: ./output/results/fold_3_results.json

📊 Generating GradCAM visualizations for fold 3...
✅ Generated 5 GradCAM visualizations for fold 3

===== Fold 4 =====
Train patients: 52, Val Patients: 14, Test patients: 16
Training samples per epoch: {'total_samples_per_epoch': 2253, 'class_distribution': {0: 693, 1: 1560}, 'oversampling_factor': 2.04}
Validation samples: 255, Test samples: 287
Patients with full 4 mags: 52
Inner training samples: 2253, batch size: 16
Class weights: Benign=3.25, Malignant=0.22
Epoch 01: Train: Loss 0.0242, Acc 0.523 | Val: Loss 0.0887, Acc 0.667, BalAcc 0.492, F1 0.799, AUC 0.570, Prec 0.679, Rec 0.971, Thresh 0.100 | LR: 0.000100
✅ New best validation balanced accuracy: 0.492, threshold: 0.100
📊 Mag Importance (Val BalAcc: 0.492): {'40': 0.2802233099937439, '100': 0.2405647337436676, '200': 0.24226799607276917, '400': 0.23694398999214172}
Epoch 02: Train: Loss 0.0214, Acc 0.561 | Val: Loss 0.0400, Acc 0.722, BalAcc 0.648, F1 0.807, AUC 0.705, Prec 0.767, Rec 0.851, Thresh 0.439 | LR: 0.000100
✅ New best validation balanced accuracy: 0.648, threshold: 0.439
📊 Mag Importance (Val BalAcc: 0.648): {'40': 0.2775278687477112, '100': 0.23838123679161072, '200': 0.24566133320331573, '400': 0.23842957615852356}
Epoch 03: Train: Loss 0.0196, Acc 0.585 | Val: Loss 0.0273, Acc 0.796, BalAcc 0.712, F1 0.863, AUC 0.775, Prec 0.796, Rec 0.943, Thresh 0.383 | LR: 0.000100
✅ New best validation balanced accuracy: 0.712, threshold: 0.383
📊 Mag Importance (Val BalAcc: 0.712): {'40': 0.2790692448616028, '100': 0.23838624358177185, '200': 0.24443364143371582, '400': 0.23811085522174835}
Epoch 04: Train: Loss 0.0177, Acc 0.618 | Val: Loss 0.0174, Acc 0.824, BalAcc 0.752, F1 0.880, AUC 0.839, Prec 0.821, Rec 0.948, Thresh 0.415 | LR: 0.000100
✅ New best validation balanced accuracy: 0.752, threshold: 0.415
📊 Mag Importance (Val BalAcc: 0.752): {'40': 0.2781831920146942, '100': 0.24048641324043274, '200': 0.24555811285972595, '400': 0.23577222228050232}
Epoch 05: Train: Loss 0.0167, Acc 0.625 | Val: Loss 0.0182, Acc 0.843, BalAcc 0.773, F1 0.894, AUC 0.848, Prec 0.832, Rec 0.966, Thresh 0.431 | LR: 0.000100
✅ New best validation balanced accuracy: 0.773, threshold: 0.431
📊 Mag Importance (Val BalAcc: 0.773): {'40': 0.2849925756454468, '100': 0.23869240283966064, '200': 0.240578293800354, '400': 0.23573675751686096}
Epoch 06: Train: Loss 0.0150, Acc 0.647 | Val: Loss 0.0170, Acc 0.855, BalAcc 0.795, F1 0.900, AUC 0.868, Prec 0.848, Rec 0.960, Thresh 0.431 | LR: 0.000100
✅ New best validation balanced accuracy: 0.795, threshold: 0.431
📊 Mag Importance (Val BalAcc: 0.795): {'40': 0.2857261300086975, '100': 0.2372324913740158, '200': 0.238869309425354, '400': 0.23817209899425507}
Epoch 07: Train: Loss 0.0150, Acc 0.639 | Val: Loss 0.0136, Acc 0.871, BalAcc 0.823, F1 0.910, AUC 0.907, Prec 0.869, Rec 0.954, Thresh 0.464 | LR: 0.000100
✅ New best validation balanced accuracy: 0.823, threshold: 0.464
📊 Mag Importance (Val BalAcc: 0.823): {'40': 0.2854081988334656, '100': 0.2385648936033249, '200': 0.23822137713432312, '400': 0.2378055602312088}
Epoch 08: Train: Loss 0.0140, Acc 0.671 | Val: Loss 0.0221, Acc 0.890, BalAcc 0.834, F1 0.925, AUC 0.862, Prec 0.869, Rec 0.989, Thresh 0.447 | LR: 0.000100
✅ New best validation balanced accuracy: 0.834, threshold: 0.447
📊 Mag Importance (Val BalAcc: 0.834): {'40': 0.2885133624076843, '100': 0.23553523421287537, '200': 0.23915202915668488, '400': 0.23679938912391663}
Epoch 09: Train: Loss 0.0132, Acc 0.704 | Val: Loss 0.0208, Acc 0.890, BalAcc 0.834, F1 0.925, AUC 0.869, Prec 0.869, Rec 0.989, Thresh 0.447 | LR: 0.000100
Epoch 10: Train: Loss 0.0135, Acc 0.696 | Val: Loss 0.0311, Acc 0.890, BalAcc 0.837, F1 0.924, AUC 0.871, Prec 0.872, Rec 0.983, Thresh 0.415 | LR: 0.000100
✅ New best validation balanced accuracy: 0.837, threshold: 0.415
📊 Mag Importance (Val BalAcc: 0.837): {'40': 0.2887522280216217, '100': 0.23098035156726837, '200': 0.24145615100860596, '400': 0.23881129920482635}
Epoch 11: Train: Loss 0.0131, Acc 0.717 | Val: Loss 0.0311, Acc 0.882, BalAcc 0.835, F1 0.918, AUC 0.858, Prec 0.875, Rec 0.966, Thresh 0.447 | LR: 0.000100
Epoch 12: Train: Loss 0.0136, Acc 0.674 | Val: Loss 0.0215, Acc 0.906, BalAcc 0.852, F1 0.935, AUC 0.875, Prec 0.879, Rec 1.000, Thresh 0.351 | LR: 0.000100
✅ New best validation balanced accuracy: 0.852, threshold: 0.351
📊 Mag Importance (Val BalAcc: 0.852): {'40': 0.28657716512680054, '100': 0.22727254033088684, '200': 0.23954787850379944, '400': 0.24660241603851318}
Epoch 13: Train: Loss 0.0118, Acc 0.677 | Val: Loss 0.0209, Acc 0.898, BalAcc 0.846, F1 0.930, AUC 0.894, Prec 0.878, Rec 0.989, Thresh 0.423 | LR: 0.000100
Epoch 14: Train: Loss 0.0128, Acc 0.710 | Val: Loss 0.0128, Acc 0.898, BalAcc 0.840, F1 0.930, AUC 0.895, Prec 0.870, Rec 1.000, Thresh 0.391 | LR: 0.000100
Epoch 15: Train: Loss 0.0114, Acc 0.720 | Val: Loss 0.0134, Acc 0.922, BalAcc 0.900, F1 0.944, AUC 0.937, Prec 0.928, Rec 0.960, Thresh 0.504 | LR: 0.000100
✅ New best validation balanced accuracy: 0.900, threshold: 0.504
📊 Mag Importance (Val BalAcc: 0.900): {'40': 0.2868485450744629, '100': 0.23058417439460754, '200': 0.24251314997673035, '400': 0.2400541603565216}
Epoch 16: Train: Loss 0.0117, Acc 0.727 | Val: Loss 0.0098, Acc 0.894, BalAcc 0.853, F1 0.926, AUC 0.930, Prec 0.889, Rec 0.966, Thresh 0.447 | LR: 0.000100
Epoch 17: Train: Loss 0.0121, Acc 0.721 | Val: Loss 0.0127, Acc 0.914, BalAcc 0.864, F1 0.941, AUC 0.896, Prec 0.888, Rec 1.000, Thresh 0.407 | LR: 0.000100
Epoch 18: Train: Loss 0.0101, Acc 0.740 | Val: Loss 0.0154, Acc 0.914, BalAcc 0.874, F1 0.940, AUC 0.915, Prec 0.900, Rec 0.983, Thresh 0.439 | LR: 0.000100
Epoch 19: Train: Loss 0.0109, Acc 0.721 | Val: Loss 0.0195, Acc 0.918, BalAcc 0.877, F1 0.942, AUC 0.892, Prec 0.901, Rec 0.989, Thresh 0.439 | LR: 0.000050
Epoch 20: Train: Loss 0.0108, Acc 0.728 | Val: Loss 0.0092, Acc 0.902, BalAcc 0.865, F1 0.931, AUC 0.936, Prec 0.898, Rec 0.966, Thresh 0.415 | LR: 0.000050
Epoch 21: Train: Loss 0.0116, Acc 0.761 | Val: Loss 0.0123, Acc 0.906, BalAcc 0.855, F1 0.935, AUC 0.915, Prec 0.883, Rec 0.994, Thresh 0.383 | LR: 0.000050
Epoch 22: Train: Loss 0.0102, Acc 0.762 | Val: Loss 0.0116, Acc 0.914, BalAcc 0.867, F1 0.940, AUC 0.916, Prec 0.892, Rec 0.994, Thresh 0.407 | LR: 0.000050
⚠️ Early stopping after 22 epochs (no improvement for 7 epochs)
✅ Best model saved: ./output/models/best_model_fold_4.pth (Val BalAcc: 0.900)
⚡️ Test Results: Acc 0.969, BalAcc 0.957, F1 0.980, AUC 0.984, Precision 0.982, Recall 0.978 (threshold: 0.504)
📊 Confusion Matrix (Fold 4):
   [[TN:  59, FP:   4]
    [FN:   5, TP: 219]]
⚡ Avg Inference Time: 0.0098s per sample
📌 Final Magnification Importance (Fold 4): {'40': 0.27749568223953247, '100': 0.23429489135742188, '200': 0.24722377955913544, '400': 0.2409856617450714}
💾 Results saved to: ./output/results/fold_4_results.json

📊 Generating GradCAM visualizations for fold 4...
✅ Generated 5 GradCAM visualizations for fold 4

=== Cross-Validation Results ===
Acc:      0.860 ± 0.083
BalAcc:   0.849 ± 0.091
F1:       0.894 ± 0.064
AUC:      0.906 ± 0.076
Precision: 0.907 ± 0.069
Recall:    0.883 ± 0.070