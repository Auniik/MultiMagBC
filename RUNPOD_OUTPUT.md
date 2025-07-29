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