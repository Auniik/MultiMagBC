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
Class weights: Benign=2.60, Malignant=0.35
Epoch 01: Train: Loss 0.0266, Acc 0.559 | Val: Loss 0.0297, Acc 0.743, BalAcc 0.592, F1 0.841, AUC 0.689, Prec 0.735, Rec 0.982, Thresh 0.318 | LR: 0.000100
✅ New best validation balanced accuracy: 0.592, threshold: 0.318
📊 Mag Importance (Val BalAcc: 0.592): {'40': 0.2531374990940094, '100': 0.2623669505119324, '200': 0.2520248591899872, '400': 0.23247067630290985}
Epoch 02: Train: Loss 0.0233, Acc 0.616 | Val: Loss 0.0402, Acc 0.751, BalAcc 0.606, F1 0.845, AUC 0.689, Prec 0.742, Rec 0.982, Thresh 0.352 | LR: 0.000100
✅ New best validation balanced accuracy: 0.606, threshold: 0.352
📊 Mag Importance (Val BalAcc: 0.606): {'40': 0.2530558109283447, '100': 0.26292937994003296, '200': 0.25196728110313416, '400': 0.23204751312732697}
Epoch 03: Train: Loss 0.0202, Acc 0.639 | Val: Loss 0.0384, Acc 0.755, BalAcc 0.628, F1 0.844, AUC 0.686, Prec 0.755, Rec 0.958, Thresh 0.360 | LR: 0.000100
✅ New best validation balanced accuracy: 0.628, threshold: 0.360
📊 Mag Importance (Val BalAcc: 0.628): {'40': 0.2527481019496918, '100': 0.2630901336669922, '200': 0.25423020124435425, '400': 0.2299315482378006}
Epoch 04: Train: Loss 0.0181, Acc 0.650 | Val: Loss 0.0311, Acc 0.793, BalAcc 0.711, F1 0.860, AUC 0.750, Prec 0.806, Rec 0.922, Thresh 0.418 | LR: 0.000100
✅ New best validation balanced accuracy: 0.711, threshold: 0.418
📊 Mag Importance (Val BalAcc: 0.711): {'40': 0.2530653178691864, '100': 0.26376304030418396, '200': 0.2513372302055359, '400': 0.23183436691761017}
Epoch 05: Train: Loss 0.0179, Acc 0.656 | Val: Loss 0.0252, Acc 0.830, BalAcc 0.749, F1 0.886, AUC 0.767, Prec 0.825, Rec 0.958, Thresh 0.388 | LR: 0.000100
✅ New best validation balanced accuracy: 0.749, threshold: 0.388
📊 Mag Importance (Val BalAcc: 0.749): {'40': 0.25407180190086365, '100': 0.26290032267570496, '200': 0.2526819109916687, '400': 0.2303459644317627}
Epoch 06: Train: Loss 0.0170, Acc 0.667 | Val: Loss 0.0237, Acc 0.813, BalAcc 0.722, F1 0.877, AUC 0.766, Prec 0.808, Rec 0.958, Thresh 0.385 | LR: 0.000100
Epoch 07: Train: Loss 0.0159, Acc 0.661 | Val: Loss 0.0272, Acc 0.813, BalAcc 0.734, F1 0.875, AUC 0.731, Prec 0.818, Rec 0.940, Thresh 0.393 | LR: 0.000100
Epoch 08: Train: Loss 0.0153, Acc 0.715 | Val: Loss 0.0220, Acc 0.822, BalAcc 0.777, F1 0.874, AUC 0.798, Prec 0.856, Rec 0.892, Thresh 0.482 | LR: 0.000100
✅ New best validation balanced accuracy: 0.777, threshold: 0.482
📊 Mag Importance (Val BalAcc: 0.777): {'40': 0.25108635425567627, '100': 0.2638722062110901, '200': 0.2508954107761383, '400': 0.23414599895477295}
Epoch 09: Train: Loss 0.0134, Acc 0.693 | Val: Loss 0.0405, Acc 0.817, BalAcc 0.744, F1 0.876, AUC 0.729, Prec 0.825, Rec 0.934, Thresh 0.428 | LR: 0.000100
Epoch 10: Train: Loss 0.0157, Acc 0.689 | Val: Loss 0.0238, Acc 0.830, BalAcc 0.753, F1 0.886, AUC 0.759, Prec 0.828, Rec 0.952, Thresh 0.447 | LR: 0.000100
Epoch 11: Train: Loss 0.0131, Acc 0.709 | Val: Loss 0.0271, Acc 0.809, BalAcc 0.757, F1 0.866, AUC 0.765, Prec 0.842, Rec 0.892, Thresh 0.486 | LR: 0.000100
Epoch 12: Train: Loss 0.0138, Acc 0.700 | Val: Loss 0.0232, Acc 0.842, BalAcc 0.773, F1 0.893, AUC 0.771, Prec 0.841, Rec 0.952, Thresh 0.431 | LR: 0.000050
Epoch 13: Train: Loss 0.0148, Acc 0.697 | Val: Loss 0.0245, Acc 0.855, BalAcc 0.812, F1 0.898, AUC 0.815, Prec 0.875, Rec 0.922, Thresh 0.486 | LR: 0.000050
✅ New best validation balanced accuracy: 0.812, threshold: 0.486
📊 Mag Importance (Val BalAcc: 0.812): {'40': 0.250963032245636, '100': 0.2622959613800049, '200': 0.2535393536090851, '400': 0.23320162296295166}
Epoch 14: Train: Loss 0.0133, Acc 0.725 | Val: Loss 0.0283, Acc 0.851, BalAcc 0.806, F1 0.895, AUC 0.768, Prec 0.870, Rec 0.922, Thresh 0.479 | LR: 0.000050
Epoch 15: Train: Loss 0.0138, Acc 0.688 | Val: Loss 0.0273, Acc 0.834, BalAcc 0.760, F1 0.888, AUC 0.774, Prec 0.832, Rec 0.952, Thresh 0.420 | LR: 0.000050
Epoch 16: Train: Loss 0.0138, Acc 0.721 | Val: Loss 0.0272, Acc 0.813, BalAcc 0.734, F1 0.875, AUC 0.770, Prec 0.818, Rec 0.940, Thresh 0.417 | LR: 0.000050
Epoch 17: Train: Loss 0.0125, Acc 0.734 | Val: Loss 0.0189, Acc 0.846, BalAcc 0.769, F1 0.898, AUC 0.829, Prec 0.835, Rec 0.970, Thresh 0.447 | LR: 0.000025
Epoch 18: Train: Loss 0.0112, Acc 0.740 | Val: Loss 0.0216, Acc 0.834, BalAcc 0.790, F1 0.883, AUC 0.827, Prec 0.863, Rec 0.904, Thresh 0.517 | LR: 0.000025
Epoch 19: Train: Loss 0.0111, Acc 0.744 | Val: Loss 0.0327, Acc 0.834, BalAcc 0.779, F1 0.885, AUC 0.772, Prec 0.851, Rec 0.922, Thresh 0.465 | LR: 0.000025
Epoch 20: Train: Loss 0.0130, Acc 0.730 | Val: Loss 0.0287, Acc 0.813, BalAcc 0.722, F1 0.877, AUC 0.764, Prec 0.808, Rec 0.958, Thresh 0.421 | LR: 0.000025
⚠️ Early stopping after 20 epochs (no improvement for 7 epochs)
✅ Best model saved: ./output/models/best_model_fold_0.pth (Val BalAcc: 0.812)
⚡️ Test Results: Acc 0.895, BalAcc 0.909, F1 0.911, AUC 0.986, Precision 0.974, Recall 0.855 (threshold: 0.486)
📊 Confusion Matrix (Fold 0):
   [[TN: 127, FP:   5]
    [FN:  32, TP: 189]]
⚡ Avg Inference Time: 0.0074s per sample
📌 Final Magnification Importance (Fold 0): {'40': 0.2499537467956543, '100': 0.25957804918289185, '200': 0.25704845786094666, '400': 0.23341970145702362}
💾 Results saved to: ./output/results/fold_0_results.json

📊 Generating GradCAM visualizations for fold 0...
Traceback (most recent call last):
  File "/workspace/MultiMagBC/main.py", line 342, in <module>
    main()
  File "/workspace/MultiMagBC/main.py", line 297, in main
    cams = gradcam.get_cam(single_images, target_class=predicted.item())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/MultiMagBC/evaluate/gradcam.py", line 20, in get_cam
    layer = getattr(self.model.extractors[f'extractor_{mag}x'], 'conv_head')
                    ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1729, in __getattr__
    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
AttributeError: 'MultiMagLightweightCNN' object has no attribute 'extractors'