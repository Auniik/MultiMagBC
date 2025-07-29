============================================================
🎯 STARTING ENHANCED TRAINING
============================================================
=== Fold-wise Dataset Summary ===
Fold 0: Train patients: 65 (images=6246, B/M = 19/46); Test patients: 17 (images=1663, B/M = 5/12)
Fold 1: Train patients: 65 (images=6228, B/M = 19/46); Test patients: 17 (images=1681, B/M = 5/12)
Fold 2: Train patients: 66 (images=6407, B/M = 19/47); Test patients: 16 (images=1502, B/M = 5/11)
Fold 3: Train patients: 66 (images=6217, B/M = 19/47); Test patients: 16 (images=1692, B/M = 5/11)
Fold 4: Train patients: 66 (images=6538, B/M = 20/46); Test patients: 16 (images=1371, B/M = 4/12)
Saved visualizations to ./output/dataset

🎯 Running 5 fold(s) out of 5 available

==================================================
Training Fold 1/5
==================================================
Dataset created: 540 samples from 55 patients
  Label distribution: Benign=160, Malignant=380
  Samples per patient: min=1, max=10, avg=9.8
Dataset created: 50 samples from 10 patients
  Label distribution: Benign=15, Malignant=35
  Samples per patient: min=5, max=5, avg=5.0
Dataset created: 17 samples from 17 patients
  Label distribution: Benign=5, Malignant=12
  Samples per patient: min=1, max=1, avg=1.0
Epoch 1 Training: 100%|----| 34/34 [00:18<00:00,  1.89it/s, Loss=0.6588]
Validation: 100%|----| 4/4 [00:01<00:00,  3.06it/s]
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1731: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
Epoch   1/50 | Time: 19.3s | LR: 1.33e-04 | Train Loss: 0.6851, Acc: 0.5667, AUC: 0.5884 | Val Loss: 0.7069, Acc: 0.3000, Bal Acc: 0.5000, AUC: 0.5771
    ★ New best validation balanced accuracy: 0.5000
Epoch 2 Training: 100%|----| 34/34 [00:16<00:00,  2.02it/s, Loss=0.4964]
Validation: 100%|----| 4/4 [00:01<00:00,  3.36it/s]
Epoch   2/50 | Time: 18.0s | LR: 2.00e-04 | Train Loss: 0.6517, Acc: 0.6500, AUC: 0.7205 | Val Loss: 0.6924, Acc: 0.8000, Bal Acc: 0.7619, AUC: 0.7067
    ★ New best validation balanced accuracy: 0.7619
Epoch 3 Training: 100%|----| 34/34 [00:17<00:00,  1.98it/s, Loss=0.4784]
Validation: 100%|----| 4/4 [00:01<00:00,  3.32it/s]
Epoch   3/50 | Time: 18.4s | LR: 2.00e-04 | Train Loss: 0.6168, Acc: 0.7056, AUC: 0.7768 | Val Loss: 0.6557, Acc: 0.7800, Bal Acc: 0.7476, AUC: 0.7314
Epoch 4 Training: 100%|----| 34/34 [00:17<00:00,  1.96it/s, Loss=0.5830]
Validation: 100%|----| 4/4 [00:01<00:00,  3.32it/s]
Epoch   4/50 | Time: 18.6s | LR: 1.95e-04 | Train Loss: 0.6241, Acc: 0.7185, AUC: 0.7752 | Val Loss: 0.6322, Acc: 0.7800, Bal Acc: 0.7476, AUC: 0.7448
Epoch 5 Training: 100%|----| 34/34 [00:17<00:00,  1.90it/s, Loss=0.9361]
Validation: 100%|----| 4/4 [00:01<00:00,  3.27it/s]
Epoch   5/50 | Time: 19.1s | LR: 1.81e-04 | Train Loss: 0.6074, Acc: 0.7444, AUC: 0.8230 | Val Loss: 0.6163, Acc: 0.7800, Bal Acc: 0.7476, AUC: 0.7714
Epoch 6 Training: 100%|----| 34/34 [00:18<00:00,  1.80it/s, Loss=0.5035]
Validation: 100%|----| 4/4 [00:01<00:00,  3.45it/s]
Epoch   6/50 | Time: 20.1s | LR: 1.59e-04 | Train Loss: 0.5977, Acc: 0.7778, AUC: 0.8201 | Val Loss: 0.6791, Acc: 0.8000, Bal Acc: 0.7619, AUC: 0.6533
Epoch 7 Training: 100%|----| 34/34 [00:18<00:00,  1.82it/s, Loss=0.4458]
Validation: 100%|----| 4/4 [00:01<00:00,  3.21it/s]
Epoch   7/50 | Time: 19.9s | LR: 1.31e-04 | Train Loss: 0.5836, Acc: 0.7704, AUC: 0.8347 | Val Loss: 0.6294, Acc: 0.8400, Bal Acc: 0.7905, AUC: 0.7543
    ★ New best validation balanced accuracy: 0.7905
Epoch 8 Training: 100%|----| 34/34 [00:17<00:00,  1.94it/s, Loss=0.5278]
Validation: 100%|----| 4/4 [00:01<00:00,  3.34it/s]
Epoch   8/50 | Time: 18.7s | LR: 1.01e-04 | Train Loss: 0.6039, Acc: 0.7444, AUC: 0.7971 | Val Loss: 0.6264, Acc: 0.8000, Bal Acc: 0.7619, AUC: 0.7543
Epoch 9 Training: 100%|----| 34/34 [00:19<00:00,  1.78it/s, Loss=0.5451]
Validation: 100%|----| 4/4 [00:01<00:00,  3.33it/s]
Epoch   9/50 | Time: 20.3s | LR: 6.98e-05 | Train Loss: 0.6083, Acc: 0.7593, AUC: 0.8014 | Val Loss: 0.6832, Acc: 0.8000, Bal Acc: 0.7619, AUC: 0.7219
Epoch 10 Training: 100%|----| 34/34 [00:18<00:00,  1.85it/s, Loss=0.5958]
Validation: 100%|----| 4/4 [00:01<00:00,  3.32it/s]
Epoch  10/50 | Time: 19.6s | LR: 4.20e-05 | Train Loss: 0.5994, Acc: 0.7722, AUC: 0.8263 | Val Loss: 0.6963, Acc: 0.8000, Bal Acc: 0.7619, AUC: 0.6914
Epoch 11 Training: 100%|----| 34/34 [00:18<00:00,  1.88it/s, Loss=0.5752]
Validation: 100%|----| 4/4 [00:01<00:00,  3.30it/s]
Epoch  11/50 | Time: 19.3s | LR: 2.00e-05 | Train Loss: 0.5628, Acc: 0.7907, AUC: 0.8595 | Val Loss: 0.6655, Acc: 0.8000, Bal Acc: 0.7619, AUC: 0.7162
Epoch 12 Training: 100%|----| 34/34 [00:19<00:00,  1.70it/s, Loss=0.4167]
Validation: 100%|----| 4/4 [00:01<00:00,  3.21it/s]
Epoch  12/50 | Time: 21.2s | LR: 5.87e-06 | Train Loss: 0.5671, Acc: 0.7796, AUC: 0.8444 | Val Loss: 0.6877, Acc: 0.8000, Bal Acc: 0.7619, AUC: 0.7162
Epoch 13 Training: 100%|----| 34/34 [00:18<00:00,  1.86it/s, Loss=0.4914]
Validation: 100%|----| 4/4 [00:01<00:00,  3.39it/s]
Epoch  13/50 | Time: 19.5s | LR: 2.00e-04 | Train Loss: 0.5907, Acc: 0.7741, AUC: 0.8108 | Val Loss: 0.7055, Acc: 0.8000, Bal Acc: 0.7619, AUC: 0.7086
Epoch 14 Training: 100%|----| 34/34 [00:18<00:00,  1.89it/s, Loss=0.4906]
Validation: 100%|----| 4/4 [00:01<00:00,  3.07it/s]
Epoch  14/50 | Time: 19.3s | LR: 1.99e-04 | Train Loss: 0.5638, Acc: 0.7796, AUC: 0.8476 | Val Loss: 0.6635, Acc: 0.7400, Bal Acc: 0.7190, AUC: 0.7467
Epoch 15 Training: 100%|----| 34/34 [00:17<00:00,  1.97it/s, Loss=0.4246]
Validation: 100%|----| 4/4 [00:01<00:00,  3.36it/s]
Epoch  15/50 | Time: 18.5s | LR: 1.95e-04 | Train Loss: 0.5801, Acc: 0.7907, AUC: 0.8375 | Val Loss: 0.6735, Acc: 0.8000, Bal Acc: 0.7619, AUC: 0.8019
Epoch 16 Training: 100%|----| 34/34 [00:20<00:00,  1.70it/s, Loss=0.5092]
Validation: 100%|----| 4/4 [00:01<00:00,  3.42it/s]
Epoch  16/50 | Time: 21.2s | LR: 1.89e-04 | Train Loss: 0.6067, Acc: 0.7852, AUC: 0.8116 | Val Loss: 0.6753, Acc: 0.8000, Bal Acc: 0.7619, AUC: 0.8095
Epoch 17 Training: 100%|----| 34/34 [00:18<00:00,  1.81it/s, Loss=0.6847]
Validation: 100%|----| 4/4 [00:01<00:00,  3.39it/s]
Epoch  17/50 | Time: 19.9s | LR: 1.81e-04 | Train Loss: 0.5837, Acc: 0.7556, AUC: 0.8218 | Val Loss: 0.7783, Acc: 0.8000, Bal Acc: 0.7619, AUC: 0.6343
    ⏹ Early stopping triggered at epoch 17
/workspace/MultiMagBC/training/enhanced_train_k_fold.py:434: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load(os.path.join(self.config.MODELS_DIR, f"best_model_fold_{fold_idx}.pth")))
Validation: 100%|----| 2/2 [00:01<00:00,  1.44it/s]
Dataset created: 17 samples from 17 patients
  Label distribution: Benign=5, Malignant=12
  Samples per patient: min=1, max=1, avg=1.0
Dataset created: 17 samples from 17 patients
  Label distribution: Benign=5, Malignant=12
  Samples per patient: min=1, max=1, avg=1.0
Dataset created: 17 samples from 17 patients
  Label distribution: Benign=5, Malignant=12
  Samples per patient: min=1, max=1, avg=1.0
Dataset created: 17 samples from 17 patients
  Label distribution: Benign=5, Malignant=12
  Samples per patient: min=1, max=1, avg=1.0
Dataset created: 17 samples from 17 patients
  Label distribution: Benign=5, Malignant=12
  Samples per patient: min=1, max=1, avg=1.0
Dataset created: 17 samples from 17 patients
  Label distribution: Benign=5, Malignant=12
  Samples per patient: min=1, max=1, avg=1.0
Dataset created: 17 samples from 17 patients
  Label distribution: Benign=5, Malignant=12
  Samples per patient: min=1, max=1, avg=1.0
Dataset created: 17 samples from 17 patients
  Label distribution: Benign=5, Malignant=12
  Samples per patient: min=1, max=1, avg=1.0
Dataset created: 17 samples from 17 patients
  Label distribution: Benign=5, Malignant=12
  Samples per patient: min=1, max=1, avg=1.0
Dataset created: 17 samples from 17 patients
  Label distribution: Benign=5, Malignant=12
  Samples per patient: min=1, max=1, avg=1.0
Dataset created: 17 samples from 17 patients
  Label distribution: Benign=5, Malignant=12
  Samples per patient: min=1, max=1, avg=1.0
Dataset created: 17 samples from 17 patients
  Label distribution: Benign=5, Malignant=12
  Samples per patient: min=1, max=1, avg=1.0
Dataset created: 17 samples from 17 patients
  Label distribution: Benign=5, Malignant=12
  Samples per patient: min=1, max=1, avg=1.0

TTA Results - Accuracy: 0.8824, Balanced Accuracy: 0.8583, AUC: 0.9667

==================================================
Training Fold 2/5
==================================================
Dataset created: 536 samples from 55 patients
  Label distribution: Benign=160, Malignant=376
  Samples per patient: min=1, max=10, avg=9.7
Dataset created: 50 samples from 10 patients
  Label distribution: Benign=15, Malignant=35
  Samples per patient: min=5, max=5, avg=5.0
Dataset created: 17 samples from 17 patients
  Label distribution: Benign=5, Malignant=12
  Samples per patient: min=1, max=1, avg=1.0
Epoch 1 Training: 100%|----| 34/34 [00:17<00:00,  1.90it/s, Loss=0.6630]
Validation: 100%|----| 4/4 [00:01<00:00,  3.18it/s]
Epoch   1/50 | Time: 19.1s | LR: 1.33e-04 | Train Loss: 0.6881, Acc: 0.6511, AUC: 0.6047 | Val Loss: 0.6656, Acc: 0.5200, Bal Acc: 0.6571, AUC: 0.9295
    ★ New best validation balanced accuracy: 0.6571
Epoch 2 Training: 100%|----| 34/34 [00:20<00:00,  1.70it/s, Loss=0.4872]
Validation: 100%|----| 4/4 [00:01<00:00,  3.50it/s]
Epoch   2/50 | Time: 21.2s | LR: 2.00e-04 | Train Loss: 0.6595, Acc: 0.6978, AUC: 0.7213 | Val Loss: 0.4465, Acc: 0.9000, Bal Acc: 0.9286, AUC: 1.0000
    ★ New best validation balanced accuracy: 0.9286
Epoch 3 Training: 100%|----| 34/34 [00:17<00:00,  1.92it/s, Loss=0.8073]
Validation: 100%|----| 4/4 [00:01<00:00,  3.50it/s]
Epoch   3/50 | Time: 18.9s | LR: 2.00e-04 | Train Loss: 0.6189, Acc: 0.7556, AUC: 0.8042 | Val Loss: 0.4184, Acc: 0.9000, Bal Acc: 0.9286, AUC: 1.0000
Epoch 4 Training: 100%|----| 34/34 [00:19<00:00,  1.77it/s, Loss=0.4553]
Validation: 100%|----| 4/4 [00:01<00:00,  3.23it/s]
Epoch   4/50 | Time: 20.5s | LR: 1.95e-04 | Train Loss: 0.6144, Acc: 0.7369, AUC: 0.7934 | Val Loss: 0.4000, Acc: 0.9000, Bal Acc: 0.9286, AUC: 1.0000
Epoch 5 Training: 100%|----| 34/34 [00:17<00:00,  1.92it/s, Loss=0.4375]
Validation: 100%|----| 4/4 [00:01<00:00,  3.34it/s]
Epoch   5/50 | Time: 18.9s | LR: 1.81e-04 | Train Loss: 0.6024, Acc: 0.7388, AUC: 0.8002 | Val Loss: 0.3997, Acc: 0.9200, Bal Acc: 0.9429, AUC: 1.0000
    ★ New best validation balanced accuracy: 0.9429
Epoch 6 Training: 100%|----| 34/34 [00:18<00:00,  1.82it/s, Loss=0.4180]
Validation: 100%|----| 4/4 [00:01<00:00,  3.46it/s]
Epoch   6/50 | Time: 19.8s | LR: 1.59e-04 | Train Loss: 0.5904, Acc: 0.7910, AUC: 0.8415 | Val Loss: 0.3938, Acc: 0.9200, Bal Acc: 0.9429, AUC: 1.0000
Epoch 7 Training: 100%|----| 34/34 [00:17<00:00,  1.92it/s, Loss=0.6478]
Validation: 100%|----| 4/4 [00:01<00:00,  3.48it/s]
Epoch   7/50 | Time: 18.9s | LR: 1.31e-04 | Train Loss: 0.5810, Acc: 0.7966, AUC: 0.8278 | Val Loss: 0.4275, Acc: 0.9000, Bal Acc: 0.9286, AUC: 0.9924
Epoch 8 Training: 100%|----| 34/34 [00:18<00:00,  1.84it/s, Loss=0.3848]
Validation: 100%|----| 4/4 [00:01<00:00,  3.39it/s]
Epoch   8/50 | Time: 19.7s | LR: 1.01e-04 | Train Loss: 0.6021, Acc: 0.7724, AUC: 0.8089 | Val Loss: 0.4304, Acc: 0.9000, Bal Acc: 0.9286, AUC: 0.9924
Epoch 9 Training: 100%|----| 34/34 [00:19<00:00,  1.76it/s, Loss=0.4295]
Validation: 100%|----| 4/4 [00:01<00:00,  3.58it/s]
Epoch   9/50 | Time: 20.4s | LR: 6.98e-05 | Train Loss: 0.5753, Acc: 0.8004, AUC: 0.8124 | Val Loss: 0.4610, Acc: 0.9000, Bal Acc: 0.9286, AUC: 0.9867
Epoch 10 Training: 100%|----| 34/34 [00:18<00:00,  1.84it/s, Loss=0.4545]
Validation: 100%|----| 4/4 [00:01<00:00,  3.57it/s]
Epoch  10/50 | Time: 19.6s | LR: 4.20e-05 | Train Loss: 0.5898, Acc: 0.7929, AUC: 0.8142 | Val Loss: 0.4519, Acc: 0.9000, Bal Acc: 0.9286, AUC: 0.9962
Epoch 11 Training: 100%|----| 34/34 [00:18<00:00,  1.89it/s, Loss=0.5461]
Validation: 100%|----| 4/4 [00:01<00:00,  3.59it/s]
Epoch  11/50 | Time: 19.1s | LR: 2.00e-05 | Train Loss: 0.5820, Acc: 0.7910, AUC: 0.8166 | Val Loss: 0.4196, Acc: 0.9200, Bal Acc: 0.9429, AUC: 0.9943
Epoch 12 Training: 100%|----| 34/34 [00:17<00:00,  1.89it/s, Loss=0.5112]
Validation: 100%|----| 4/4 [00:01<00:00,  3.49it/s]
Epoch  12/50 | Time: 19.1s | LR: 5.87e-06 | Train Loss: 0.5707, Acc: 0.8097, AUC: 0.8448 | Val Loss: 0.4277, Acc: 0.9000, Bal Acc: 0.9286, AUC: 0.9981
Epoch 13 Training: 100%|----| 34/34 [00:18<00:00,  1.84it/s, Loss=0.8458]
Validation: 100%|----| 4/4 [00:01<00:00,  3.50it/s]
Epoch  13/50 | Time: 19.6s | LR: 2.00e-04 | Train Loss: 0.5568, Acc: 0.8302, AUC: 0.8500 | Val Loss: 0.4239, Acc: 0.9000, Bal Acc: 0.9286, AUC: 0.9962
Epoch 14 Training: 100%|----| 34/34 [00:17<00:00,  1.95it/s, Loss=0.4678]
Validation: 100%|----| 4/4 [00:01<00:00,  3.51it/s]
Epoch  14/50 | Time: 18.6s | LR: 1.99e-04 | Train Loss: 0.5837, Acc: 0.7929, AUC: 0.8433 | Val Loss: 0.4260, Acc: 0.9000, Bal Acc: 0.9286, AUC: 1.0000
Epoch 15 Training: 100%|----| 34/34 [00:17<00:00,  1.98it/s, Loss=0.4300]
Validation: 100%|----| 4/4 [00:01<00:00,  3.44it/s]
Epoch  15/50 | Time: 18.3s | LR: 1.95e-04 | Train Loss: 0.5739, Acc: 0.8060, AUC: 0.8323 | Val Loss: 0.3986, Acc: 0.9200, Bal Acc: 0.9429, AUC: 0.9981
    ⏹ Early stopping triggered at epoch 15
/workspace/MultiMagBC/training/enhanced_train_k_fold.py:434: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load(os.path.join(self.config.MODELS_DIR, f"best_model_fold_{fold_idx}.pth")))
Validation: 100%|----| 2/2 [00:01<00:00,  1.43it/s]
Dataset created: 17 samples from 17 patients
  Label distribution: Benign=5, Malignant=12
  Samples per patient: min=1, max=1, avg=1.0
Dataset created: 17 samples from 17 patients
  Label distribution: Benign=5, Malignant=12
  Samples per patient: min=1, max=1, avg=1.0
Dataset created: 17 samples from 17 patients
  Label distribution: Benign=5, Malignant=12
  Samples per patient: min=1, max=1, avg=1.0
Dataset created: 17 samples from 17 patients
  Label distribution: Benign=5, Malignant=12
  Samples per patient: min=1, max=1, avg=1.0
Dataset created: 17 samples from 17 patients
  Label distribution: Benign=5, Malignant=12
  Samples per patient: min=1, max=1, avg=1.0
Dataset created: 17 samples from 17 patients
  Label distribution: Benign=5, Malignant=12
  Samples per patient: min=1, max=1, avg=1.0
Dataset created: 17 samples from 17 patients
  Label distribution: Benign=5, Malignant=12
  Samples per patient: min=1, max=1, avg=1.0
Dataset created: 17 samples from 17 patients
  Label distribution: Benign=5, Malignant=12
  Samples per patient: min=1, max=1, avg=1.0
Dataset created: 17 samples from 17 patients
  Label distribution: Benign=5, Malignant=12
  Samples per patient: min=1, max=1, avg=1.0
Dataset created: 17 samples from 17 patients
  Label distribution: Benign=5, Malignant=12
  Samples per patient: min=1, max=1, avg=1.0
Dataset created: 17 samples from 17 patients
  Label distribution: Benign=5, Malignant=12
  Samples per patient: min=1, max=1, avg=1.0
Dataset created: 17 samples from 17 patients
  Label distribution: Benign=5, Malignant=12
  Samples per patient: min=1, max=1, avg=1.0
Dataset created: 17 samples from 17 patients
  Label distribution: Benign=5, Malignant=12
  Samples per patient: min=1, max=1, avg=1.0

TTA Results - Accuracy: 0.8824, Balanced Accuracy: 0.8583, AUC: 0.7500

==================================================
Training Fold 3/5
==================================================
Dataset created: 548 samples from 56 patients
  Label distribution: Benign=160, Malignant=388
  Samples per patient: min=1, max=10, avg=9.8
Dataset created: 50 samples from 10 patients
  Label distribution: Benign=15, Malignant=35
  Samples per patient: min=5, max=5, avg=5.0
Dataset created: 16 samples from 16 patients
  Label distribution: Benign=5, Malignant=11
  Samples per patient: min=1, max=1, avg=1.0
Epoch 1 Training: 100%|----| 35/35 [00:21<00:00,  1.63it/s, Loss=0.6515]
Validation: 100%|----| 4/4 [00:01<00:00,  3.24it/s]
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1731: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
Epoch   1/50 | Time: 22.8s | LR: 1.33e-04 | Train Loss: 0.7131, Acc: 0.4909, AUC: 0.5945 | Val Loss: 0.6670, Acc: 0.3000, Bal Acc: 0.5000, AUC: 1.0000
    ★ New best validation balanced accuracy: 0.5000
Epoch 2 Training: 100%|----| 35/35 [00:18<00:00,  1.91it/s, Loss=0.7012]
Validation: 100%|----| 4/4 [00:01<00:00,  3.36it/s]
Epoch   2/50 | Time: 19.6s | LR: 2.00e-04 | Train Loss: 0.6773, Acc: 0.5602, AUC: 0.6639 | Val Loss: 0.4313, Acc: 1.0000, Bal Acc: 1.0000, AUC: 1.0000
    ★ New best validation balanced accuracy: 1.0000
Epoch 3 Training: 100%|----| 35/35 [00:17<00:00,  1.98it/s, Loss=0.5390]
Validation: 100%|----| 4/4 [00:01<00:00,  3.45it/s]
Epoch   3/50 | Time: 18.9s | LR: 2.00e-04 | Train Loss: 0.6466, Acc: 0.5675, AUC: 0.7057 | Val Loss: 0.5051, Acc: 0.7800, Bal Acc: 0.8429, AUC: 1.0000
Epoch 4 Training: 100%|----| 35/35 [00:18<00:00,  1.90it/s, Loss=0.5787]
Validation: 100%|----| 4/4 [00:01<00:00,  3.36it/s]
Epoch   4/50 | Time: 19.7s | LR: 1.95e-04 | Train Loss: 0.6335, Acc: 0.6496, AUC: 0.7594 | Val Loss: 0.4379, Acc: 0.9600, Bal Acc: 0.9714, AUC: 1.0000
Epoch 5 Training: 100%|----| 35/35 [00:18<00:00,  1.88it/s, Loss=0.5029]
Validation: 100%|----| 4/4 [00:01<00:00,  3.43it/s]
Epoch   5/50 | Time: 19.8s | LR: 1.81e-04 | Train Loss: 0.6108, Acc: 0.6989, AUC: 0.7830 | Val Loss: 0.4244, Acc: 1.0000, Bal Acc: 1.0000, AUC: 1.0000
Epoch 6 Training: 100%|----| 35/35 [00:17<00:00,  2.01it/s, Loss=0.4982]
Validation: 100%|----| 4/4 [00:01<00:00,  3.46it/s]
Epoch   6/50 | Time: 18.6s | LR: 1.59e-04 | Train Loss: 0.6133, Acc: 0.7080, AUC: 0.7753 | Val Loss: 0.4107, Acc: 1.0000, Bal Acc: 1.0000, AUC: 1.0000
Epoch 7 Training: 100%|----| 35/35 [00:17<00:00,  2.01it/s, Loss=0.5301]
Validation: 100%|----| 4/4 [00:01<00:00,  3.34it/s]
Epoch   7/50 | Time: 18.6s | LR: 1.31e-04 | Train Loss: 0.6005, Acc: 0.7482, AUC: 0.8044 | Val Loss: 0.3996, Acc: 1.0000, Bal Acc: 1.0000, AUC: 1.0000
Epoch 8 Training: 100%|----| 35/35 [00:19<00:00,  1.81it/s, Loss=0.8946]
Validation: 100%|----| 4/4 [00:01<00:00,  3.40it/s]
Epoch   8/50 | Time: 20.5s | LR: 1.01e-04 | Train Loss: 0.6161, Acc: 0.7701, AUC: 0.7927 | Val Loss: 0.4014, Acc: 1.0000, Bal Acc: 1.0000, AUC: 1.0000
Epoch 9 Training: 100%|----| 35/35 [00:18<00:00,  1.93it/s, Loss=1.2864]
Validation: 100%|----| 4/4 [00:01<00:00,  3.34it/s]
Epoch   9/50 | Time: 19.3s | LR: 6.98e-05 | Train Loss: 0.6339, Acc: 0.7737, AUC: 0.7803 | Val Loss: 0.4112, Acc: 1.0000, Bal Acc: 1.0000, AUC: 1.0000
Epoch 10 Training: 100%|----| 35/35 [00:17<00:00,  1.98it/s, Loss=1.4222]
Validation: 100%|----| 4/4 [00:01<00:00,  3.32it/s]
Epoch  10/50 | Time: 18.9s | LR: 4.20e-05 | Train Loss: 0.6087, Acc: 0.7682, AUC: 0.8099 | Val Loss: 0.3813, Acc: 1.0000, Bal Acc: 1.0000, AUC: 1.0000
Epoch 11 Training: 100%|----| 35/35 [00:17<00:00,  1.97it/s, Loss=0.5790]
Validation: 100%|----| 4/4 [00:01<00:00,  3.05it/s]
Epoch  11/50 | Time: 19.1s | LR: 2.00e-05 | Train Loss: 0.5812, Acc: 0.7847, AUC: 0.8474 | Val Loss: 0.3846, Acc: 1.0000, Bal Acc: 1.0000, AUC: 1.0000
Epoch 12 Training: 100%|----| 35/35 [00:18<00:00,  1.93it/s, Loss=0.9601]
Validation: 100%|----| 4/4 [00:01<00:00,  3.38it/s]
Epoch  12/50 | Time: 19.3s | LR: 5.87e-06 | Train Loss: 0.5653, Acc: 0.7737, AUC: 0.8384 | Val Loss: 0.3915, Acc: 1.0000, Bal Acc: 1.0000, AUC: 1.0000
    ⏹ Early stopping triggered at epoch 12
/workspace/MultiMagBC/training/enhanced_train_k_fold.py:434: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load(os.path.join(self.config.MODELS_DIR, f"best_model_fold_{fold_idx}.pth")))
Validation: 100%|----| 1/1 [00:01<00:00,  1.37s/it]
Dataset created: 16 samples from 16 patients
  Label distribution: Benign=5, Malignant=11
  Samples per patient: min=1, max=1, avg=1.0
Dataset created: 16 samples from 16 patients
  Label distribution: Benign=5, Malignant=11
  Samples per patient: min=1, max=1, avg=1.0
Dataset created: 16 samples from 16 patients
  Label distribution: Benign=5, Malignant=11
  Samples per patient: min=1, max=1, avg=1.0
Dataset created: 16 samples from 16 patients
  Label distribution: Benign=5, Malignant=11
  Samples per patient: min=1, max=1, avg=1.0
Dataset created: 16 samples from 16 patients
  Label distribution: Benign=5, Malignant=11
  Samples per patient: min=1, max=1, avg=1.0
Dataset created: 16 samples from 16 patients
  Label distribution: Benign=5, Malignant=11
  Samples per patient: min=1, max=1, avg=1.0
Dataset created: 16 samples from 16 patients
  Label distribution: Benign=5, Malignant=11
  Samples per patient: min=1, max=1, avg=1.0
Dataset created: 16 samples from 16 patients
  Label distribution: Benign=5, Malignant=11
  Samples per patient: min=1, max=1, avg=1.0
Dataset created: 16 samples from 16 patients
  Label distribution: Benign=5, Malignant=11
  Samples per patient: min=1, max=1, avg=1.0
Dataset created: 16 samples from 16 patients
  Label distribution: Benign=5, Malignant=11
  Samples per patient: min=1, max=1, avg=1.0
Dataset created: 16 samples from 16 patients
  Label distribution: Benign=5, Malignant=11
  Samples per patient: min=1, max=1, avg=1.0
Dataset created: 16 samples from 16 patients
  Label distribution: Benign=5, Malignant=11
  Samples per patient: min=1, max=1, avg=1.0
Dataset created: 16 samples from 16 patients
  Label distribution: Benign=5, Malignant=11
  Samples per patient: min=1, max=1, avg=1.0

TTA Results - Accuracy: 0.8750, Balanced Accuracy: 0.8545, AUC: 0.8182

==================================================
Training Fold 4/5
==================================================
Dataset created: 556 samples from 56 patients
  Label distribution: Benign=160, Malignant=396
  Samples per patient: min=9, max=10, avg=9.9
Dataset created: 50 samples from 10 patients
  Label distribution: Benign=15, Malignant=35
  Samples per patient: min=5, max=5, avg=5.0
Dataset created: 16 samples from 16 patients
  Label distribution: Benign=5, Malignant=11
  Samples per patient: min=1, max=1, avg=1.0
Epoch 1 Training: 100%|----| 35/35 [00:19<00:00,  1.79it/s, Loss=0.5956]
Validation: 100%|----| 4/4 [00:01<00:00,  3.26it/s]
Epoch   1/50 | Time: 20.8s | LR: 1.33e-04 | Train Loss: 0.6969, Acc: 0.4712, AUC: 0.6148 | Val Loss: 0.6180, Acc: 0.7000, Bal Acc: 0.7857, AUC: 0.9657
    ★ New best validation balanced accuracy: 0.7857
Epoch 2 Training: 100%|----| 35/35 [00:19<00:00,  1.76it/s, Loss=0.5997]
Validation: 100%|----| 4/4 [00:01<00:00,  3.35it/s]
Epoch   2/50 | Time: 21.1s | LR: 2.00e-04 | Train Loss: 0.6499, Acc: 0.5845, AUC: 0.7266 | Val Loss: 0.5062, Acc: 0.8800, Bal Acc: 0.8952, AUC: 0.9695
    ★ New best validation balanced accuracy: 0.8952
Epoch 3 Training: 100%|----| 35/35 [00:17<00:00,  1.98it/s, Loss=0.7143]
Validation: 100%|----| 4/4 [00:01<00:00,  3.32it/s]
Epoch   3/50 | Time: 18.9s | LR: 2.00e-04 | Train Loss: 0.6130, Acc: 0.6745, AUC: 0.7927 | Val Loss: 0.4790, Acc: 0.8800, Bal Acc: 0.8190, AUC: 0.9657
Epoch 4 Training: 100%|----| 35/35 [00:18<00:00,  1.89it/s, Loss=0.9452]
Validation: 100%|----| 4/4 [00:01<00:00,  3.19it/s]
Epoch   4/50 | Time: 19.8s | LR: 1.95e-04 | Train Loss: 0.5986, Acc: 0.7662, AUC: 0.8097 | Val Loss: 0.4874, Acc: 0.9000, Bal Acc: 0.8905, AUC: 0.9752
Epoch 5 Training: 100%|----| 35/35 [00:18<00:00,  1.91it/s, Loss=0.5065]
Validation: 100%|----| 4/4 [00:01<00:00,  3.35it/s]
Epoch   5/50 | Time: 19.5s | LR: 1.81e-04 | Train Loss: 0.5886, Acc: 0.7716, AUC: 0.8295 | Val Loss: 0.4995, Acc: 0.9000, Bal Acc: 0.8333, AUC: 0.9771
Epoch 6 Training: 100%|----| 35/35 [00:19<00:00,  1.78it/s, Loss=0.4266]
Validation: 100%|----| 4/4 [00:01<00:00,  3.33it/s]
Epoch   6/50 | Time: 20.9s | LR: 1.59e-04 | Train Loss: 0.5580, Acc: 0.8237, AUC: 0.8573 | Val Loss: 0.4937, Acc: 0.8800, Bal Acc: 0.8571, AUC: 0.9276
Epoch 7 Training: 100%|----| 35/35 [00:17<00:00,  2.04it/s, Loss=0.4435]
Validation: 100%|----| 4/4 [00:01<00:00,  2.94it/s]
Epoch   7/50 | Time: 18.6s | LR: 1.31e-04 | Train Loss: 0.5621, Acc: 0.8058, AUC: 0.8654 | Val Loss: 0.5185, Acc: 0.8800, Bal Acc: 0.8190, AUC: 0.8952
Epoch 8 Training: 100%|----| 35/35 [00:18<00:00,  1.87it/s, Loss=0.4798]
Validation: 100%|----| 4/4 [00:01<00:00,  3.39it/s]
Epoch   8/50 | Time: 19.9s | LR: 1.01e-04 | Train Loss: 0.5596, Acc: 0.8129, AUC: 0.8452 | Val Loss: 0.4732, Acc: 0.9200, Bal Acc: 0.8857, AUC: 0.9543
Epoch 9 Training: 100%|----| 35/35 [00:18<00:00,  1.86it/s, Loss=0.6147]
Validation: 100%|----| 4/4 [00:01<00:00,  3.31it/s]
Epoch   9/50 | Time: 20.1s | LR: 6.98e-05 | Train Loss: 0.5635, Acc: 0.8112, AUC: 0.8578 | Val Loss: 0.4772, Acc: 0.8800, Bal Acc: 0.8190, AUC: 0.9790
Epoch 10 Training: 100%|----| 35/35 [00:18<00:00,  1.84it/s, Loss=0.5534]
Validation: 100%|----| 4/4 [00:01<00:00,  3.35it/s]
Epoch  10/50 | Time: 20.2s | LR: 4.20e-05 | Train Loss: 0.5501, Acc: 0.8273, AUC: 0.8701 | Val Loss: 0.4954, Acc: 0.9000, Bal Acc: 0.8333, AUC: 0.9486
Epoch 11 Training: 100%|----| 35/35 [00:18<00:00,  1.94it/s, Loss=0.6340]
Validation: 100%|----| 4/4 [00:01<00:00,  2.87it/s]
Epoch  11/50 | Time: 19.4s | LR: 2.00e-05 | Train Loss: 0.5889, Acc: 0.8022, AUC: 0.8178 | Val Loss: 0.4957, Acc: 0.9000, Bal Acc: 0.8333, AUC: 0.9505
Epoch 12 Training: 100%|----| 35/35 [00:18<00:00,  1.87it/s, Loss=0.7624]
Validation: 100%|----| 4/4 [00:01<00:00,  3.38it/s]
Epoch  12/50 | Time: 20.0s | LR: 5.87e-06 | Train Loss: 0.5728, Acc: 0.8076, AUC: 0.8543 | Val Loss: 0.5001, Acc: 0.9000, Bal Acc: 0.8333, AUC: 0.9524
    ⏹ Early stopping triggered at epoch 12
/workspace/MultiMagBC/training/enhanced_train_k_fold.py:434: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load(os.path.join(self.config.MODELS_DIR, f"best_model_fold_{fold_idx}.pth")))
Validation: 100%|----| 1/1 [00:01<00:00,  1.45s/it]
Dataset created: 16 samples from 16 patients
  Label distribution: Benign=5, Malignant=11
  Samples per patient: min=1, max=1, avg=1.0
Dataset created: 16 samples from 16 patients
  Label distribution: Benign=5, Malignant=11
  Samples per patient: min=1, max=1, avg=1.0
Dataset created: 16 samples from 16 patients
  Label distribution: Benign=5, Malignant=11
  Samples per patient: min=1, max=1, avg=1.0
Dataset created: 16 samples from 16 patients
  Label distribution: Benign=5, Malignant=11
  Samples per patient: min=1, max=1, avg=1.0
Dataset created: 16 samples from 16 patients
  Label distribution: Benign=5, Malignant=11
  Samples per patient: min=1, max=1, avg=1.0
Dataset created: 16 samples from 16 patients
  Label distribution: Benign=5, Malignant=11
  Samples per patient: min=1, max=1, avg=1.0
Dataset created: 16 samples from 16 patients
  Label distribution: Benign=5, Malignant=11
  Samples per patient: min=1, max=1, avg=1.0
Dataset created: 16 samples from 16 patients
  Label distribution: Benign=5, Malignant=11
  Samples per patient: min=1, max=1, avg=1.0
Dataset created: 16 samples from 16 patients
  Label distribution: Benign=5, Malignant=11
  Samples per patient: min=1, max=1, avg=1.0
Dataset created: 16 samples from 16 patients
  Label distribution: Benign=5, Malignant=11
  Samples per patient: min=1, max=1, avg=1.0
Dataset created: 16 samples from 16 patients
  Label distribution: Benign=5, Malignant=11
  Samples per patient: min=1, max=1, avg=1.0
Dataset created: 16 samples from 16 patients
  Label distribution: Benign=5, Malignant=11
  Samples per patient: min=1, max=1, avg=1.0
Dataset created: 16 samples from 16 patients
  Label distribution: Benign=5, Malignant=11
  Samples per patient: min=1, max=1, avg=1.0

TTA Results - Accuracy: 0.6250, Balanced Accuracy: 0.6182, AUC: 0.8000

==================================================
Training Fold 5/5
==================================================
Dataset created: 559 samples from 56 patients
  Label distribution: Benign=170, Malignant=389
  Samples per patient: min=9, max=10, avg=10.0
Dataset created: 50 samples from 10 patients
  Label distribution: Benign=15, Malignant=35
  Samples per patient: min=5, max=5, avg=5.0
Dataset created: 16 samples from 16 patients
  Label distribution: Benign=4, Malignant=12
  Samples per patient: min=1, max=1, avg=1.0
Epoch 1 Training: 100%|----| 35/35 [00:18<00:00,  1.92it/s, Loss=0.6123]
Validation: 100%|----| 4/4 [00:01<00:00,  3.22it/s]
Epoch   1/50 | Time: 19.5s | LR: 1.33e-04 | Train Loss: 0.6827, Acc: 0.5796, AUC: 0.6144 | Val Loss: 0.6766, Acc: 0.3600, Bal Acc: 0.5429, AUC: 0.7600
    ★ New best validation balanced accuracy: 0.5429
Epoch 2 Training: 100%|----| 35/35 [00:19<00:00,  1.77it/s, Loss=0.5720]
Validation: 100%|----| 4/4 [00:01<00:00,  3.49it/s]
Epoch   2/50 | Time: 20.9s | LR: 2.00e-04 | Train Loss: 0.6231, Acc: 0.7138, AUC: 0.7663 | Val Loss: 0.6400, Acc: 0.7600, Bal Acc: 0.7143, AUC: 0.7448
    ★ New best validation balanced accuracy: 0.7143
Epoch 3 Training: 100%|----| 35/35 [00:18<00:00,  1.85it/s, Loss=0.6890]
Validation: 100%|----| 4/4 [00:01<00:00,  3.28it/s]
Epoch   3/50 | Time: 20.1s | LR: 2.00e-04 | Train Loss: 0.6166, Acc: 0.7478, AUC: 0.7824 | Val Loss: 0.5117, Acc: 0.8800, Bal Acc: 0.8190, AUC: 0.8800
    ★ New best validation balanced accuracy: 0.8190
Epoch 4 Training: 100%|----| 35/35 [00:18<00:00,  1.92it/s, Loss=0.6223]
Validation: 100%|----| 4/4 [00:01<00:00,  3.22it/s]
Epoch   4/50 | Time: 19.5s | LR: 1.95e-04 | Train Loss: 0.6319, Acc: 0.7102, AUC: 0.7658 | Val Loss: 0.5376, Acc: 0.9000, Bal Acc: 0.8333, AUC: 0.8248
    ★ New best validation balanced accuracy: 0.8333
Epoch 5 Training: 100%|----| 35/35 [00:18<00:00,  1.88it/s, Loss=0.4538]
Validation: 100%|----| 4/4 [00:01<00:00,  3.42it/s]
Epoch   5/50 | Time: 19.8s | LR: 1.81e-04 | Train Loss: 0.5780, Acc: 0.7478, AUC: 0.8398 | Val Loss: 0.5456, Acc: 0.9000, Bal Acc: 0.8333, AUC: 0.7867
Epoch 6 Training: 100%|----| 35/35 [00:18<00:00,  1.94it/s, Loss=0.4522]
Validation: 100%|----| 4/4 [00:01<00:00,  3.29it/s]
Epoch   6/50 | Time: 19.2s | LR: 1.59e-04 | Train Loss: 0.5812, Acc: 0.7889, AUC: 0.8289 | Val Loss: 0.4990, Acc: 0.9000, Bal Acc: 0.8333, AUC: 0.9486
Epoch 7 Training: 100%|----| 35/35 [00:18<00:00,  1.87it/s, Loss=0.4828]
Validation: 100%|----| 4/4 [00:01<00:00,  3.46it/s]
Epoch   7/50 | Time: 19.8s | LR: 1.31e-04 | Train Loss: 0.5954, Acc: 0.7657, AUC: 0.8067 | Val Loss: 0.5283, Acc: 0.9000, Bal Acc: 0.8333, AUC: 0.8514
Epoch 8 Training: 100%|----| 35/35 [00:18<00:00,  1.93it/s, Loss=0.4845]
Validation: 100%|----| 4/4 [00:01<00:00,  2.90it/s]
Epoch   8/50 | Time: 19.5s | LR: 1.01e-04 | Train Loss: 0.5898, Acc: 0.7746, AUC: 0.8230 | Val Loss: 0.5350, Acc: 0.9000, Bal Acc: 0.8333, AUC: 0.8000
Epoch 9 Training: 100%|----| 35/35 [00:18<00:00,  1.86it/s, Loss=0.4623]
Validation: 100%|----| 4/4 [00:01<00:00,  3.36it/s]
Epoch   9/50 | Time: 20.0s | LR: 6.98e-05 | Train Loss: 0.5741, Acc: 0.8014, AUC: 0.8310 | Val Loss: 0.5332, Acc: 0.9000, Bal Acc: 0.8333, AUC: 0.8076
Epoch 10 Training: 100%|----| 35/35 [00:19<00:00,  1.82it/s, Loss=0.6676]
Validation: 100%|----| 4/4 [00:01<00:00,  2.93it/s]
Epoch  10/50 | Time: 20.6s | LR: 4.20e-05 | Train Loss: 0.5713, Acc: 0.8032, AUC: 0.8311 | Val Loss: 0.5338, Acc: 0.9000, Bal Acc: 0.8333, AUC: 0.8419
Epoch 11 Training: 100%|----| 35/35 [00:18<00:00,  1.88it/s, Loss=0.4606]
Validation: 100%|----| 4/4 [00:01<00:00,  3.39it/s]
Epoch  11/50 | Time: 19.8s | LR: 2.00e-05 | Train Loss: 0.6085, Acc: 0.7889, AUC: 0.8080 | Val Loss: 0.5342, Acc: 0.9000, Bal Acc: 0.8333, AUC: 0.8590
Epoch 12 Training: 100%|----| 35/35 [00:18<00:00,  1.87it/s, Loss=0.5256]
Validation: 100%|----| 4/4 [00:01<00:00,  3.45it/s]
Epoch  12/50 | Time: 19.8s | LR: 5.87e-06 | Train Loss: 0.6082, Acc: 0.8050, AUC: 0.8267 | Val Loss: 0.5242, Acc: 0.9000, Bal Acc: 0.8333, AUC: 0.8724
Epoch 13 Training: 100%|----| 35/35 [00:17<00:00,  1.99it/s, Loss=0.5601]
Validation: 100%|----| 4/4 [00:01<00:00,  3.38it/s]
Epoch  13/50 | Time: 18.8s | LR: 2.00e-04 | Train Loss: 0.6063, Acc: 0.7674, AUC: 0.8163 | Val Loss: 0.5279, Acc: 0.9000, Bal Acc: 0.8333, AUC: 0.8743
Epoch 14 Training: 100%|----| 35/35 [00:17<00:00,  2.03it/s, Loss=0.4069]
Validation: 100%|----| 4/4 [00:01<00:00,  3.34it/s]
Epoch  14/50 | Time: 18.5s | LR: 1.99e-04 | Train Loss: 0.6040, Acc: 0.7621, AUC: 0.8244 | Val Loss: 0.5424, Acc: 0.9000, Bal Acc: 0.8333, AUC: 0.8705
    ⏹ Early stopping triggered at epoch 14
/workspace/MultiMagBC/training/enhanced_train_k_fold.py:434: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load(os.path.join(self.config.MODELS_DIR, f"best_model_fold_{fold_idx}.pth")))
Validation: 100%|----| 1/1 [00:01<00:00,  1.43s/it]
Dataset created: 16 samples from 16 patients
  Label distribution: Benign=4, Malignant=12
  Samples per patient: min=1, max=1, avg=1.0
Dataset created: 16 samples from 16 patients
  Label distribution: Benign=4, Malignant=12
  Samples per patient: min=1, max=1, avg=1.0
Dataset created: 16 samples from 16 patients
  Label distribution: Benign=4, Malignant=12
  Samples per patient: min=1, max=1, avg=1.0
Dataset created: 16 samples from 16 patients
  Label distribution: Benign=4, Malignant=12
  Samples per patient: min=1, max=1, avg=1.0
Dataset created: 16 samples from 16 patients
  Label distribution: Benign=4, Malignant=12
  Samples per patient: min=1, max=1, avg=1.0
Dataset created: 16 samples from 16 patients
  Label distribution: Benign=4, Malignant=12
  Samples per patient: min=1, max=1, avg=1.0
Dataset created: 16 samples from 16 patients
  Label distribution: Benign=4, Malignant=12
  Samples per patient: min=1, max=1, avg=1.0
Dataset created: 16 samples from 16 patients
  Label distribution: Benign=4, Malignant=12
  Samples per patient: min=1, max=1, avg=1.0
Dataset created: 16 samples from 16 patients
  Label distribution: Benign=4, Malignant=12
  Samples per patient: min=1, max=1, avg=1.0
Dataset created: 16 samples from 16 patients
  Label distribution: Benign=4, Malignant=12
  Samples per patient: min=1, max=1, avg=1.0
Dataset created: 16 samples from 16 patients
  Label distribution: Benign=4, Malignant=12
  Samples per patient: min=1, max=1, avg=1.0
Dataset created: 16 samples from 16 patients
  Label distribution: Benign=4, Malignant=12
  Samples per patient: min=1, max=1, avg=1.0
Dataset created: 16 samples from 16 patients
  Label distribution: Benign=4, Malignant=12
  Samples per patient: min=1, max=1, avg=1.0

TTA Results - Accuracy: 0.9375, Balanced Accuracy: 0.9583, AUC: 1.0000

============================================================
FINAL CROSS-VALIDATION RESULTS
============================================================
Standard Testing:
  Accuracy: 0.8529 ± 0.0857
  Balanced Accuracy: 0.8495 ± 0.0765
  AUC: 0.8739 ± 0.0947

Test-Time Augmentation:
  Accuracy: 0.8404 ± 0.1100
  Balanced Accuracy: 0.8295 ± 0.1127
  AUC: 0.8670 ± 0.0982

✅ Enhanced training completed!
📊 Results saved to: ./output/results/enhanced_final_results.json
📈 Current best accuracy: 0.9375. Target: 96%
💡 Consider further hyperparameter tuning or model architecture changes.

============================================================
🎉 TRAINING COMPLETED SUCCESSFULLY!
============================================================
📊 Results saved to: ./output/results
📈 Logs saved to: ./output/logs
💾 Models saved to: ./output/models

⏰ Completed at: 2025-07-29 12:15:04