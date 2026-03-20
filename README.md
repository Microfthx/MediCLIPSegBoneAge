# MediCLIPSegBoneAge

![Overview](assets/overview.svg)

MediCLIPSegBoneAge is a starter project for adapting MedCLIPSeg style vision language modeling to pediatric bone age assessment. The current repository contains a first working baseline for RSNA bone age regression, built as a standalone research codebase with a MedCLIP inspired text conditioned image encoder and a simple fusion head for image features, prompt features, and gender metadata.

![Pipeline](assets/pipeline.svg)

## Project layout

```text
MediCLIPSegBoneAge/
├── configs/
├── mediclipsegboneage/
├── scripts/
├── train.py
└── README.md
```

## Current scope

This initial version trains a text conditioned CLIP based bone age regressor on RSNA style CSV annotations. The model uses:

- image encoder from the local `MedCLIPSeg` project
- prompt encoder for task text
- learnable fusion of image features, text features, and gender metadata
- MAE regression objective

## First baseline

The first completed baseline was trained on the RSNA split for 10 epochs. The best validation result was obtained at epoch 10 with `val_mae = 30.7205`.

| Epoch | Train MAE | Val MAE | Val Loss | LR |
| --- | ---: | ---: | ---: | ---: |
| 1 | 47.5851 | 32.3768 | 200.0837 | 1e-4 |
| 4 | 32.8038 | 32.2359 | 200.4504 | 1e-4 |
| 5 | 32.5282 | 31.2861 | 189.1032 | 1e-4 |
| 6 | 31.4517 | 30.8666 | 180.1267 | 1e-4 |
| 7 | 31.5487 | 30.7430 | 178.5027 | 1e-4 |
| 10 | 31.0308 | 30.7205 | 178.4001 | 1e-4 |

The full training history used for this baseline is stored in [results/rsna_clip_baseline_history.csv](/data/hxly/projects/MediCLIPSegBoneAge/results/rsna_clip_baseline_history.csv). The local training artifacts are:

- best checkpoint: `outputs/rsna_clip_baseline/best.pt`
- last checkpoint: `outputs/rsna_clip_baseline/last.pt`
- raw history file: `outputs/rsna_clip_baseline/history.csv`

Training log summary:

```text
Epoch 005 | train loss=197.7099 mae=32.5282 | val loss=189.1032 mae=31.2861 | lr=0.0001000
Saved new best checkpoint with val_mae=31.2861
Epoch 006 | train loss=189.0268 mae=31.4517 | val loss=180.1267 mae=30.8666 | lr=0.0001000
Saved new best checkpoint with val_mae=30.8666
Epoch 007 | train loss=189.2668 mae=31.5487 | val loss=178.5027 mae=30.7430 | lr=0.0001000
Saved new best checkpoint with val_mae=30.7430
Epoch 008 | train loss=181.8035 mae=30.7032 | val loss=179.0077 mae=30.9678 | lr=0.0001000
Epoch 009 | train loss=183.1067 mae=30.8954 | val loss=191.3968 mae=31.9023 | lr=0.0001000
Epoch 010 | train loss=185.1844 mae=31.0308 | val loss=178.4001 mae=30.7205 | lr=0.0001000
Saved new best checkpoint with val_mae=30.7205
```

## Expected data format

The dataset loader expects:

- images in one folder, named as `<ID>.png`
- CSV files with columns `ID`, `Male`, and `Boneage` for labeled splits

## Quick start

```bash
python train.py \
  --image-dir /data/hxly/datasets/BoneAge/RSNA/rsna_all \
  --train-csv /data/hxly/datasets/BoneAge/RSNA/annotations/RSNA_Annotations/RSNA_Boneage_Training.csv \
  --val-csv /data/hxly/datasets/BoneAge/RSNA/annotations/RSNA_Annotations/RSNA_Boneage_Validation.csv \
  --output-dir outputs/rsna_clip_baseline
```

## Next steps

The current code is a clean starting point. The next research upgrades worth doing are:

- replace global CLIP pooling with patch token pooling from MedCLIPSeg
- add a hand region branch or external hand mask
- compare generic prompts and age aware prompt templates
- test uncertainty aware regression heads
