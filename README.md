# GeomCLIP: Contrastive Geometry-Text Pre-training for Molecules

## Requirements

Run this command to create the environment of `GeomCLIP`: 

```bash
conda env create -f environment.yml
```

## Dataset
https://drive.google.com/file/d/1j5J2ZkqUNS_TKqDbSbtrEi9CD9__ON6t/view

## Run
```bash
python main.py  --devices '[0]' --mode train --filename stage1  --tune_gnn --batch_size 64 --max_epochs 30 --store_path ./all_checkpoints
```

