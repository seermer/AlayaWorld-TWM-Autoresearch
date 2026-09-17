#!/usr/bin/env bash
# Overfit test on the standard training entrance (docs/TRAINING.md), end to end on GPUs 0-3:
# base-model generation, LoRA training on 29 pencil-sketch clips, trained-model generation.
# Needs data/overfit/c3vd_pencil (29 clips) and data/overfit/c3vd_pencil_gen (4 of them).
# Videos: outputs/overfit_c3vd_pencil/gen_{base,trained}/validation/step-000000/c3vd_rollout/*_pred_clean.mp4
set -euo pipefail

python scripts/tools/check_dataset.py --config configs/overfit/c3vd_pencil_train.yaml
CUDA_VISIBLE_DEVICES=0,1 ALAYA_GEMMA_MAX_MEMORY="0=13GiB,1=13GiB" python scripts/tools/precache_train_text_embeds.py --config configs/overfit/c3vd_pencil_train.yaml --device-map auto
CUDA_VISIBLE_DEVICES=0,1 ALAYA_GEMMA_MAX_MEMORY="0=13GiB,1=13GiB" python scripts/tools/precache_train_text_embeds.py --config configs/overfit/c3vd_pencil_gen_base.yaml --device-map auto
rm -rf outputs/overfit_c3vd_pencil
CUDA_VISIBLE_DEVICES=0,1,2,3 VALIDATE_ONLY=1 CONFIG_PATH=configs/overfit/c3vd_pencil_gen_base.yaml bash scripts/finetune/lowcompute_4x4090.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 CONFIG_PATH=configs/overfit/c3vd_pencil_train.yaml bash scripts/finetune/lowcompute_4x4090.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 VALIDATE_ONLY=1 CONFIG_PATH=configs/overfit/c3vd_pencil_gen_trained.yaml bash scripts/finetune/lowcompute_4x4090.sh
ls outputs/overfit_c3vd_pencil/gen_base/validation/step-000000/c3vd_rollout/*_pred_clean.mp4 outputs/overfit_c3vd_pencil/gen_trained/validation/step-000000/c3vd_rollout/*_pred_clean.mp4
