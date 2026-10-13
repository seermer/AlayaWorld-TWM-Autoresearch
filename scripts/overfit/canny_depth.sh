#!/usr/bin/env bash
# Overfit test on the standard training entrance (docs/TRAINING.md), end to end on GPUs 0-3:
# data, LoRA training on 100 clips that turn into a canny edge map or a depth map when the text
# says so, then the trained model on 3 scenes it was not trained on. Run from the repo root in the
# alayaworld env, with HF_TOKEN set (DL3DV is gated).
# Videos: outputs/overfit_canny_depth/videos/case_<id>_combined.mp4
set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID
TRAIN=configs/overfit/canny_depth_train.yaml
RENDER=configs/overfit/canny_depth_render.yaml
OUT=outputs/overfit_canny_depth

python scripts/overfit/fetch_dl3dv.py
for r in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$r python scripts/overfit/poses_depth.py --rank $r --world 4 & done; wait
python scripts/overfit/build_dataset.py

python scripts/tools/check_dataset.py --config $TRAIN
CUDA_VISIBLE_DEVICES=0,1 ALAYA_GEMMA_MAX_MEMORY="0=13GiB,1=13GiB" python scripts/tools/precache_train_text_embeds.py --config $TRAIN --device-map auto
CUDA_VISIBLE_DEVICES=0,1,2,3 CONFIG_PATH=$TRAIN bash scripts/finetune/lowcompute_4x4090.sh

python scripts/tools/concat_loras.py --loras weights/alaya-world-dmd/lora.safetensors \
  $OUT/train/checkpoint-400/lora.safetensors --output $OUT/lora
CUDA_VISIBLE_DEVICES=0,1 ALAYA_GEMMA_MAX_MEMORY="0=13GiB,1=13GiB" python -m scripts.tools.precache_wbench_text_embeds --config $RENDER --device-map auto
python scripts/tools/run_wbench.py --config $RENDER --gpus 0,1,2,3 --cases 9101,9102,9103,9111,9112,9113,9121,9122,9123
ls $OUT/videos/*.mp4
