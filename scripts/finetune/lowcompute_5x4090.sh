#!/usr/bin/env bash
# ============================================================================
# Low-compute launcher: stage2b fine-tuning on five 24GB cards.
#
#   bash scripts/finetune/lowcompute_5x4090.sh
#
# Sets only environment; all training logic stays in scripts/finetune/train.sh.
# Every variable below is already honoured by the existing code -- see
# docs/LOWCOMPUTE.md for why each one is needed.
# ============================================================================
set -euo pipefail

# GPU 5 is deliberately excluded and must stay free.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4}

# The 13B DiT is 26GB in bf16 and does not fit unsharded on a 24GB card. Building
# it on CPU lets FSDP move and shard it one attention block at a time.
export ALAYA_INIT_TRANSFORMER_ON_CPU=1

# LTX23Model is constructed in fp32 (~52GB) before the bf16 cast, so five
# concurrent builds would need ~260GB of host RAM. Load one rank at a time.
export ALAYA_SERIAL_MODEL_LOAD=1

# Gemma-3-12B is 24GB in bf16; prompts come from the on-disk cache instead.
# Fill it first with scripts/tools/precache_train_text_embeds.py.
export ALAYA_SKIP_TEXT_ENCODER=1

# FA3 cannot coexist with the flash-attn-3 bundled inside the xformers that ViGeo
# pulls in -- they register the same torch operator namespace and abort in C++.
export ALAYA_USE_FA3=0

export CONFIG_PATH=${CONFIG_PATH:-configs/stage2b_arsft_lowcompute.yaml}
exec bash "$(dirname "$0")/train.sh"
