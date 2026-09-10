# Low-compute stage2b fine-tuning (5 x RTX 4090, 24GB)

The shipped `configs/stage2b_arsft_vigeo.yaml` assumes >=80GB per GPU and a full-parameter
fine-tune. This is a sibling config that fine-tunes **the released v1.1 stage2b checkpoint**
with LoRA on five 24GB cards, on a small slice of a different corpus.

**What this is not:** it is not a reproduction of stage2b, and the run described below makes
no claim about output quality. It demonstrates that the training loop, memory budget,
checkpointing and merge path all work at this scale.

Inference, WBench evaluation and the original stage0-stage3 training paths are unchanged.
Every edit to a pre-existing file is additive and inert at its default value (section 5).

## 1. Why the shipped stage2b does not fit

The DiT is 13.12B parameters, 26GB in bf16.

| Cost | Full SFT, sharded over 5 ranks | Fits in 24GB? |
|---|---|---|
| Parameters | 5.2GB | yes |
| Gradients (bf16) | 5.2GB | |
| AdamW states (fp32 m+v) | 20.8GB | **no** |
| `next_forcing` head, replicated per rank | ~670M params, ~8GB with AdamW | **no** |
| Gemma-3-12B text encoder, per rank | 24GB | **no** |

Freezing the base removes the gradient and optimizer terms. The rest is handled by LoRA,
by disabling `next_forcing`, and by serving text embeddings from disk.

## 2. Setup

Environment, LTX-2.3 base weights, the released AR/DMD checkpoints and ViGeo are all as in
[`docs/WBENCH.md`](WBENCH.md) section 1 — the same conda env (`alayaworld`) serves both.
`pip install -r requirements-dev.txt` additionally installs pytest for the unit tests.

### 2.1 Dataset

[SpatialVID-HQ](https://huggingface.co/datasets/SpatialVID/SpatialVID-HQ) (CC-BY-NC-SA 4.0)
is real video carrying the three things the loader needs: per-frame camera poses,
intrinsics, and captions. It is gated behind an auto-approve form; accept it, then:

```bash
export HF_TOKEN=<your token>
python scripts/tools/prepare_spatialvid.py --num-clips 200
```

The full dataset is 3.53TB in 74 groups. The importer takes a slice of one group and
downloads only what it keeps: it selects eligible clips from the metadata CSV, then streams
the group's video tarball and stops after the first N eligible members. Members are in
**arbitrary** order, so a stream prefix is an unbiased sample of the eligible set and is
reached after a few percent of a 13.6GB file. Measured: 200 clips, **728MB** of video.

Filters exist to keep real dataset bias out of a small sample. Group_0001's annotated-length
distribution has a long short tail (p10 is 16 annotated frames, ~2.8s — far too short for a
training window), so `--min-frames 360` is the default; `--max-ocr` drops screen-text-heavy
clips, and `--min-dist-level` drops clips whose camera never moves.

Measured over the 200 clips actually used:

| | |
|---|---|
| frames per clip | 360 / 450 / 580 / 899 / 900 (min / p25 / median / p75 / max) |
| training window needs | 57 pixel frames — the shortest clip is a **6.3x** margin |
| median per-frame camera translation | 0.0045 (min 0.0007, max 0.0349); **zero clips are static** |
| caption length | 437 / 626 / 800 chars (min / median / max) |
| resolution | 1280x720, uniform |

### 2.2 Camera poses are interpolated — read this

SpatialVID annotates only every `int(fps/5)` frames, i.e. ~5Hz against 24-60fps video, while
the loader indexes poses per video frame. `alaya/data/spatialvid.py` slerps rotations and
linearly interpolates translations between consecutive **real** measurements 0.2s apart, and
clamps rather than extrapolates past the last annotation.

This is the only derived quantity in the pipeline. It is smoothing between real samples, not
fabrication, and it is milder than the shipped Sekai poses, which that dataset's own card
describes as estimates rather than source annotations. The alternatives were rejected
deliberately: re-encoding to 5fps would show the model 5fps motion labelled as 24fps, lying
about motion speed; lowering `sample.fps` to 5 would break the temporal conventions
(`norm_by_fps`, RoPE positions) the checkpoint was trained under.

Poses arrive as `(n,7)` `[tx,ty,tz,qx,qy,qz,qw]` **world-to-camera** rows in OpenCV
convention (`w2c = [R(quat) | t]`, quaternion in scipy's `(x,y,z,w)` order, per the dataset's
own `utils/quat_to_mat.py`); the importer inverts them to camera-to-world.

Intrinsics are written in **pixel** space on purpose: `_load_camera_params`
(`fastvideo/dataset/t2v_datasets.py:843`) treats `cx > 1.0` as "this file is in pixel units"
and derives each clip's own resolution from it. SpatialVID clip resolutions vary, so this is
more correct than a single `original_width` in the source config.

### 2.3 Text embeddings

Gemma-3-12B is 24GB in bf16 and cannot sit beside the sharded DiT, so training runs with
`ALAYA_SKIP_TEXT_ENCODER=1` and serves prompts from disk. A cache **miss raises** rather than
silently mis-encoding, so the cache must be complete before training starts:

```bash
CUDA_VISIBLE_DEVICES=0,1 ALAYA_GEMMA_MAX_MEMORY="0=13GiB,1=13GiB" \
  python scripts/tools/precache_train_text_embeds.py \
      --config configs/stage2b_arsft_lowcompute.yaml --device-map auto
```

Measured: **401 prompts in 1.9 min**, 3.2GB on disk (two per clip plus the CFG negative
prompt — `_clip_prompts` reaches both `overall.short_prompt` and `overall_caption`; caching a
superset of what training draws is harmless).

This is only sound because the config makes the prompt set finite: `use_segment_caption:
False` and `abstract_caption_prob: 0.0` give each clip exactly one caption. The tool
**asserts** both and refuses to run otherwise, so the invariant cannot silently rot.

> **Re-running the importer invalidates the sample cache.** `MultiSourceVideoDataset` keys its
> cached sample list on the source name, the jsonl *filename* and the pose subdir — not on the
> jsonl's contents. Re-running `prepare_spatialvid.py` with a different `--num-clips` rewrites
> the same path, so the stale pickle would be reused and training would run on the previous
> clip count with entirely healthy-looking logs. The importer now deletes that pickle after
> writing the jsonl. (This was caught in practice: a 200-clip import enumerated 9 prompts.)

## 3. Run

```bash
bash scripts/finetune/lowcompute_5x4090.sh
```

The launcher only sets environment; all training logic stays in `scripts/finetune/train.sh`.

| Variable | Why |
|---|---|
| `CUDA_VISIBLE_DEVICES=0,1,2,3,4` | GPU 5 stays free. An explicitly-set value is **validated**: the script refuses to start if it names device 5. |
| `ALAYA_INIT_TRANSFORMER_ON_CPU=1` | 26GB bf16 does not fit unsharded on a 24GB card; building on CPU lets FSDP shard it one attention block at a time |
| `ALAYA_SERIAL_MODEL_LOAD=1` | `LTX23Model` is constructed in fp32 (~52GB) before the bf16 cast; five concurrent builds would need ~260GB of host RAM |
| `ALAYA_SKIP_TEXT_ENCODER=1` | see 2.3 |
| `ALAYA_USE_FA3=0` | FA3 cannot coexist with the flash-attn-3 bundled inside the xformers ViGeo pulls in — they register the same torch operator namespace and abort in C++ |

`ALAYA_LOG_MEMORY=1` additionally prints a per-rank `[Mem]` line with the true peak (see 4).

## 4. Recipe deltas from stage2b, and measured cost

| Knob | stage2b | low-compute | Why |
|---|---|---|---|
| `training.mode` | `sft` | **`lora`** | section 1 |
| `paths.resume_checkpoint` | stage2a merged | **`weights/alaya-world-ar`** | start from the released stage2b |
| `lora` | disabled | **r=64, alpha=64**, 480 targeted Linears (**327M params**, 654MB bf16) | attn1/attn2/ff across 48 blocks |
| `next_forcing` | on | **off** | ~670M replicated params; also halves the target token count |
| `memory.train` | true | true (kept) | the HistoryEncoder is 34MB and is the load-bearing memory branch |
| `anti_drift` | on | on (kept) | the error bank is CPU-resident bf16, effectively free |
| `sample.height/width` | 544x960 | **416x736** | ~40% fewer tokens; both divisible by the VAE's 32x factor, aspect 1.769 vs 1.778 |
| `optimizer.grad_accum_steps` | n/a | **4** | 5 ranks x bs1 x 4 = effective batch 20 |
| `validation.enabled` | true | **false** | a rollout costs minutes; enable it deliberately |
| `runtime.vae_latent_cache_dir` | 544x960 dir | **null** | the cache is resolution-keyed and unusable at K=4 anyway |
| `runtime.attention_type` | `flash_attention_3` | unchanged | falls back to FA2 under `ALAYA_USE_FA3=0` |

Everything else — `memory`, `layout`, `control`, `anti_drift`, the rest of `spatial_memory` —
is byte-identical to stage2b.

### Measured on this hardware

| | |
|---|---|
| **Peak memory, per rank** | **17.9GB allocated** / 22.9GB reserved, of 24GB |
| Throughput | **11.9s per micro-batch**, so ~49s per optimizer step at `grad_accum_steps: 4` |
| | 300 optimizer steps ~ **4.1h** (the shipped `max_steps`); 600 would be ~8.2h. Note the `time=` field in `[Train]` lines is the LAST micro-batch only — `step_start` resets every dataloader iteration — so it under-reports the optimizer step by the accumulation factor. |
| Startup | ~6 min — ranks build the model one at a time under `ALAYA_SERIAL_MODEL_LOAD` |
| GPU 5 | 94MiB throughout, i.e. untouched |

**Read the memory number carefully.** `nvidia-smi` reports *reserved* memory, and PyTorch's
caching allocator grows to fill whatever is free — on this box it read 23.5GB/24 while the
live-tensor peak was 17.9GB. Halving `spatial_memory.vigeo_cache_budget` on the strength of
that reading changed the reserved figure by nothing, which is what prompted measuring
properly with `ALAYA_LOG_MEMORY=1`. There is ~6GB of genuine headroom, not 0.5GB.

If a longer schedule or a larger resolution does OOM, the knobs in order of preference are:
`spatial_memory.vigeo_cache_budget` down, `lora.rank` 64 -> 32, activation CPU offload,
resolution down to 352x608. 544x960 is unlikely to fit — it is ~1.7x the tokens.

## 5. What was changed in this repo, and why

| Change | Why | Inert by default? |
|---|---|---|
| `alaya/data/spatialvid.py` (new) | SpatialVID -> loader schema conversion; pure functions, unit-tested | new file |
| `scripts/tools/prepare_spatialvid.py` (new) | streaming dataset importer | new file |
| `scripts/tools/precache_train_text_embeds.py` (new) | fills the prompt cache; asserts the prompt set is finite | new file |
| `scripts/finetune/lowcompute_5x4090.sh` (new) | env-only launcher, execs `train.sh` | new file |
| `configs/stage2b_arsft_lowcompute.yaml` (new) | the config above | new file |
| `alaya/config/schema.py`: `optimizer.grad_accum_steps` | 5 ranks x bs1 is an effective batch of 5 | yes — defaults to 1 |
| `alaya/trainer/rollout_trainer.py`: gradient accumulation | as above | yes — `grad_accum_steps=1` is the pre-existing code path, same order of operations |
| `alaya/trainer/rollout_trainer.py`: `[Mem]` diagnostic | see section 4 | yes — off unless `ALAYA_LOG_MEMORY=1` |
| `fastvideo/dataset/t2v_datasets.py`: `spatialvid_hq` source | registers the corpus | yes — unreachable unless a config names it |
| `scripts/tools/merge_lora_for_rollout.py`: `.pt` bases | the released stage2b is `transformer.pt`; see section 6 | yes — `.safetensors` path unchanged |
| `alaya/model/loader.py`: `_text_encoder_disabled` message | it named only the WBench precache tool, misdirecting a training cache miss | message string only |

The gradient-accumulation all-reduce is gated to the last micro-step deliberately:
`_allreduce_grads` does `all_reduce(SUM)` then `div_(world_size)` and is **not** idempotent,
so running it mid-window would divide already-reduced gradients twice.

One caveat for `grad_accum_steps > 1`: a partially-filled accumulation window at the end of a
run is discarded. It cannot arise at the default of 1.

## 6. Using the result

Training writes `lora.safetensors` + `history_encoder.pt` per checkpoint. Merge the LoRA into
the released base to get a checkpoint the existing inference and WBench paths already
understand:

```bash
python scripts/tools/merge_lora_for_rollout.py \
    --ckpt_dir outputs/stage2b_lowcompute/checkpoint-300 \
    --base_transformer weights/alaya-world-ar/transformer.pt \
    --output outputs/stage2b_lowcompute/checkpoint-300-merged \
    --lora_rank 64 --lora_alpha 64
```

Because the base is a `.pt`, the output is `transformer.pt` and the merged directory has the
same shape as `weights/alaya-world-ar`. Point `paths.resume_checkpoint` at it and
`configs/infer_i2v_camera_ar.yaml` or `configs/wbench_full.yaml` run unchanged — **that is the
mechanism by which this work leaves inference and WBench alone**, rather than a promise.

An unmatched LoRA key is now fatal rather than a warning: a silently skipped adapter produces
a checkpoint that loads cleanly and generates subtly wrong video.

### Three things to change before rendering on 24GB cards

`configs/infer_i2v_camera_ar.yaml` is written for 80GB hardware, so a copy of it needs the same
two adaptations `configs/wbench_full.yaml` already carries. Copy it rather than editing it in
place -- the shipped config is left alone on purpose.

1. **`dmd.enabled: false`.** Left at `true`, every rank builds a *second* frozen 13B score
   model that validation never touches. On this box that is an immediate host-RAM OOM: the
   rank is SIGKILLed and torchrun reports the unhelpful `exitcode: -9`. Confirmed by
   `[DMD] real/fake score base` appearing once per rank in the log.
2. **`spatial_memory.vigeo_cache_budget: 65536`** (the AR config ships 262144, sized for 80GB
   cards). Left alone, a 5-round rollout at 544x960 dies with a CUDA OOM partway through --
   measured here at round 3 of 5, with 22.27GB already allocated. `validation.save_debug_videos:
   false` is worth setting too: the diagnostic strip decodes far more pixels than the output
   video itself.
3. **The text encoder must be off.** Gemma-3-12B is 24GB and cannot sit beside the sharded
   DiT at inference either, so precache the prompts and run with `ALAYA_SKIP_TEXT_ENCODER=1`,
   exactly as [`docs/WBENCH.md`](WBENCH.md) does for generation.

Also note that with `ALAYA_INIT_TRANSFORMER_ON_CPU=1` all five ranks hold a 26GB CPU-resident
model at once. Merging writes 26GB straight into the page cache, so running a merge and a
rollout back to back can tip a 251GB host over; `sync` and let the cache drain first.

### Disk

A checkpoint is ~690MB (`lora.safetensors` 654MB + `history_encoder.pt` 34MB); no
`transformer.pt` is written in `lora` mode. The **merge** in section 6 writes a full 26GB
`transformer.pt`, so free that much before running it -- prune intermediate checkpoints if
needed.

## 7. Limitations

- The acceptance run is 300 optimizer steps on 200 clips. That is enough to show the machinery
  works end to end and far too little to claim any quality improvement. No such claim is made.
- 416x736 is not the checkpoint's native 544x960, nor the resolution WBench evaluates at, so a
  quality comparison against the released model would not be apples-to-apples.
- `next_forcing` is off, so this is not the full stage2b recipe.
- Camera poses are interpolated from ~5Hz annotations (section 2.2).
- The corpus is one slice of one SpatialVID group, uniform 1280x720. Other groups will differ.
