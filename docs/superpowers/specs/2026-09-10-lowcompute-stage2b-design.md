# Low-compute stage2b fine-tuning on 5 x RTX 4090

**Status:** design approved, not yet implemented
**Date:** 2026-09-10

Fine-tune the released AlayaWorld v1.1 stage2b checkpoint on new data using five
24GB RTX 4090s (GPUs 0-4; GPU 5 is never used). The shipped `stage2b_arsft_vigeo.yaml`
assumes >=80GB per card and full-parameter SFT with FSDP; this variant reaches the
same trainer through a different config and a handful of additive code changes.

Inference, WBench evaluation and the original stage0-stage3 training paths must be
unaffected. Every code change below is additive and inert at its default value.

---

## 1. Why the shipped stage2b does not fit

The DiT is ~13B parameters, 26GB in bf16.

| Cost | Full SFT | Fits in 24GB? |
|---|---|---|
| Parameters, FSDP-sharded over 5 ranks | 5.2GB | yes |
| Gradients (bf16, sharded) | 5.2GB | |
| AdamW states (fp32 m+v, sharded) | 20.8GB | **no** |
| `next_forcing` head, replicated per rank | ~670M params -> ~8GB with AdamW | **no** |
| Gemma-3-12B text encoder, per rank | 24GB | **no** |

Freezing the base removes the gradient and optimizer terms entirely. The remaining
three problems are solved by LoRA, by disabling `next_forcing`, and by serving text
embeddings from disk.

## 2. Recipe

`configs/stage2b_arsft_lowcompute.yaml` is a sibling of `stage2b_arsft_vigeo.yaml`.
Deltas, and only these:

| Key | stage2b | low-compute | Reason |
|---|---|---|---|
| `training.mode` | `sft` | `lora` | see section 1 |
| `paths.resume_checkpoint` | stage2a merged dir | `weights/alaya-world-ar` | start from the released stage2b |
| `paths.history_encoder` | stage2a | `weights/alaya-world-ar/history_encoder.pt` | ditto |
| `lora.enabled` / `.train` | false | **true** | |
| `lora.rank` / `.alpha` | 128 | **64 / 64** | ~250M params, ~3.0GB incl. AdamW |
| `next_forcing.enabled` | true | **false** | 670M replicated head; also halves the target token count |
| `memory.train` | true | true (kept) | HistoryEncoder is 34MB and is the load-bearing memory branch |
| `anti_drift.*` | on | on (kept) | the error bank is CPU-resident bf16, effectively free |
| `sample.height` / `.width` | 544 / 960 | **416 / 736** | ~40% fewer tokens; both divisible by the VAE's 32x spatial factor, aspect 1.769 vs 1.778 |
| `data.sources` | `sekai_real_hq` | `spatialvid_hq` | section 4 |
| `spatial_memory.vigeo_cache_budget` | 262144 | **65536** | the value the WBench work proved on these cards |
| `runtime.vae_latent_cache_dir` | 544x960 dir | **null** | the cache is resolution-keyed and unusable at K=4 anyway (see `docs/vigeo/README.md` section 7) |
| `runtime.text_embed_cache_dir` | shared | `cache/text_embed_spatialvid` | filled by a separate pass, section 5 |
| `validation.enabled` | true | **false** by default | one rollout costs minutes; enabled explicitly for the acceptance run |
| `optimizer.grad_accum_steps` | n/a | **4** | 5 ranks x bs1 x 4 = effective batch 20 |
| `runtime.fsdp` | true | true | FSDP1, unchanged |
| `runtime.gradient_checkpointing` | true | true | unchanged |
| `runtime.attention_type` | `flash_attention_3` | unchanged | `ALAYA_USE_FA3=0` already makes this fall back to FA2 (`ltx2/modules/attention.py:301`), which is what the ViGeo path requires |

A documented `# override for 544x960` block in the config comments records how to
restore native resolution on a machine with more headroom.

### Per-rank VRAM budget

| Item | GB |
|---|---|
| DiT parameters, FSDP-sharded | 5.2 |
| FSDP all-gather working set (~2 blocks + prefetch) | 1.1 |
| VAE encoder + decoder | 1.5 |
| ViGeo model + KV cache at budget 65536 | ~2.5 |
| LoRA r64: params + grads + AdamW fp32 | 3.0 |
| Activations under gradient checkpointing at 416x736 | ~2 |
| **Total** | **~15.3 of 24** |

This is an estimate, not a measurement. Section 7 measures it before anything is claimed.

**Escalation ladder** if the measured peak exceeds ~22GB, applied in this order and
each documented in `docs/LOWCOMPUTE.md`:
1. `spatial_memory.vigeo_cache_budget` 65536 -> 32768
2. `lora.rank` 64 -> 32
3. activation CPU offload (`torch.autograd.graph.save_on_cpu`) behind a new env flag
4. resolution 416x736 -> 352x608
5. FSDP2 (`fully_shard`) to shard the LoRA parameters and AdamW state

Only step 5 is a new code path; steps 1, 2 and 4 are config, step 3 is ~15 lines.

## 3. Code changes

All four are additive. None alters behaviour at its default.

### 3.1 `alaya/config/schema.py`

```python
class OptimizerConfig:
    grad_accum_steps: int = 1   # micro-batches per optimizer step; 1 = today's behaviour
```

### 3.2 `alaya/trainer/rollout_trainer.py` — gradient accumulation

`train_one_step` currently does `zero_grad -> forward -> backward -> sync -> clip ->
step` on every batch. It gains three keyword-only arguments, all defaulting to
today's behaviour:

```python
def train_one_step(self, batch, *, accum_first=True, accum_last=True, loss_scale=1.0):
```

- `zero_grad(set_to_none=True)` runs only when `accum_first`
- the loss is multiplied by `loss_scale` before `backward()`
- `_sync_grads_outside_fsdp()`, `clip_grad_norm_`, `optimizer.step()` and
  `scheduler.step()` run only when `accum_last`

Gating the all-reduce on the last micro-step is not merely an optimisation, it is
required for correctness: `_allreduce_grads` does `all_reduce(SUM)` followed by
`div_(world_size)`, so calling it on every micro-step would divide already-reduced
gradients again. The frozen base contributes no FSDP-internal gradient traffic, so
no `no_sync()` context is needed.

`train()` drives the counter and increments `self.global_step` only on an optimizer
step, so checkpoint, validation and `max_steps` triggers keep counting optimizer
steps rather than micro-batches.

`_train_one_step_inline` (`layout.condition.type == "inline"`) is a separate path
that stage2b does not use. It raises a clear error if `grad_accum_steps > 1`, rather
than silently ignoring it.

With `grad_accum_steps=1` the code path is identical to today's, including the order
of operations.

### 3.3 `fastvideo/dataset/t2v_datasets.py` — one new source

A single new entry in `MultiSourceVideoDataset.SOURCE_CONFIGS`:

```python
'spatialvid_hq': {
    'has_camera': True,
    'annotation_subdir': 'spatialvid_hq',
    'jsonl': 'spatialvid_hq.jsonl',
    'video_subdir': '',
    'caption_subdir': 'caption',
    'pose_subdir': 'pose',
    'original_width': 1920.0,
    'original_height': 1080.0,
    'use_segment_caption': False,
},
```

`use_segment_caption: False` routes captions through `_load_caption`, which already
reads `overall_caption` / `caption` / `text` / `overall.description`. No loader logic
changes; the prepare script emits a file those branches already understand.

Unreachable unless a config names `spatialvid_hq` in `data.sources`.

### 3.4 `scripts/tools/merge_lora_for_rollout.py` — `.pt` bases

The tool currently requires a `.safetensors` base and writes
`diffusion_pytorch_model.safetensors`. The released stage2b is `transformer.pt`.
A `--base-format {safetensors,pt}` flag (auto-detected from the suffix) lets it read
`transformer.pt` and write `transformer.pt`, so a merged low-compute checkpoint
directory is consumed unchanged by `configs/infer_i2v_camera_ar.yaml` and
`configs/wbench_full.yaml` via `paths.resume_checkpoint`. **This is what keeps the
inference and WBench code untouched** — the fine-tune is delivered as a checkpoint in
the shape they already load, not as a new loading path.

### 3.5 New scripts

- `scripts/finetune/lowcompute_5x4090.sh` — sets `CUDA_VISIBLE_DEVICES=0,1,2,3,4`,
  `ALAYA_INIT_TRANSFORMER_ON_CPU=1`, `ALAYA_SERIAL_MODEL_LOAD=1`,
  `ALAYA_SKIP_TEXT_ENCODER=1`, `ALAYA_USE_FA3=0`, then execs the existing
  `scripts/finetune/train.sh`. No logic of its own.
- `scripts/tools/prepare_spatialvid.py` — section 4.
- `scripts/tools/precache_train_text_embeds.py` — section 5.

## 4. Dataset: SpatialVID-HQ

[SpatialVID-HQ](https://huggingface.co/datasets/SpatialVID/SpatialVID-HQ) (CC-BY-NC-SA
4.0, gated with an auto-approve form) is real video carrying exactly the three things
the loader needs: per-frame camera poses, intrinsics, and captions. It is 3.53TB in
74 groups; this uses a ~200-clip slice of one group.

### Source format

| File | Content |
|---|---|
| `videos/group_XXXX/<id>.mp4` | the clip, typically 1920x1080 |
| `annotations/group_XXXX/<id>/poses.npy` | `(n,7)` `[tx,ty,tz,qx,qy,qz,qw]`, **world-to-camera**, OpenCV convention |
| `annotations/group_XXXX/<id>/intrinsics.npy` | `(n,4)` normalized `[fx,fy,cx,cy]` |
| `annotations/group_XXXX/<id>/caption.json` | `SceneSummary`, `SceneDescription`, `CameraMotion`, `ShotImmersion`, `CategoryTags` |
| `annotations/group_XXXX/<id>/indexes.txt` | which video frames the `n` annotations correspond to |
| `data/train/SpatialVID_HQ_metadata.csv` | 141MB; per-clip fps, frame count, resolution, scores, scene tags |

**Annotations exist only every `int(fps/5)` frames** — roughly 5Hz against a 24-30fps video.

### `scripts/tools/prepare_spatialvid.py`

1. Fetch the metadata CSV; select `--num-clips` (default 200) from `--group`
   (default `group_0001`), filtered on frame count (long enough for a training
   window), low `ocr score`, and a non-degenerate `distLevel` so the camera actually moves.
2. **Stream-extract** only the selected ids from `annotations/group_XXXX.tar.gz` and
   `videos/group_XXXX.tar.gz` over HTTP, stopping once all are found. tar.gz is
   sequential, so this downloads a prefix (~1.5GB) rather than the full 15GB. The
   machine has ~48GB free, which this respects.
3. Convert each clip into the schema `docs/vigeo/README.md` section 4 documents:
   - **poses**: `[t|quat]` -> 4x4 world-to-camera -> invert to **camera-to-world** ->
     interpolate onto the full video frame grid -> `cam_c2w [N,4,4]` float32
   - **intrinsics**: normalized `[fx,fy,cx,cy]` -> normalized 3x3 `K`, written into
     the same npz under the `intrinsics` key that `_load_camera_params` reads
   - **caption**: `{"overall_caption": <SceneSummary + " " + SceneDescription>,
     "overall": {"short_prompt": SceneSummary, "full_prompt": SceneDescription}}` —
     verbatim dataset text, nothing invented
   - one jsonl line per clip with `video`, `prompt`, `pose`, `num_frames`
4. Leave the mp4 untouched under `data/Video/spatialvid_hq/`. Output layout:

```
data/Video/spatialvid_hq/<id>.mp4
data/Annotation/spatialvid_hq/spatialvid_hq.jsonl
data/Annotation/spatialvid_hq/caption/<id>.json
data/Annotation/spatialvid_hq/pose/<id>.npz
```

### Pose interpolation is the one derived quantity

Poses are measured at ~5Hz; the loader indexes them per original video frame. The
script slerps rotations and lerps translations between consecutive real measurements
0.2s apart. This is smoothing between real samples, not fabrication, and it is milder
than the shipped Sekai poses, which the dataset card states are estimates rather than
source annotations.

The alternatives are worse and are rejected explicitly: re-encoding to 5fps would
show the model 5fps motion labelled as 24fps, lying about motion speed; setting
`sample.fps: 5` would break the temporal conventions (`norm_by_fps`, RoPE positions)
the checkpoint was trained under.

`docs/LOWCOMPUTE.md` states this plainly so no reader mistakes interpolated poses for
per-frame ground truth.

## 5. Text embeddings

Gemma-3-12B is 24GB in bf16 and cannot sit beside the DiT, so training runs with
`ALAYA_SKIP_TEXT_ENCODER=1` and serves prompts from
`runtime.text_embed_cache_dir`. A cache miss raises rather than silently
mis-encoding, so the cache must be complete before training starts.

This is sound only because the low-compute config makes the prompt set finite and
enumerable: `use_segment_caption: False` and `data.abstract_caption_prob: 0.0` give
each clip exactly one caption string, and `_caption_with_prefix` applies a fixed
per-source prefix. The set is `num_clips + 1` strings (the extra being the CFG
negative prompt).

`scripts/tools/precache_train_text_embeds.py` follows
`precache_wbench_text_embeds.py`: build only the text encoder and the training
dataset, enumerate every caption, encode with Gemma spread over two GPUs
(`ALAYA_GEMMA_MAX_MEMORY="0=13GiB,1=13GiB"`), write to the cache. Incremental on
re-run. It **asserts** the config yields a finite prompt set and refuses to run
otherwise, so the invariant cannot silently rot.

## 6. Delivering the result to inference

Training writes `lora.safetensors` + `history_encoder.pt` per checkpoint (already
`save_checkpoint`'s behaviour for `training.mode: lora`). To use the result:

```bash
python scripts/tools/merge_lora_for_rollout.py \
    --ckpt_dir outputs/stage2b_lowcompute/checkpoint-NNN \
    --base_transformer weights/alaya-world-ar/transformer.pt \
    --output outputs/stage2b_lowcompute/checkpoint-NNN-merged
```

The merged directory has the same shape as `weights/alaya-world-ar`, so pointing
`paths.resume_checkpoint` at it runs the existing inference and WBench paths with no
code change.

## 7. Acceptance

In order; nothing is claimed working before its evidence exists.

1. `DESCRIBE=1` resolves the new config without touching a GPU.
2. `prepare_spatialvid.py` produces ~200 real clips; spot-check that decoded frame
   counts match the jsonl, that `cam_c2w` is per-frame with plausible translation
   magnitudes, and that captions are non-empty dataset text.
3. Precache the text embeddings for that clip set.
4. **Smoke run**, `optimizer.max_steps: 30`: record the loss trace and
   `torch.cuda.max_memory_allocated()` per rank. If the peak exceeds ~22GB, walk the
   escalation ladder in section 2 and record what it took.
5. **Acceptance run**, ~300-600 optimizer steps: checkpoint, merge the LoRA, render
   one validation rollout from the merged checkpoint. This demonstrates the machinery
   end to end. It does **not** claim the fine-tune improved quality — 300-600 steps on
   200 clips is far too little for that, and the write-up will say so.
6. **Non-interference**: `DESCRIBE=1` clean on all five original configs
   (`stage0_precache`, `stage1_pretrain_bidir`, `stage2a_histpretrain`,
   `stage2b_arsft_vigeo`, `stage3_dmd_vigeo`) plus `wbench_full`; and a diff review
   confirming every edit to an existing file is additive and default-off.

## 8. Out of scope

- FSDP2, unless step 4 of the acceptance criteria forces it (ladder step 5)
- QLoRA / any weight quantisation — explicitly excluded by the request
- Multi-node
- Stage3 DMD distillation on low compute
- Any change to `inference/`, `reactor/`, `alaya/inference/`, `run_wbench.py`, or the
  original stage configs
