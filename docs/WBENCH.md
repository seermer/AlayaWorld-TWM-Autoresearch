# Evaluating AlayaWorld on WBench (full 289-case split)

AlayaWorld is **text-driven**: every turn is served by a per-turn prompt, so it is
evaluated on WBench's **full split (289 cases / 1058 turns)**, not the 158-case
navigation subset that camera- and action-conditioned models are limited to.

Everything below assumes the two repos side by side:

```
WM-AutoResearch/
├── WBench/        benchmark: data/, weights/, work_dirs/<model>/videos/
└── WorldModel/    this repo
```

## 1. One-time setup

> **In the AutoResearcher project** the envs live at `AutoResearcher/.envs/<name>` (`alayaworld`, `wbench-main`, `wbench-vp`); use `conda activate <path-to>/AutoResearcher/.envs/<name>` or run `<path>/.envs/<name>/bin/python`. See `AutoResearcher/docs/PORTABILITY.md`.

**Environment** — conda env `alayaworld` (python 3.10, torch 2.7.1+cu128):

```bash
conda create -y -n alayaworld python=3.10 && conda activate alayaworld
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt easydict av matplotlib
pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/\
flash_attn-2.8.3.post1%2Bcu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
```

Pin `transformers<5`: v5 breaks the `Gemma3ForConditionalGeneration` call in
`alaya/model/loader.py`.

**Weights** (paths are the ones `configs/wbench_full.yaml` expects):

```bash
hf download AlayaLab/AlayaWorld-v1.1-stage2b --local-dir weights/alaya-world-ar
hf download AlayaLab/AlayaWorld-v1.1-stage3  --local-dir weights/alaya-world-dmd
hf download Lightricks/LTX-2.3 --include "ltx-2.3-22b-dev.safetensors" --local-dir weights/ltx-2.3
hf download google/gemma-3-12b-it-qat-q4_0-unquantized \
    --local-dir weights/ltx-2.3/google/gemma-3-12b-it-qat-q4_0-unquantized   # gated
git clone https://github.com/aigc3d/ViGeo third_party/ViGeo                  # code
hf download pkqbajng/ViGeo1.1 --local-dir third_party/ViGeo/checkpoints/ViGeo1.1
```

**Text-embedding cache** — Gemma-3-12B is 24GB in bf16 and cannot sit next to the
DiT on a 24GB card. The WBench prompt set is finite (747 strings: one base caption
plus one accumulated schedule entry per turn, plus the CFG negative prompt), so
encode it once up front and run generation with the encoder switched off:

```bash
CUDA_VISIBLE_DEVICES=0,1 ALAYA_GEMMA_MAX_MEMORY="0=13GiB,1=13GiB" \
  python scripts/tools/precache_wbench_text_embeds.py \
      --config configs/wbench_full.yaml --device-map auto      # ~4 min
```

Re-run it after any change that alters prompt text; it is incremental, and a cache
miss during generation raises rather than silently mis-encoding.

## 2. Generate

```bash
python scripts/tools/run_wbench.py --gpus 0,1,2,3            # all 289 cases
python scripts/tools/run_wbench.py --cases 1,7,23              # a subset
python scripts/tools/run_wbench.py --resume                    # skip finished cases
```

Output lands directly in WBench's layout: `../WBench/work_dirs/alayaworld/videos/
case_<id>_combined.mp4`, each with a sidecar `.json` recording the per-turn frame
segments, actions and prompt schedule.

The driver groups cases by turn count and emits **one validation mode per group**
in a single launch. FSDP requires every rank to issue the same forwards, so a mode
rolls every case out to `max(turns) * wbench_chunks_per_turn` rounds; grouping keeps
that equal to what each case actually needs (693 collective rounds instead of 1734)
while still loading the model only once.

Each turn is 3 chunks x 4 latents x 8 = 96 frames = 4.0s at 24fps.

## 3. Evaluate

```bash
cd ../WBench && conda activate wbench-main
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export VLM_API_KEY=<volcengine-ark-key>          # 6 of the 22 metrics

python main.py --model alayaworld --phase precompute --gpus 0,1,2,3  # SAM2 + DA3 + MegaSAM
python main.py --model alayaworld --phase gpu        --gpus 0,1,2,3
python main.py --model alayaworld --phase vlm
python main.py --model alayaworld --phase report

conda activate wbench-vp                          # visual_plausibility only
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/run_visual_plausibility.py --model alayaworld
```

WBench's `--gpus` defaults to every visible GPU, so pass it explicitly.

## 4. What was changed in this repo, and why

| Change | Why |
|---|---|
| `alaya/data/wbench.py`: `include_non_navigation` | `WBenchNaviDataset` kept only the 158 cases with a W/A/S/D/arrow turn. The flag keeps all 289. |
| `alaya/data/wbench.py`: `_perspective_switch_clauses` | `perspective_switch` actions are codes (`fp_to_tp`, `tp_to_tp: switch to follow ...`) and were appended to the `<camera>` block — which `_strip_camera()` deletes wholesale, so those 61 turns reached the model as no-ops. They are now rendered into the scene narrative using WBench's own `PERSPECTIVE_TYPE_DESC` strings verbatim (the same text its judge is shown), so no wording is invented. Only the newest switch applies (it is a state, not an accumulating event). |
| `alaya/data/wbench.py`: `_short_subject` article strip | `desc.lstrip("aA ")` strips a *character set*, not an article: "an elf ..." became "n elf ...". Affects 13 of the 289 cases. Only the `WBENCH_ACTION_CLAUSES` path consumes it. |
| `alaya/model/loader.py`: `ALAYA_INIT_TRANSFORMER_ON_CPU` | 13B bf16 = 26GB does not fit unsharded on a 24GB card. Building on CPU lets FSDP move and shard the model one attention block at a time. |
| `alaya/model/loader.py`: streaming state-dict reads | The LTX-2.3 release file is 43GB but only ~26GB of it is the video DiT. Reading tensor by tensor avoids materialising the audio branch, vocoder and VAE per rank. |
| `alaya/trainer/rollout_trainer.py`: `validation.per_sample_seed` | The rollout noise was drawn from the rank's global CUDA stream, so a case's video depended on which rank it landed on and how many samples ran before it. Keyed on the case id instead, a case reproduces from `run.seed` alone. |
| `alaya/model/loader.py`: `ALAYA_SKIP_TEXT_ENCODER`, `ALAYA_GEMMA_DEVICE_MAP`, `ALAYA_GEMMA_MAX_MEMORY` | Serve prompts from the on-disk cache instead of holding a 24GB text encoder; and let the precache pass spread Gemma over two cards. |
| `alaya/model/loader.py`: `_release_host_arenas()` | `LTX23Model` is built in fp32 (52GB) and cast to bf16; glibc keeps the freed arenas, so five ranks pinned all 251GB of host RAM. |
| `alaya/trainer/rollout_trainer.py`: `_decode_latent_overlap_tiled` | A whole-video VAE decode at 544x960 exceeds a 24GB card. Ported the da3 engine's overlap-tiling decode, which matches a whole decode to within ~0.1/255 mean (max 13/255, measured at overlap 6) rather than leaving a seam per chunk. |
| `alaya/trainer/rollout_trainer.py`: benchmark output written before the diagnostic dump; `validation.save_debug_videos` | The diagnostic strip decodes far more pixels than the output video and OOM'd on 24-round rollouts, taking the real output down with it. |
| `alaya/trainer/rollout_trainer.py` + `alaya/train.py`: `ALAYA_SKIP_TRAIN_DATALOADER` | `--validate-only` built the training dataloader, so rendering videos required the Sekai training corpus to be on disk. |
| `configs/wbench_full.yaml`: `vigeo_cache_budget` 262144 -> 65536 | The shipped value is sized for 80GB cards; ViGeo's KV cache alone then needs ~12GB on top of the sharded DiT. |
| `configs/wbench_full.yaml`: `dmd.enabled: false` | Validation never touches the DMD critic, but `dmd.enabled` builds a second frozen 13B score model. The 4-step student still loads via `paths.dmd_resume`. |

Two of these change the shipped defaults: `validation.per_sample_seed` is now true
(reproducibility should not be opt-in), and `scripts/finetune/train.sh` defaults to
FA2 with FA3 opt-in via `ALAYA_USE_FA3=1` (FA3 is Hopper-only, must be built locally,
and aborts the process at the C++ level when it meets the xformers ViGeo/DA3 pull in).
The rest are additive and inert unless enabled.

Interaction turns reuse the previous navigation action, and a case that never
navigates gets `W` throughout (`dataset.no_navigation_action`; upstream only rendered
navigation cases, so it never reached these 131). No WBench metric grades the camera on
those cases (`navigation_trajectory` is scored on 0 of the 131) while `dynamic_degree`
is scored on all of them. A paired A/B on all 131 (`experiments/2026-09-30_wbench_ab`)
found no gain from holding the camera (`stop`): AutoResearcher score +0.001, 95% CI
[-0.009, +0.011]; `stop` was worse per case on 79 of 131. It drops `dynamic_degree`
from 0.93 to 0.09 and raises every consistency and quality metric, but leaves the
interaction-adherence metrics where they were. So `W` stays the default.

## 5. Two traps worth knowing

**`lr_proj` is snapshotted, not loaded.** The history encoder's `lr_proj_weight` /
`lr_proj_bias` are `persistent=False` buffers, so they are absent from
`history_encoder.pt` and its `missing=0` says nothing about them. They are filled in
`RolloutTrainer.setup()` by copying `transformer.patchify_proj` *at that moment*,
which is before `load_checkpoint_weights` runs. So the base transformer weights must
already be loaded by then -- skipping or deferring that read leaves the memory branch
projecting through whatever `patchify_proj` happened to hold (random init, seeded
`run.seed + rank`, i.e. a different model on every rank). Stage2b training used the
same ordering, so the LTX-base snapshot is what the released history encoder expects.

**Rollout noise.** `validation.per_sample_seed` now defaults to true. Set it false and
the noise is drawn from the rank's global CUDA stream instead, so a case's video
depends on which rank it landed on and how many samples preceded it -- a different GPU
count, case order, subset or `--resume` all change the result. Seeded, a case is a
function of `run.seed` and its case id alone (verified: the same case rendered on rank
1 of a 4-case run and rank 0 of a 1-case run is pixel-identical).

## 6. Notes on this hardware (5 x RTX 4090, 24GB)

- Startup is ~6 min: ranks build the model one at a time (`ALAYA_SERIAL_MODEL_LOAD=1`)
  because five concurrent fp32 builds need ~260GB of host RAM.
- Steady-state VRAM is 19-21GB per card, peaking near 24GB on the longest (27-round)
  rollouts. `spatial_memory.vigeo_cache_budget` is the first knob to lower if a
  longer schedule OOMs.
- Throughput is ~30-35s per collective rollout round, so the full split is ~6-7h.
