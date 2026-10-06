# Fine-tuning AlayaWorld on your own videos

This guide covers one workflow: a **LoRA fine-tune of the released stage 2b checkpoint**
(`weights/alaya-world-ar`) on your own data, on 24GB cards. It needs no code changes. You put
your data in one of three standard formats, copy an example recipe, and run four commands:

```bash
python scripts/tools/check_dataset.py            --config configs/my_finetune.yaml   # 1. check the data
python scripts/tools/precache_train_text_embeds.py --config configs/my_finetune.yaml # 2. encode the prompts
CONFIG_PATH=configs/my_finetune.yaml bash scripts/finetune/lowcompute_4x4090.sh    # 3. train
ls outputs/my_finetune/checkpoint-*/                                                 # 4. weights
```

Environment, base weights and ViGeo are set up as in [`docs/WBENCH.md`](WBENCH.md) section 1;
activate the `alayaworld` conda env before running anything (in AutoResearcher it is `AutoResearcher/.envs/alayaworld`: `conda activate <path>`). Why the recipe looks the way it
does on 24GB cards (LoRA, no text encoder, serial loading) is in [`docs/LOWCOMPUTE.md`](LOWCOMPUTE.md).

Contents:
1. [Data](#1-data): modalities, the three formats, checking a dataset
2. [Config](#2-config): a recipe for your data, segment vs per-chunk prompts, key fields
3. [Launch](#3-launch): starting a run, logs, where the weights are

---

## 1. Data

### 1.1 How the model sees a training sample

Training never shows the model a whole clip. Each step draws a **window** of 57 frames at
24 fps (2.375 s) from a clip. The first 25 frames are history, and the model learns to generate
the next 32 frames, one rollout **round** (a *chunk*), from them. At inference the same thing
repeats: round 1 generates frames 25-56, round 2 frames 57-88, and so on, each round
conditioned on the ones before it.

This is why clips must be at least 2.375 s long, why the frame rate must be at least 24 fps, and
why the per-chunk prompt mode (section 2.4) cares where prompt changes fall.

One **epoch** is one window from every clip, so the number of clips, not their length, sets how
many samples an epoch has. You need **at least as many clips as GPUs**: clips are split evenly
across GPUs and the remainder is dropped, and a run with fewer clips than GPUs stops with an error
before training.

### 1.2 Modalities

A **modality** is one kind of information about a clip. A **format** (section 1.3) is a fixed
combination of modalities.

#### Video

The clip itself: one `.mp4` per clip, `videos/<id>.mp4`.

- **Frame rate at least 24 fps** (23.976 counts). Higher rates are subsampled to 24 fps: 30 fps
  keeps every 1.25th frame, 60 fps every 2.5th. The checker rejects lower rates.
- **At least 2.375 s long.** Longer is better: every window position is a different sample.
- **Aspect ratio close to 16:9.** Frames are resized to `sample.width` x `sample.height`
  (736x416) **without cropping**, so a 4:3 video is squashed. Crop before saving.
- Any resolution. It is resized anyway; 720p is plenty.

Prepare it with any tool that writes H.264 or MPEG-4 mp4. To crop a 4:3 source to 16:9 and trim it to 10 s:

```bash
ffmpeg -i raw.mov -t 10 -vf "crop=iw:iw*9/16" -c:v libx264 -crf 18 videos/street_0001.mp4
```

#### Caption

One sentence or paragraph describing the whole clip: what is in the scene, what moves and how,
and the look (lighting, style). Stored in `captions/<id>.json`:

```json
{"caption": "A fixed camera in a small office with a desk, a whiteboard and bookshelves; a man in a red T-shirt walks in, stops by the shelves and reads a book."}
```

Write captions in the style you will prompt with at inference. The model learns the
association between these words and what happens on screen.

#### Timed prompts

Several prompts for one clip, each tied to a time range, for clips where something changes
partway through: a style change, an event, weather, a new instruction. This is how you teach
the model **interactive prompting**, where the user changes the prompt mid-rollout and the
video follows. Stored in the same `captions/<id>.json`, next to the caption:

```json
{
  "caption": "Colonoscopy video inside a silicone colon phantom that turns into a black-and-white pencil sketch drawing partway through while the camera keeps moving.",
  "segments": [
    {"time_range_s": [0.0,    3.7083], "prompt": "Colonoscopy video recorded inside a silicone colon phantom: the colonoscope camera slowly pulls back and pushes forward through the glossy pink mucosal lumen, rolling as it moves."},
    {"time_range_s": [3.7083, 5.0417], "prompt": "Colonoscopy video inside a silicone colon phantom that is smoothly transforming into a black-and-white pencil sketch drawing of the same colon while the camera keeps moving."},
    {"time_range_s": [5.0417, 8.0],    "prompt": "A black-and-white pencil sketch drawing of the inside of a colon, with grainy paper texture and dark pencil strokes, as the camera slowly moves through the lumen."}
  ]
}
```

- `time_range_s` is `[start, end)` in seconds of the video. Segments must not overlap and must
  end within the video; gaps are allowed (no prompt is drawn there).
- Each prompt describes what is on screen **during its range**, phrased the way a user would
  prompt at that moment.
- `caption` is still required. Segment mode trains on it 10% of the time
  (`data.overall_caption_prob`); per-chunk mode does not use it.
- If you train with `prompt_mode: per_chunk` (section 2.4), boundaries between prompts must sit
  on round boundaries, **25/24 + k x 32/24 s**: 1.042, 2.375, 3.708, 5.042, 6.375, 7.708 s, and so
  on. The example above changes prompt at 3.708 s (round 3 starts) and 5.042 s (round 4). Plan
  your events or edits around those times, or cut clips so they line up.

#### Camera extrinsics (pose)

Where the camera is and where it looks, **for every video frame**: a camera-to-world 4x4
matrix per frame, in OpenCV convention (x right, y down, z forward). Stored as
`poses/<id>.npz` with array `cam_c2w`, shape `[N, 4, 4]`, where N is the number of frames in the
mp4 (not the 24 fps count):

```python
import numpy as np
c2w = np.load("poses/street_0001.npz")["cam_c2w"]   # (300, 4, 4) for a 10 s 30 fps clip
c2w[0]
# [[ 1.  0.  0.  0. ]
#  [ 0.  1.  0.  0. ]
#  [ 0.  0.  1.  0. ]
#  [ 0.  0.  0.  1. ]]
```

- Rotations must be orthonormal and the bottom row `[0, 0, 0, 1]`.
- Translation units are used as given (`data.camera_norm_mode: none`); use one consistent scale
  within a clip. Monocular estimates in arbitrary units are fine.
- Where to get poses: from the capture device (game engine, robot, AR session), or estimated
  from the video with a structure-from-motion or video depth/pose model (COLMAP, MegaSaM,
  VGGT, and similar). Convert world-to-camera outputs by inverting them.
- If your tool annotates only some frames, interpolate (slerp rotations, lerp translations)
  to every frame. `alaya/data/spatialvid.py` does this for SpatialVID's 5 Hz poses.

The model's spatial memory uses the poses and intrinsics to warp frames it has already
seen into the view being generated, so wrong poses quietly teach wrong geometry. Run the
checker.

#### Camera intrinsics

The lens: focal length and principal point, as a 3x3 matrix in **pixels of the mp4's own
resolution**. Stored in the same `.npz` as `intrinsics`, shape `[3, 3]` (or `[N, 3, 3]`, in
which case the matrix for the first frame of each training window is used):

```python
np.load("poses/street_0001.npz")["intrinsics"]
# [[900.55,   0.  , 640.],     fx, cx
#  [  0.  , 900.55, 360.],     fy, cy
#  [  0.  ,   0.  ,   1.]]
```

- Principal point at the image centre: the loader reads the image size as `2*cx` x `2*cy`.
- Optional. Without it, training assumes fx = width and fy = height (about a 53° horizontal
  field of view), and the checker warns.
- If you only know the horizontal field of view: `fx = fy = (width / 2) / tan(hfov / 2)`.

#### What is not a modality here

- **Actions** (keyboard, joystick) are not a separate file. The recipe trains with
  `control.candidates: [[]]`, so the model follows the camera through its pose and memory
  conditioning, not an action signal.
- **Depth, masks, audio**: not used by this recipe.

### 1.3 The three formats

Every format uses the same directory layout. Files are paired by `<id>`:

```
<root>/
  videos/<id>.mp4
  captions/<id>.json
  poses/<id>.npz        # camera formats only
```

| Format | Video | Caption | Timed prompts | Extrinsics | Intrinsics | Use it for |
|---|---|---|---|---|---|---|
| `video_caption_camera` | required | required | - | required | optional | moving-camera footage with one description per clip |
| `video_timed_prompts_camera` | required | required | required | required | optional | clips where the content or style changes and the prompt should follow |
| `video_caption_static` | required | required | - | - (fixed) | - | tripod, surveillance or other fixed-camera footage |

For `video_caption_static`, training uses an identity pose for every frame. Do not use it for a
camera that pans or shakes even slightly: the model would learn that camera motion happens
without a pose change. Any `poses/` directory is ignored.

Each format has a small real example dataset and a recipe. The datasets are not in git; they are
built locally under `data/examples/`, and the recipes in `configs/examples/` point at them.

| Format | Example recipe | Example data (local) |
|---|---|---|
| `video_caption_camera` | [`configs/examples/finetune_video_caption_camera.yaml`](../configs/examples/finetune_video_caption_camera.yaml) | `data/examples/video_caption_camera`: 6 SpatialVID-HQ clips, 9-15 s each at 24-50 fps, with captions and per-frame poses |
| `video_timed_prompts_camera` | [`configs/examples/finetune_video_timed_prompts_camera.yaml`](../configs/examples/finetune_video_timed_prompts_camera.yaml) | `data/examples/video_timed_prompts_camera`: 6 C3VD colonoscopy clips, 8 s each, that turn into a pencil sketch during round 3 (the timed-prompt example above) |
| `video_caption_static` | [`configs/examples/finetune_video_caption_static.yaml`](../configs/examples/finetune_video_caption_static.yaml) | `data/examples/video_caption_static`: 6 clips, 10 s each, two from each of three CDnet 2012 fixed-camera scenes (highway, office, peopleInShade) |

### 1.4 Check the dataset

The loader skips a clip it cannot read and silently draws another, so a dataset with a
systematic problem trains on whatever is left. Check it first. The checker reads the window
layout from the same recipe you will train with:

```bash
python scripts/tools/check_dataset.py --config configs/examples/finetune_video_timed_prompts_camera.yaml
```

```
layout: 57 frames per window at 24 fps (25 history + 32 target)

[OK] example (video_timed_prompts_camera, prompt_mode=per_chunk) at data/examples/video_timed_prompts_camera: 6 clips, 0.8 min, 30 round-aligned windows, 0 errors, 0 warnings
```

It exits 1 on any error.

- **Errors:** a missing caption or pose, an empty caption, a clip too short or below 24 fps,
  pose count different from the frame count, a non-rigid pose, overlapping segments or ones
  past the end of the video, and per-chunk boundaries that are not on round boundaries.
- **Warnings:** missing intrinsics, an off-centre principal point, segments too short for
  segment mode, and files without a matching video.

---

## 2. Config

### 2.1 Start from the example recipe

Copy the example for your format and give the run its own name:

```bash
cp configs/examples/finetune_video_caption_camera.yaml configs/my_finetune.yaml
```

The examples are `configs/stage2b_arsft_lowcompute.yaml` with only the data section, run name,
validation dataset and prompt cache changed, so everything else (model, memory, anti-drift
regularizers, LoRA) is the tested low-compute stage 2b setup.

### 2.2 Point it at your data

Datasets go under `data.datasets`, one entry per dataset:

```yaml
run:
  name: my_finetune
  output_dir: ./outputs/my_finetune      # checkpoints go here
  log_dir: ./logs/my_finetune
data:
  sources: {}                            # built-in, code-registered corpora; leave empty
  datasets:
    street:                              # any name that is not a built-in source
      root: /data/street_clips           # absolute, or relative to the repo root
      format: video_caption_camera
      weight: 1.0
    street_night:
      root: /data/street_night_clips
      format: video_timed_prompts_camera
      prompt_mode: per_chunk             # timed prompts only: per_chunk or segment
      weight: 0.5                        # drawn half as often as `street`
runtime:
  text_embed_cache_dir: cache/text_embed_my_finetune
```

| Field | Meaning |
|---|---|
| `root` | the dataset directory from section 1.3 |
| `format` | `video_caption_camera`, `video_timed_prompts_camera` or `video_caption_static` |
| `prompt_mode` | timed prompts only: how windows and prompts are paired, section 2.4 |
| `weight` | relative sampling weight between datasets; `0` disables one |

Several datasets of different formats can be mixed in one run. Unknown keys and invalid
combinations are rejected when the config loads.

The recipe has a validation mode whose `dataset.source` names a dataset; validation is off by
default (`validation.enabled: false`). If you rename `example`, rename it there too.

### 2.3 Mix datasets and formats

One run can train on any number of datasets, each in its own format and, for timed prompts,
with its own `prompt_mode`, through the same weighted mixing the loader uses for built-in
corpora. Mixing is also how to keep the model's existing abilities while teaching a new one:
mix your new data with general footage instead of training on the new data alone.
[`configs/examples/finetune_mixed.yaml`](../configs/examples/finetune_mixed.yaml) mixes all
three example datasets:

```yaml
data:
  datasets:
    cam:    {root: data/examples/video_caption_camera,       format: video_caption_camera,       weight: 1.0}
    pencil: {root: data/examples/video_timed_prompts_camera, format: video_timed_prompts_camera, prompt_mode: per_chunk, weight: 2.0}
    static: {root: data/examples/video_caption_static,       format: video_caption_static,       weight: 1.0}
```

- **Weights are sampling probabilities.** Each draw picks a dataset with probability
  `weight / sum(weights)` (25% cam, 50% pencil, 25% static here), whatever the dataset sizes.
  A small dataset with a large weight is repeated to reach its share, so it overfits sooner.
- **One epoch** is sized so the dataset needing the most draws for its share is seen about once:
  `max(clips / share)` windows, here max(6/0.25, 6/0.5, 6/0.25) = 24.
- **Each sample keeps its own format.** A static clip gets a fixed camera, a camera clip its
  poses, a timed-prompt clip its round-aligned window and prompt. Different GPUs can draw
  different formats in the same optimizer step.
- The checker and the prompt precache handle every dataset in the config in one call.

Measured with that config on 4x RTX 4090:
- **Draws:** one epoch's 24 draws came out 6 / 12 / 6.
- **Poses:** every static sample had identity poses, and every camera sample had moving poses.
- **Training:** 8 steps drew from all three datasets and wrote a checkpoint, with an 18.35GB
  peak per GPU.

### 2.4 Timed prompts: `per_chunk` or `segment`

Both modes read the same `video_timed_prompts_camera` data. They differ in which windows are
drawn and which prompt each window is trained with.

**`per_chunk`**: windows start only on round boundaries (frame 0, 32, 64, ... at 24 fps), exactly
where an interactive rollout's rounds fall. Each window is trained with the prompt active at the
midpoint of its 32 target frames. When the prompt changes at a round boundary, the window for that
round has history under the old prompt and a target under the new one. That is the
interactive-prompting situation: the prompt switches and the next round must follow it.

**`segment`**: each draw picks one segment, then places the window at a random position
**inside** it and trains with that segment's prompt (or with `caption`, 10% of the time).
A window never spans two prompts. Segments shorter than one window (2.375 s) are never drawn.

Measured on the example dataset through the training loader (300 draws each):

| | `segment` | `per_chunk` |
|---|---|---|
| prompts trained | raw 129, sketch 139, `caption` 32, **transforming 0** | raw 131, transforming 51, sketch 118 |
| windows whose history is under a different prompt than the trained one | **0%** | **38.3%** |
| distinct windows per 8 s, 30 fps clip | 60 (41 start frames inside raw, 19 inside sketch) | 5 (one per round) |

The "transforming" segment lasts 1.33 s, shorter than a window, so segment mode never sees it.
Per-chunk mode trains the transition in rounds 3 and 4, where the prompt changes.

Use **`per_chunk`** when the point is to react to a prompt change during rollout: style
switches, triggered events, any "now do X" instruction.
- **Pro:** it teaches the switch itself, at the timing inference uses, including changes
  shorter than a window.
- **Con:** only round-aligned windows, so fewer distinct samples per clip (overfits sooner on
  little data).
- **Con:** prompt boundaries must be on round boundaries (the checker enforces this).

Use **`segment`** when prompts describe long, mostly steady stretches and you want the model to
follow whichever prompt it is given, for example a long walk annotated scene by scene.
- **Pro:** more varied windows from the same data.
- **Pro:** arbitrary boundaries.
- **Pro:** the whole-clip caption is mixed in.
- **Con:** it never trains a window where the prompt changes, so it will not teach a
  mid-rollout switch.
- **Con:** segments under 2.375 s are wasted.

### 2.5 Key fields

The defaults in the example recipes are the tested low-compute stage 2b settings. These are the
ones you are most likely to change:

| Field | Example value | What to consider |
|---|---|---|
| `run.name`, `run.output_dir`, `run.log_dir` | per recipe | give every run its own; checkpoints and logs go here |
| `data.datasets` | one dataset | section 2.2 |
| `data.overall_caption_prob` | `null` (0.1) | segment mode only: how often `caption` replaces the segment prompt |
| `optimizer.lr` | `5.0e-05` | LoRA learning rate. The 29-clip C3VD overfit test used `2.0e-04` for 300 steps |
| `optimizer.max_steps` | `300` | optimizer steps. Windows seen = steps x GPUs x `batch_size` x `grad_accum_steps`. Training stops at `max_steps` or after `optimizer.epochs` (5000) epochs, whichever comes first |
| `optimizer.grad_accum_steps` | `4` | effective batch = GPUs x `batch_size` x this; `batch_size` must stay 1 on 24GB |
| `optimizer.warmup_steps` | `50` | linear warmup |
| `optimizer.checkpoint_steps` | `100` | save interval |
| `optimizer.max_checkpoints` | `5` | older checkpoints are deleted beyond this |
| `lora.rank`, `lora.alpha` | `64`, `64` | adapter size; lower rank if memory is tight. Pass the same values to the merge tool |
| `memory.drop_prob`, `anti_drift.*` | stage 2b values | regularizers that keep long rollouts stable; leave on |
| `sample.height`, `sample.width` | `416`, `736` | training resolution; both must be divisible by 32. 544x960 also fits on 4x 24GB (measured: 20.1GB peak per rank against 19.0GB, ~30% slower per step) |
| `runtime.text_embed_cache_dir` | per recipe | where prompt embeddings are cached, section 2.6 |
| `validation.enabled` | `false` | a rollout during training costs minutes; generate from a checkpoint instead (section 3.4) |

Do not change the `layout`, `spatial_memory`, `training.mode` or `next_forcing` sections for
a fine-tune. They describe the released checkpoint's architecture and window layout.

### 2.6 Encode the prompts

On 24GB cards the text encoder (Gemma-3-12B, 24GB) cannot be loaded next to the model, so
training reads prompt embeddings from a disk cache and **stops with an error on a prompt that
is not cached**. Fill the cache once per dataset, and again whenever captions change:

```bash
CUDA_VISIBLE_DEVICES=0,1 ALAYA_GEMMA_MAX_MEMORY="0=13GiB,1=13GiB" \
  python scripts/tools/precache_train_text_embeds.py --config configs/my_finetune.yaml --device-map auto
```

It enumerates every caption and every segment prompt the config can draw, the negative prompt
and any literal `prompt_schedule` prompts (section 3.4), and skips any already on disk. `--dry-run` only counts them. It takes two GPUs for
Gemma; after loading it encodes about 2.5 prompts per second.

---

## 3. Launch

### 3.1 Start training

```bash
conda activate alayaworld
CONFIG_PATH=configs/my_finetune.yaml bash scripts/finetune/lowcompute_4x4090.sh
```

- One process per GPU in `CUDA_VISIBLE_DEVICES`, defaulting to `0,1,2,3`.
- The launcher sets the four environment variables the 24GB setup needs (explained in
  [`docs/LOWCOMPUTE.md`](LOWCOMPUTE.md) section 3) and runs `scripts/finetune/train.sh`.
- Add `LOG_FILTER=all` to keep the full output in the log instead of only step lines.
- Startup takes several minutes: ranks load the 26GB model one at a time to stay within host
  RAM.

### 3.2 Watch it

Logs go to `<run.log_dir>/<config name>/train_node0_<timestamp>.log`, e.g.
`logs/my_finetune/my_finetune/train_node0_20260917_101500.log`. One line per optimizer step;
`loss` and `time` are those of the step's last micro-batch:

```
[Train] step=2 epoch=1 source=example video=79987614-... fs=184 fe=300 K=4 ... sigma=0.996 loss=0.261719 grad=0.0579 lr=5.00e-05 time=7.92s
```

- `fs`/`fe` are the window's first and last source frame, handy for checking what was drawn.
- `ALAYA_LOG_MEMORY=1` adds per-rank peak memory lines.
- A cache miss, a missing file or an out-of-memory error stops the run with the error in the
  same log.

### 3.3 Where the weights are

Every `optimizer.checkpoint_steps` steps, and at `max_steps`, a checkpoint is written to
`<run.output_dir>/checkpoint-<step>/`:

```
outputs/my_finetune/checkpoint-300/
  lora.safetensors      # the LoRA adapters (~654MB at rank 64): the fine-tuned part of the DiT
  history_encoder.pt    # the fine-tuned memory HistoryEncoder (~34MB)
  trainer_state.pt      # optimizer and step state
  README.txt
```

The base transformer is not copied; a checkpoint only makes sense on top of
`weights/alaya-world-ar`. Only the newest `optimizer.max_checkpoints` are kept.

### 3.4 Use the weights

**Load the checkpoint directly** into any generation config built on the same base (for
example a copy of your recipe run with `VALIDATE_ONLY=1`). Point both paths at the checkpoint;
the HistoryEncoder is taken from `paths.history_encoder`, not from the checkpoint directory:

```yaml
paths:
  resume_checkpoint: weights/alaya-world-ar
  dmd_resume: outputs/my_finetune/checkpoint-300                        # loads lora.safetensors
  history_encoder: outputs/my_finetune/checkpoint-300/history_encoder.pt
```

**Or merge** the LoRA into a standalone `transformer.pt` that the inference and WBench configs
load like the released checkpoint:

```bash
python scripts/tools/merge_lora_for_rollout.py \
    --ckpt_dir outputs/my_finetune/checkpoint-300 \
    --base_transformer weights/alaya-world-ar/transformer.pt \
    --output outputs/my_finetune/checkpoint-300-merged \
    --lora_rank 64 --lora_alpha 64
```

Merging writes 26GB and needs ~52GB of host RAM. Rendering on 24GB cards needs four config
changes; both are covered in [`docs/LOWCOMPUTE.md`](LOWCOMPUTE.md) section 6.

To try an interactive prompt change, give the validation mode a `prompt_schedule` with one entry
per round, for example `["caption", "caption", "<transition prompt>", "<new style prompt>"]`,
where `caption` means the clip's own caption. Then run `precache_train_text_embeds.py` on that
config: it caches the literal schedule prompts along with the training prompts. If the new style
matches words in the default `validation.negative_prompt` (it lists "grainy texture" and
"stylized filters", for example), set `negative_prompt: ''`, or classifier-free guidance will
steer away from the style.

---

## 4. Verified

Each format was verified on its example dataset, on 4x RTX 4090, with exactly the commands in
this guide:
1. `check_dataset.py` passes.
2. The prompts are precached.
3. A two-step training run (`max_steps: 2`, `grad_accum_steps: 1`) trains and writes `checkpoint-2`
   with `lora.safetensors` and `history_encoder.pt`, in about 7 minutes including model loading.

The `video_timed_prompts_camera` format was run in both prompt modes. Per-chunk windows were
compared against the hand-cut round windows of the C3VD pencil-sketch overfit experiment that
trained successfully: same frames (a one-frame shift is 2-13x further off) and identical
prompts. The loader logic is covered by `tests/test_standard_dataset.py` and
`tests/test_standard_check.py`.
