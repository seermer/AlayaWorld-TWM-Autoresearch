from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
from typing import Callable

import safetensors
import safetensors.torch
import torch
import torch.nn as nn

from fastvideo.ltx2_streaming_vae import StreamingVAEEncoder
from ltx2.modules.attention import AttentionFunction
from ltx2.modules.model_ltx_2_3 import LTX23Model
from ltx2.modules.rope import LTXRopeType
from ltx2.modules.vae import create_video_decoder, create_video_encoder

from alaya.config.schema import TrainConfig
from alaya.model.components import ModelComponents
from alaya.model.fsdp import maybe_enable_gradient_checkpointing
from alaya.model.lora import LoRAForwardManager


def _release_host_arenas() -> None:
    """gc + malloc_trim: return freed host pages to the OS after a big load."""
    import ctypes
    import gc

    gc.collect()
    try:
        ctypes.CDLL(None).malloc_trim(0)
    except (AttributeError, OSError):
        pass


def _rank0() -> bool:
    """torchrun sets RANK per process (0 when unset). Print from rank 0 only to avoid duplicate lines."""
    import os
    return os.environ.get("RANK", "0") == "0"


def build_model_components(cfg: TrainConfig, device: torch.device, dtype: torch.dtype) -> ModelComponents:
    checkpoint_path = cfg.paths.effective_transformer
    if _rank0():
        print(f"[Paths] transformer checkpoint = {checkpoint_path}")
    transformer = load_transformer(checkpoint_path, cfg, device=device, dtype=dtype)
    if cfg.training.mode == "lora":
        for param in transformer.parameters():
            param.requires_grad_(False)
    elif cfg.training.mode == "sft":
        for param in transformer.parameters():
            param.requires_grad_(True)
    else:
        raise ValueError(f"unknown training.mode={cfg.training.mode!r}")

    if cfg.control.uses("action"):
        action_count = _set_action_adaln_trainable(transformer, enabled=True)
        if action_count == 0:
            raise RuntimeError("control uses 'action' but no action AdaLN parameters were created")
        print(f"[Control:action] trainable action_adaln tensors={action_count}")

    lora_manager = None
    if cfg.training.mode == "lora" and cfg.lora.enabled:
        lora_manager = LoRAForwardManager(trainable=cfg.lora.train)
        count = lora_manager.init_for_training(
            transformer,
            target_keywords=cfg.lora.targets,
            rank=cfg.lora.rank,
            alpha=cfg.lora.alpha,
            dtype=dtype,
            device=device,
        )
        hooks = lora_manager.register_hooks(transformer)
        lora_manager.enable()
        print(f"[LoRA] initialized={count} hooks={hooks} trainable={cfg.lora.train}")
    elif cfg.training.mode == "sft" and cfg.lora.enabled:
        print("[LoRA] ignored because training.mode=sft")

    maybe_enable_gradient_checkpointing(transformer, cfg.runtime.gradient_checkpointing)

    score_model = None
    critic_lora = None
    gan_discriminator = None
    if cfg.dmd.enabled:
        score_model, critic_lora = build_dmd_score_model(cfg, device=device, dtype=dtype)
        if cfg.dmd.is_use_gan:
            from alaya.dmd.discriminator import GanDiscriminator

            gan_discriminator = GanDiscriminator(
                score_model,
                hooks=list(cfg.dmd.gan_hooks),
                inner_dim=score_model.inner_dim,
                cond_map_dim=cfg.dmd.gan_cond_map_dim,
                dtype=dtype,
                device=device,
            )
            n_params = sum(p.numel() for p in gan_discriminator.parameters()) / 1e6
            print(
                f"[DMD-GAN] discriminator heads on blocks={cfg.dmd.gan_hooks} "
                f"inner_dim={score_model.inner_dim} cond_map_dim={cfg.dmd.gan_cond_map_dim} "
                f"params={n_params:.1f}M"
            )

    next_forcing_head = None
    if cfg.next_forcing.enabled:
        from alaya.dmd.next_forcing import NextForcingHead

        next_forcing_head = NextForcingHead(
            transformer,
            hook_layers=list(cfg.next_forcing.hook_layers),
            num_blocks=cfg.next_forcing.num_blocks,
            fuse_hidden_mult=cfg.next_forcing.fuse_hidden_mult,
            dtype=dtype,
            device=device,
        )
        nf_params = sum(p.numel() for p in next_forcing_head.parameters()) / 1e6
        print(
            f"[NextForcing] depth=1 MCP head: hooks={cfg.next_forcing.hook_layers} "
            f"num_blocks={cfg.next_forcing.num_blocks} loss_weight={cfg.next_forcing.loss_weight} "
            f"sigma_shift={cfg.next_forcing.sigma_shift} params={nf_params:.1f}M"
        )

    vae_encoder_raw, vae_decoder = load_vae(cfg.paths.vae, device=device, dtype=dtype)
    vae_encoder = StreamingVAEEncoder(vae_encoder_raw, device=device, dtype=dtype)
    if os.environ.get("ALAYA_SKIP_TEXT_ENCODER", "0") == "1":
        # Gemma-3-12B is 24GB in bf16 and does not fit next to the DiT on a 24GB card.
        # With every reachable prompt already in runtime.text_embed_cache_dir (see
        # scripts/tools/precache_wbench_text_embeds.py) the encoder is never called,
        # so skip loading it. A cache miss raises instead of silently mis-encoding.
        if _rank0():
            print("[TextEncoder] skipped (ALAYA_SKIP_TEXT_ENCODER=1); prompts must come from the disk cache")
        text_encoder, encode_text = None, _text_encoder_disabled
    else:
        text_encoder, encode_text = load_text_encoder(checkpoint_path, cfg.paths.gemma, device=device, dtype=dtype)

    return ModelComponents(
        transformer=transformer,
        vae_encoder=vae_encoder,
        vae_decoder=vae_decoder,
        text_encoder=text_encoder,
        encode_text=encode_text,
        lora_manager=lora_manager,
        score_model=score_model,
        critic_lora=critic_lora,
        gan_discriminator=gan_discriminator,
        next_forcing_head=next_forcing_head,
    )


def build_dmd_score_model(cfg: TrainConfig, device: torch.device, dtype: torch.dtype):
    """Build the frozen real score model plus trainable critic LoRA for DMD."""
    base = cfg.paths.real_score_model or cfg.paths.effective_transformer
    if not base:
        raise ValueError("dmd.enabled requires paths.real_score_model or an effective transformer path")
    print(f"[DMD] real/fake score base = {base}")
    score_model = load_transformer(base, cfg, device=device, dtype=dtype)
    score_model.requires_grad_(False)

    critic_lora = LoRAForwardManager(trainable=True)
    count = critic_lora.init_for_training(
        score_model,
        target_keywords=cfg.dmd.critic_lora_targets or cfg.lora.targets,
        rank=cfg.dmd.critic_lora_rank,
        alpha=cfg.dmd.critic_lora_alpha,
        dtype=dtype,
        device=device,
    )
    hooks = critic_lora.register_hooks(score_model)
    critic_lora.enable()
    maybe_enable_gradient_checkpointing(score_model, cfg.runtime.gradient_checkpointing)
    print(f"[DMD] critic LoRA initialized={count} hooks={hooks} rank={cfg.dmd.critic_lora_rank}")
    return score_model, critic_lora


def load_transformer(checkpoint_path: str, cfg: TrainConfig, device: torch.device, dtype: torch.dtype) -> LTX23Model:
    _configure_control_env(cfg)
    config = _read_transformer_config(checkpoint_path)
    config.update(_runtime_transformer_overrides(cfg))

    valid_params = set(inspect.signature(LTX23Model.__init__).parameters.keys())
    filtered = {key: value for key, value in config.items() if key in valid_params}
    model = LTX23Model(**filtered)

    # NOTE: do not be tempted to skip this read because paths.resume_checkpoint
    # overwrites every tensor anyway. The history encoder's lr_proj buffers are
    # snapshotted from transformer.patchify_proj in RolloutTrainer.setup(), which runs
    # *before* the resume load, so the base weights must already be in place by then.
    if checkpoint_path and Path(checkpoint_path).exists():
        model_keys = {name for name, _ in model.named_parameters()}
        model_keys.update(name for name, _ in model.named_buffers())
        converted = _load_transformer_state_streaming(checkpoint_path, model_keys)
        missing, unexpected = model.load_state_dict(converted, strict=False)
        del converted
        if _rank0():
            print(f"[Transformer] missing={len(missing)} unexpected={len(unexpected)}")

    # ALAYA_INIT_TRANSFORMER_ON_CPU=1 keeps the DiT on CPU here so that
    # maybe_wrap_fsdp() shards it onto the GPUs unit by unit. Needed whenever the
    # unsharded model does not fit in one device (13B bf16 = 26GB vs a 24GB card);
    # each FSDP unit is moved and sharded on its own, so the GPU never holds the
    # whole model. Unset, the behaviour is unchanged.
    if os.environ.get("ALAYA_INIT_TRANSFORMER_ON_CPU", "0") == "1":
        model.to(device="cpu", dtype=dtype)
        # LTX23Model is constructed in fp32 (52GB for 13B), so the bf16 cast leaves
        # ~26GB of freed-but-unreturned arenas behind. With one rank per GPU that is
        # the difference between fitting in host RAM and thrashing, so hand it back.
        _release_host_arenas()
        if _rank0():
            print("[Transformer] kept on CPU for FSDP sharding (ALAYA_INIT_TRANSFORMER_ON_CPU=1)")
    else:
        model.to(device=device, dtype=dtype)
    if _rank0():
        print(f"[Transformer] params={sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    return model


def _read_transformer_config(checkpoint_path: str) -> dict:
    if not checkpoint_path or not Path(checkpoint_path).exists() or not checkpoint_path.endswith(".safetensors"):
        return {}
    with safetensors.safe_open(checkpoint_path, framework="pt") as handle:
        metadata = handle.metadata() or {}
        return json.loads(metadata.get("config", "{}")).get("transformer", {})


def _runtime_transformer_overrides(cfg: TrainConfig) -> dict:
    attention_map = {
        "flash_attention_3": AttentionFunction.FLASH_ATTENTION_3,
        "xformers": AttentionFunction.XFORMERS,
        "pytorch": AttentionFunction.PYTORCH,
    }
    use_action = cfg.control.uses("action")
    return {
        "attention_type": attention_map.get(cfg.runtime.attention_type, AttentionFunction.FLASH_ATTENTION_3),
        "rope_type": LTXRopeType.SPLIT,
        "normalize_time_by_fps": cfg.runtime.norm_by_fps,
        "normalize_rope_positions": cfg.runtime.norm_by_max_frames,
        "positional_embedding_max_pos": [
            int(x.strip()) for x in cfg.runtime.positional_embedding_max_pos.split(",")
        ],
        "apply_gated_attention": True,
        "cross_attention_adaln": True,
        "caption_proj_before_connector": True,
        "enable_action_control": use_action,
        "compact_spatial_tokens": cfg.runtime.compact_spatial_tokens,
    }


def _configure_control_env(cfg: TrainConfig) -> None:
    use_action = cfg.control.uses("action")
    os.environ["LTX_USE_ACTION_CONTROL"] = "1" if use_action else "0"
    os.environ["LTX_ACTION_SCALE"] = cfg.control.action_scale
    os.environ["LTX_ACTION_FREQ_SCALE"] = str(cfg.control.action_freq_scale)
    os.environ["LTX_ACTION_FREQ_DIM_PER_AXIS"] = str(cfg.control.action_freq_dim_per_axis)


def _set_action_adaln_trainable(model: nn.Module, enabled: bool) -> int:
    count = 0
    for name, param in model.named_parameters():
        if "action_adaln_embedder" in name or "action_adaln_projection" in name:
            param.requires_grad_(enabled)
            count += 1
    return count


_TRANSFORMER_SKIP_PREFIXES = (
    "audio_",
    "av_ca_",
    "_a2v_",
    "_v2a_",
    "vae.",
    "vocoder.",
    "text_embedding_projection.",
    "model.diffusion_model.video_embeddings_connector.",
    "model.diffusion_model.audio_",
    "model.diffusion_model.av_ca_",
)


def resolve_transformer_key(raw_key: str, model_keys: set[str]) -> str | None:
    """Map a checkpoint key onto the model's own name, or None if it is not ours."""
    if any(raw_key.startswith(prefix) for prefix in _TRANSFORMER_SKIP_PREFIXES):
        return None
    candidates = []
    if raw_key.startswith("model.diffusion_model."):
        candidates.append(raw_key.removeprefix("model.diffusion_model.").replace("transformer_blocks.", "blocks."))
    cleaned = raw_key.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", "")
    cleaned = cleaned.replace("transformer_blocks.", "blocks.")
    candidates.extend([cleaned, raw_key])
    for key in candidates:
        if key in model_keys:
            return key
    return None


def convert_transformer_state_dict(state_dict: dict, model_keys: set[str]) -> dict:
    converted = {}
    for raw_key, value in state_dict.items():
        key = resolve_transformer_key(raw_key, model_keys)
        if key is not None:
            converted[key] = value
    return converted


def _load_transformer_state_streaming(checkpoint_path: str, model_keys: set[str]) -> dict:
    """Read only the tensors this model actually uses, one at a time.

    The LTX-2.3 release file is 43GB but roughly 26GB of it is the video DiT; the
    rest (audio branch, vocoder, VAE, text projections) is skipped here. Loading
    the whole dict first costs that full 43GB of host RAM per rank, which does not
    scale to one rank per GPU.
    """
    converted: dict = {}
    with safetensors.safe_open(checkpoint_path, framework="pt") as handle:
        for raw_key in handle.keys():
            key = resolve_transformer_key(raw_key, model_keys)
            if key is not None:
                converted[key] = handle.get_tensor(raw_key)
    return converted


def load_vae(
    checkpoint_path: str, device: torch.device, dtype: torch.dtype, state_dict: dict | None = None
) -> tuple[nn.Module, nn.Module]:
    config = {}
    if checkpoint_path and Path(checkpoint_path).exists():
        with safetensors.safe_open(checkpoint_path, framework="pt") as handle:
            config = json.loads((handle.metadata() or {}).get("config", "{}"))

    encoder = create_video_encoder(config)
    decoder = create_video_decoder(config)

    # da3 inference: a preloaded merged state may be shared so the one-file
    # checkpoint is read once; None (vigeo paths) keeps the original behavior.
    if state_dict is None and checkpoint_path and Path(checkpoint_path).exists():
        # Only the vae.* tensors, not the whole (43GB) release file — see
        # _load_transformer_state_streaming for why.
        state_dict = {}
        with safetensors.safe_open(checkpoint_path, framework="pt") as handle:
            for raw_key in handle.keys():
                if raw_key.startswith("vae."):
                    state_dict[raw_key] = handle.get_tensor(raw_key)
    if state_dict is not None:
        enc = {}
        dec = {}
        for key, value in state_dict.items():
            if key.startswith("vae.encoder."):
                enc[key.removeprefix("vae.encoder.")] = value
            elif key.startswith("vae.decoder."):
                dec[key.removeprefix("vae.decoder.")] = value
            elif key.startswith("vae.per_channel_statistics."):
                short = key.removeprefix("vae.")
                enc[short] = value
                dec[short] = value
        encoder.load_state_dict(enc, strict=True)
        decoder.load_state_dict(dec, strict=True)

    encoder.to(device=device, dtype=dtype).eval()
    decoder.to(device=device, dtype=dtype).eval()
    for module in (encoder, decoder):
        for param in module.parameters():
            param.requires_grad_(False)
    return encoder, decoder


def _text_encoder_disabled(*args, **kwargs):
    raise RuntimeError(
        "text encoder is disabled (ALAYA_SKIP_TEXT_ENCODER=1) but a prompt missed the "
        "on-disk embedding cache. Re-run scripts/tools/precache_train_text_embeds.py "
        "for a training run, or scripts/tools/precache_wbench_text_embeds.py for WBench "
        "generation, so every prompt of this run is cached, or unset ALAYA_SKIP_TEXT_ENCODER."
    )


def load_text_encoder(
    checkpoint_path: str,
    gemma_root: str,
    device: torch.device,
    dtype: torch.dtype,
    state_dict: dict | None = None,
) -> tuple[nn.Module, Callable]:
    from transformers import Gemma3ForConditionalGeneration
    from ltx2.modules.text_encoder import (
        AVGemmaTextEncoderModel,
        Embeddings1DConnector,
        GemmaFeaturesExtractorProjLinear,
        LTXVGemmaTokenizer,
    )

    with safetensors.safe_open(checkpoint_path, framework="pt") as handle:
        config = json.loads((handle.metadata() or {}).get("config", "{}"))
    tf_config = config.get("transformer", {})

    caption_proj_before_connector = tf_config.get("caption_proj_before_connector", True)
    if caption_proj_before_connector:
        video_inner_dim = tf_config.get("num_attention_heads", 32) * tf_config.get("attention_head_dim", 128)
        feature_extractor = GemmaFeaturesExtractorProjLinear(out_dim=video_inner_dim, bias=True, use_video_key=True)
    else:
        feature_extractor = GemmaFeaturesExtractorProjLinear()

    connector_head_dim = tf_config.get("connector_attention_head_dim", 128)
    connector_heads = tf_config.get("connector_num_attention_heads", 32)
    connector = Embeddings1DConnector(
        attention_head_dim=connector_head_dim,
        num_attention_heads=connector_heads,
        num_layers=tf_config.get("connector_num_layers", 8),
        positional_embedding_max_pos=tf_config.get("connector_positional_embedding_max_pos", [1]),
        rope_type=LTXRopeType(tf_config.get("rope_type", "interleaved")),
        apply_gated_attention=tf_config.get("connector_apply_gated_attention", True),
    )

    tokenizer = LTXVGemmaTokenizer(gemma_root)
    # ALAYA_GEMMA_DEVICE_MAP (e.g. "auto") spreads the 24GB text encoder over several
    # devices via accelerate, for machines where it does not fit on one GPU.
    _device_map = os.environ.get("ALAYA_GEMMA_DEVICE_MAP", "").strip()
    if _device_map:
        # ALAYA_GEMMA_MAX_MEMORY, e.g. "0=14GiB,1=14GiB", forces an even split;
        # device_map="auto" alone fills the first GPU before spilling to the next.
        _max_memory = None
        _raw_mm = os.environ.get("ALAYA_GEMMA_MAX_MEMORY", "").strip()
        if _raw_mm:
            _max_memory = {}
            for item in _raw_mm.split(","):
                k, _, v = item.partition("=")
                k = k.strip()
                _max_memory[int(k) if k.isdigit() else k] = v.strip()
        gemma = Gemma3ForConditionalGeneration.from_pretrained(
            gemma_root,
            local_files_only=True,
            dtype=dtype,
            torch_dtype=dtype,
            device_map=_device_map,
            max_memory=_max_memory,
        ).eval()
    else:
        gemma = Gemma3ForConditionalGeneration.from_pretrained(
            gemma_root,
            local_files_only=True,
            dtype=dtype,
            torch_dtype=dtype,
        ).to(device).eval()
    text_encoder = AVGemmaTextEncoderModel(
        feature_extractor,
        connector,
        None,
        tokenizer=tokenizer,
        model=gemma,
        dtype=dtype,
        use_v2_norm=caption_proj_before_connector,
        gemma_embedding_dim=3840,
    )

    if state_dict is None:  # da3: shared merged state when provided; else read here (vigeo path)
        # Only the two prefixes consumed below, not the whole release file.
        _wanted = ("text_embedding_projection.", "model.diffusion_model.video_embeddings_connector.")
        state_dict = {}
        with safetensors.safe_open(checkpoint_path, framework="pt") as handle:
            for raw_key in handle.keys():
                if raw_key.startswith(_wanted):
                    state_dict[raw_key] = handle.get_tensor(raw_key)
    fe = {
        key.removeprefix("text_embedding_projection."): value
        for key, value in state_dict.items()
        if key.startswith("text_embedding_projection.")
    }
    if fe:
        text_encoder.feature_extractor_linear.load_state_dict(fe, strict=False)
    ec = {
        key.replace("model.diffusion_model.video_embeddings_connector.", ""): value
        for key, value in state_dict.items()
        if "video_embeddings_connector" in key
    }
    if ec:
        text_encoder.embeddings_connector.load_state_dict(ec, strict=False)

    if _device_map:
        # Gemma is already placed by accelerate across several devices; moving the
        # whole wrapper would undo that. Only the two small LTX heads need placing,
        # on the device Gemma's own output lands on.
        _head_device = getattr(gemma, "device", device)
        text_encoder.feature_extractor_linear.to(_head_device)
        text_encoder.embeddings_connector.to(_head_device)
        # The encoder stacks *every* layer's hidden state; with the layers split over
        # devices those tensors come back on whichever device produced them, so bring
        # them together before torch.stack sees them.
        _inner_forward = gemma.forward

        def _forward_aligned(*a, **kw):
            out = _inner_forward(*a, **kw)
            hs = getattr(out, "hidden_states", None)
            if hs is not None:
                out.hidden_states = tuple(h.to(_head_device) for h in hs)
            return out

        gemma.forward = _forward_aligned
        text_encoder.eval()
    else:
        text_encoder.to(device).eval()
    for param in text_encoder.parameters():
        param.requires_grad_(False)

    def encode_text(encoder: nn.Module, prompts: list[str]):
        outputs = []
        with torch.no_grad():
            for prompt in prompts:
                outputs.append(encoder(prompt).video_encoding)
        return outputs

    return text_encoder, encode_text
