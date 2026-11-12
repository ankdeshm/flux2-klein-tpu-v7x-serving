#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Stages FLUX.2 Klein 4B & 9B checkpoints into `/dev/shm` Diffusers layout for TPU v7x serving.

Assembles:
  1. `/dev/shm/FLUX.2-klein-4B` from pre-staged `black-forest-labs/FLUX.2-klein-4B`
     (skipping the duplicate root single-file `.safetensors` to save 7.3 GB RAM).
  2. `/dev/shm/Flux-2-Klein-9B-Diffusers` (`BF16` Diffusers `Flux2KleinPipeline` layout) by:
       - Dequantizing calibrated `F8_E4M3` weights (`w_bf16 = w_fp8 * weight_scale`) while
         preserving native `BF16` image-KV / norm tensors from
         `black-forest-labs/FLUX.2-klein-9b-kv-fp8/flux-2-klein-9b-kv-fp8.safetensors`
         and converting via `diffusers.loaders.single_file_utils.convert_flux2_transformer_checkpoint_to_diffusers`.
       - Pairing with `Comfy-Org/Flux-2-Klein-9B/split_files/text_encoders/qwen_3_8b.safetensors`
         (`Qwen3ForCausalLM`, `hidden_size=4096`, `36` layers -> `joint_attention_dim=12288`)
         and `AutoencoderKLFlux2` (`32` latent channels) + `Qwen2TokenizerFast` + `FlowMatchEulerDiscreteScheduler`.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
import shutil
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def fast_copy_tree(src_dir: Path, dst_dir: Path, skip_names: set[str] | None = None) -> float:
    skip = skip_names or set()
    files = [
        p
        for p in src_dir.rglob("*")
        if p.is_file()
        and "/.cache/" not in p.as_posix()
        and p.name not in skip
        and not p.name.endswith((".jpg", ".png", ".webp", ".gif"))
    ]
    t0 = time.perf_counter()

    def _cp(src: Path) -> int:
        rel = src.relative_to(src_dir)
        dst = dst_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copyfile(src, dst)
        return dst.stat().st_size

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(max(len(files), 1), 8)) as pool:
        total_bytes = sum(pool.map(_cp, files))
    dt = max(time.perf_counter() - t0, 1e-3)
    print(f"[stage] Staged {src_dir} -> {dst_dir} ({total_bytes / 1e9:.2f} GB in {dt:.1f}s)", flush=True)
    return total_bytes / 1e9


def build_flux2_klein_9b_diffusers(models_root: Path, dst_dir: Path) -> Path:
    marker = dst_dir / ".ready_verified"
    if marker.exists():
        print(f"[stage] {dst_dir} already verified ready.", flush=True)
        return dst_dir

    t0 = time.perf_counter()
    src_4b = models_root / "black-forest-labs" / "FLUX.2-klein-4B"
    src_9b_kv_sf = models_root / "black-forest-labs" / "FLUX.2-klein-9b-kv-fp8" / "flux-2-klein-9b-kv-fp8.safetensors"
    src_qwen8b_sf = models_root / "Comfy-Org" / "Flux-2-Klein-9B" / "split_files" / "text_encoders" / "qwen_3_8b.safetensors"

    if not src_4b.exists():
        raise FileNotFoundError(f"Missing {src_4b}")
    if not src_9b_kv_sf.exists():
        raise FileNotFoundError(f"Missing {src_9b_kv_sf}")
    if not src_qwen8b_sf.exists():
        raise FileNotFoundError(f"Missing {src_qwen8b_sf}")

    dst_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy model_index.json, scheduler, tokenizer, and vae from FLUX.2-klein-4B
    shutil.copyfile(src_4b / "model_index.json", dst_dir / "model_index.json")
    for sub in ("scheduler", "tokenizer", "vae"):
        fast_copy_tree(src_4b / sub, dst_dir / sub)

    # 2. Stage Qwen3-8B text_encoder (`hidden_size=4096`, `intermediate_size=12288`, `36` layers)
    te_dir = dst_dir / "text_encoder"
    te_dir.mkdir(parents=True, exist_ok=True)
    te_sf_dst = te_dir / "model.safetensors"
    if not te_sf_dst.exists() or te_sf_dst.stat().st_size != src_qwen8b_sf.stat().st_size:
        t_te = time.perf_counter()
        shutil.copyfile(src_qwen8b_sf, te_sf_dst)
        print(f"[stage] Copied Qwen3-8B text_encoder ({te_sf_dst.stat().st_size / 1e9:.2f} GB) in {time.perf_counter()-t_te:.1f}s", flush=True)

    with safe_open(str(te_sf_dst), framework="pt", device="cpu") as qf:
        q_keys = set(qf.keys())
        has_lm_head = "lm_head.weight" in q_keys
        emb_shape = list(qf.get_slice("model.embed_tokens.weight").get_shape())
        up_shape = list(qf.get_slice("model.layers.0.mlp.up_proj.weight").get_shape())
        q_shape = list(qf.get_slice("model.layers.0.self_attn.q_proj.weight").get_shape())
        k_shape = list(qf.get_slice("model.layers.0.self_attn.k_proj.weight").get_shape())
        num_te_layers = len({int(k.split(".")[2]) for k in q_keys if k.startswith("model.layers.")})

    base_te_cfg = json.loads((src_4b / "text_encoder" / "config.json").read_text())
    head_dim = int(base_te_cfg.get("head_dim", 128))
    base_te_cfg.update(
        {
            "vocab_size": emb_shape[0],
            "hidden_size": emb_shape[1],
            "intermediate_size": up_shape[0],
            "num_hidden_layers": num_te_layers,
            "max_window_layers": num_te_layers,
            "layer_types": ["full_attention"] * num_te_layers,
            "num_attention_heads": q_shape[0] // head_dim,
            "num_key_value_heads": k_shape[0] // head_dim,
            "tie_word_embeddings": not has_lm_head,
            "dtype": "bfloat16",
        }
    )
    (te_dir / "config.json").write_text(json.dumps(base_te_cfg, indent=2) + "\n")
    if (src_4b / "text_encoder" / "generation_config.json").exists():
        shutil.copyfile(src_4b / "text_encoder" / "generation_config.json", te_dir / "generation_config.json")

    # 3. Convert FLUX.2-klein-9B transformer checkpoint to Diffusers BF16 layout
    from diffusers.loaders.single_file_utils import convert_flux2_transformer_checkpoint_to_diffusers

    tf_dir = dst_dir / "transformer"
    tf_dir.mkdir(parents=True, exist_ok=True)
    shm_raw_sf = Path("/dev/shm/_tmp_9b_kv_fp8.safetensors")
    shutil.copyfile(src_9b_kv_sf, shm_raw_sf)

    raw_bfl_sd: dict[str, torch.Tensor] = {}
    with safe_open(str(shm_raw_sf), framework="pt", device="cpu") as sf:
        all_keys = list(sf.keys())
        num_double = len({int(k.split(".")[1]) for k in all_keys if k.startswith("double_blocks.")})
        num_single = len({int(k.split(".")[1]) for k in all_keys if k.startswith("single_blocks.")})
        has_guidance = any(k.startswith("guidance_in.") for k in all_keys)
        for k in all_keys:
            if k.endswith((".input_scale", ".weight_scale")):
                continue
            t = sf.get_tensor(k)
            if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                scale_key = k[: -len(".weight")] + ".weight_scale"
                if scale_key in all_keys:
                    w_scale = sf.get_tensor(scale_key).to(torch.float32)
                    t = (t.to(torch.float32) * w_scale).to(torch.bfloat16)
                else:
                    t = t.to(torch.bfloat16)
            elif t.dtype != torch.bfloat16:
                t = t.to(torch.bfloat16)
            raw_bfl_sd[k] = t

    shm_raw_sf.unlink(missing_ok=True)

    txt_in_shape = list(raw_bfl_sd["txt_in.weight"].shape)  # [4096, 12288]
    img_in_shape = list(raw_bfl_sd["img_in.weight"].shape)  # [4096, 128]
    hidden_dim = img_in_shape[0]
    in_channels = img_in_shape[1]
    joint_dim = txt_in_shape[1]
    attn_head_dim = 128
    num_heads = hidden_dim // attn_head_dim

    diffusers_sd = convert_flux2_transformer_checkpoint_to_diffusers(raw_bfl_sd)
    del raw_bfl_sd
    save_file(diffusers_sd, str(tf_dir / "diffusion_pytorch_model.safetensors"))

    tf_cfg = json.loads((src_4b / "transformer" / "config.json").read_text())
    tf_cfg.update(
        {
            "_name_or_path": "Flux-2-Klein-9B/transformer",
            "attention_head_dim": attn_head_dim,
            "guidance_embeds": has_guidance,
            "in_channels": in_channels,
            "joint_attention_dim": joint_dim,
            "mlp_ratio": 3.0,
            "num_attention_heads": num_heads,
            "num_layers": num_double,
            "num_single_layers": num_single,
            "patch_size": 1,
        }
    )
    (tf_dir / "config.json").write_text(json.dumps(tf_cfg, indent=2) + "\n")
    marker.write_text(f"ready in {time.perf_counter()-t0:.1f}s\n")
    print(
        f"[stage] Assembled Flux-2-Klein-9B Diffusers checkpoint at {dst_dir} "
        f"(double={num_double}, single={num_single}, hidden={hidden_dim}, joint={joint_dim}) "
        f"in {time.perf_counter()-t0:.1f}s",
        flush=True,
    )
    return dst_dir


def stage_flux2_klein_4b_shm(models_root: Path, dst_dir: Path) -> Path:
    marker = dst_dir / ".ready_verified"
    if marker.exists():
        print(f"[stage] {dst_dir} already verified ready.", flush=True)
        return dst_dir
    src_4b = models_root / "black-forest-labs" / "FLUX.2-klein-4B"
    fast_copy_tree(src_4b, dst_dir, skip_names={"flux-2-klein-4b.safetensors"})
    marker.write_text("ready\n")
    return dst_dir


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare FLUX.2 Klein 4B & 9B checkpoints in /dev/shm")
    p.add_argument("--models-root", default="/models/models")
    p.add_argument("--shm-root", default="/dev/shm/flux2_models")
    p.add_argument("--skip-4b", action="store_true")
    p.add_argument("--skip-9b", action="store_true")
    args = p.parse_args()

    models_root = Path(args.models_root)
    shm_root = Path(args.shm_root)
    shm_root.mkdir(parents=True, exist_ok=True)
    if not args.skip_4b:
        stage_flux2_klein_4b_shm(models_root, shm_root / "FLUX.2-klein-4B")
    if not args.skip_9b:
        build_flux2_klein_9b_diffusers(models_root, shm_root / "Flux-2-Klein-9B")


if __name__ == "__main__":
    main()
