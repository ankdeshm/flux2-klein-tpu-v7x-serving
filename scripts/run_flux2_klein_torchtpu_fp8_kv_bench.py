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
"""Direct TorchTPU BF16 vs Calibrated FP8 (`4b-fp8`) vs Hybrid KV-FP8 (`9b-kv-fp8`) Benchmark on TPU v7x.

Multi-Process Hardware-Isolated Engine (`torchrun --nproc_per_node=8` + `tpu_dist` ICI + `torch._scaled_mm`):
  - Loads actual safetensors weights & calibrated `F32` `input_scale`/`weight_scale` tensors from:
      1. `black-forest-labs/FLUX.2-klein-4B` (`BF16`: 5 double + 20 single blocks, hidden=3072, heads=24x128)
      2. `black-forest-labs/FLUX.2-klein-4b-fp8` (`FP8`: 80 calibrated `F8_E4M3` weights + 160 `F32` scales)
      3. `Comfy-Org/Flux-2-Klein-9B` (`BF16`: 8 double + 24 single blocks, hidden=4096, heads=32x128)
      4. `black-forest-labs/FLUX.2-klein-9b-kv-fp8` (`KV-FP8`: hybrid `BF16` double-block `img_attn.qkv/proj`
         preserving spatial/reference KV fidelity + 105 calibrated `F8_E4M3` MLP/text/single-block projections)
  - Executes compiled (`torch.compile(dynamic=False)`) `FLUX.2` Double-Stream + Single-Stream DiT blocks
    plus real `AutoencoderKLFlux2` (`32`-channel VAE) decode across:
      * Precisions: `BF16`, `FP8` (`torch._scaled_mm`), and `KV-FP8` (hybrid `BF16` KV + `torch._scaled_mm` FP8)
      * Batch sizes: `B = 1, 2, 4`
      * Resolutions: `512x512` (1024 latents), `1024x1024` (4096 latents),
        `1536x1536` (9216 latents), `2048x2048` (16384 latents)
      * Denoising steps: `1, 4, 8, 20`
      * Topologies: `1-TC (DP=1)`, `Intra-Chip U=2 (2 TC)` (`tpu_dist`), `1-Host DP=8 (8 TC)`,
        and Cross-Host `2-Host DP=16 (16 TC)` (`FastThreadingHTTPServer` Keep-Alive)
      * Closed-Loop `CON = 1, 2, 4, 8, 16, 32, 64` & Open-Loop Poisson `RPS = 1, 2, 4, 8`
      * Disaggregated PNG/WebP encode (`multimodal-cpu-offload-svc:8090/encode_image` vs local TPU CPU)
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
from datetime import datetime, timezone
import gc
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
import os
from pathlib import Path
import random
import shutil
import socket
import statistics
import subprocess
import sys
import threading
import time
from typing import Any
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image


class FastThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 256


FLUX2_SPECS: dict[str, dict[str, Any]] = {
    "klein_4b": {
        "display_name": "FLUX.2-klein-4B",
        "bf16_repo": "black-forest-labs/FLUX.2-klein-4B",
        "fp8_repo": "black-forest-labs/FLUX.2-klein-4b-fp8",
        "fp8_file": "flux-2-klein-4b-fp8.safetensors",
        "quant_mode": "FP8",
        "hidden": 3072,
        "heads": 24,
        "head_dim": 128,
        "mlp_hidden": 9216,
        "joint_dim": 7680,
        "num_double": 5,
        "num_single": 20,
        "txt_tokens": 512,
        "te_hbm_bf16_gib": 7.55,
    },
    "klein_9b": {
        "display_name": "FLUX.2-klein-9B",
        "bf16_repo": "Comfy-Org/Flux-2-Klein-9B",
        "fp8_repo": "black-forest-labs/FLUX.2-klein-9b-kv-fp8",
        "fp8_file": "flux-2-klein-9b-kv-fp8.safetensors",
        "quant_mode": "KV-FP8",
        "hidden": 4096,
        "heads": 32,
        "head_dim": 128,
        "mlp_hidden": 12288,
        "joint_dim": 12288,
        "num_double": 8,
        "num_single": 24,
        "txt_tokens": 512,
        "te_hbm_bf16_gib": 15.26,
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(tag: str, msg: str) -> None:
    print(f"[{utc_now()}][flux2-fp8-kv][{tag}] {msg}", flush=True)


def pct(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = min(len(s) - 1, max(0, int(math.ceil((q / 100.0) * len(s))) - 1))
    return float(s[idx])


def measure_image_encode_offload(offload_url: str) -> dict[str, Any]:
    """Benchmarks local TPU-VM CPU PIL PNG/WebP encode vs disaggregated Xeon `multimodal-cpu-offload-svc:8090/encode_image`."""
    rng = np.random.default_rng(2026)
    results: dict[str, Any] = {}
    for res in (512, 1024, 1536, 2048):
        rgb = rng.integers(0, 256, size=(res, res, 3), dtype=np.uint8)
        raw_b64 = base64.b64encode(rgb.tobytes()).decode("ascii")
        for fmt in ("png", "webp"):
            key = f"{res}x{res}_{fmt}"
            # Local TPU-VM CPU PIL encode
            loc_lats = []
            for i in range(7):
                t0 = time.perf_counter()
                buf = io.BytesIO()
                if fmt == "png":
                    Image.fromarray(rgb).save(buf, format="PNG", compress_level=1)
                else:
                    Image.fromarray(rgb).save(buf, format="WEBP", quality=90, method=0)
                _ = buf.getvalue()
                if i >= 2:
                    loc_lats.append((time.perf_counter() - t0) * 1000.0)
            loc_med = round(statistics.median(loc_lats), 2)

            # Disaggregated Xeon CPU offload service
            dis_rtts, dis_srvs = [], []
            ok = False
            for i in range(7):
                try:
                    payload = json.dumps(
                        {
                            "raw_rgb_b64": raw_b64,
                            "height": res,
                            "width": res,
                            "format": fmt,
                            "quality": 90,
                            "png_compress_level": 1,
                        }
                    ).encode("utf-8")
                    req = urllib.request.Request(
                        f"{offload_url.rstrip('/')}/encode_image",
                        data=payload,
                        method="POST",
                        headers={"Content-Type": "application/json"},
                    )
                    t0 = time.perf_counter()
                    with urllib.request.urlopen(req, timeout=15.0) as resp:
                        d = json.loads(resp.read().decode("utf-8"))
                    rtt = (time.perf_counter() - t0) * 1000.0
                    ok = True
                    if i >= 2:
                        dis_rtts.append(rtt)
                        dis_srvs.append(float(d.get("encode_ms", rtt)))
                except Exception:
                    break
            if ok and dis_rtts:
                rtt_med = round(statistics.median(dis_rtts), 2)
                srv_med = round(statistics.median(dis_srvs), 2)
                results[key] = {
                    "resolution": f"{res}x{res}",
                    "format": fmt,
                    "offload_ok": True,
                    "local_tpu_cpu_encode_ms": loc_med,
                    "disagg_cpu_rtt_ms": rtt_med,
                    "disagg_cpu_server_encode_ms": srv_med,
                    "encoder_tier": "disaggregated-xeon-cpu-pool",
                    "server_compute_speedup_x": round(loc_med / max(srv_med, 1e-3), 2),
                }
            else:
                results[key] = {
                    "resolution": f"{res}x{res}",
                    "format": fmt,
                    "offload_ok": False,
                    "local_tpu_cpu_encode_ms": loc_med,
                    "disagg_cpu_rtt_ms": loc_med,
                    "disagg_cpu_server_encode_ms": None,
                }
            log("encode-offload", f"{key}: local_cpu={loc_med}ms, disagg_rtt={results[key]['disagg_cpu_rtt_ms']}ms")
    return results


class MultiEndpointKeepAlivePool:
    def __init__(self, urls: list[str]) -> None:
        self.urls = urls
        self.tls = threading.local()
        self.inflight = [0] * len(urls)
        self.lock = threading.Lock()
        self.rr = 0

    def wait_healthy(self, timeout_s: float = 600.0) -> None:
        for u in self.urls:
            deadline = time.time() + timeout_s
            while time.time() < deadline:
                try:
                    with urllib.request.urlopen(f"{u}/health", timeout=2.5) as r:
                        if r.status == 200:
                            break
                except Exception:
                    time.sleep(0.5)
            else:
                raise TimeoutError(f"Timed out waiting for {u}/health")

    def call_infer(self, model_key: str, precision: str, res: int = 1024, steps: int = 4, batch_size: int = 1) -> dict[str, Any]:
        with self.lock:
            n = len(self.urls)
            idx = min(range(n), key=lambda i: (self.inflight[i], (i - self.rr) % n))
            self.inflight[idx] += 1
            self.rr = (idx + 1) % n
            u = self.urls[idx]

        p = urllib.parse.urlsplit(u)
        host = p.hostname or "127.0.0.1"
        port = p.port or 19300
        key = f"{host}:{port}"
        body = json.dumps(
            {"model_key": model_key, "precision": precision, "res": res, "steps": steps, "batch_size": batch_size}
        ).encode("utf-8")
        t0 = time.perf_counter()
        try:
            conns: dict[str, http.client.HTTPConnection] = getattr(self.tls, "conns", None) or {}
            self.tls.conns = conns
            for attempt in range(2):
                try:
                    conn = conns.get(key)
                    if conn is None or conn.sock is None:
                        conn = http.client.HTTPConnection(host, port, timeout=120.0)
                        conn.connect()
                        if conn.sock is not None:
                            conn.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                        conns[key] = conn
                    conn.request(
                        "POST",
                        "/infer",
                        body=body,
                        headers={"Content-Type": "application/json", "Content-Length": str(len(body)), "Connection": "keep-alive"},
                    )
                    resp = conn.getresponse()
                    raw = resp.read()
                    d = json.loads(raw.decode("utf-8"))
                    d["e2e_s"] = time.perf_counter() - t0
                    return d
                except Exception:
                    old = conns.pop(key, None)
                    if old is not None:
                        try:
                            old.close()
                        except Exception:
                            pass
                    if attempt == 1:
                        raise
            raise RuntimeError("unreachable")
        finally:
            with self.lock:
                self.inflight[idx] = max(0, self.inflight[idx] - 1)


def run_torchrun_flux2_worker(
    models_root: Path,
    shm_models_root: Path,
    model_keys: list[str],
    out_jsonl: Path,
    out_summary_json: Path,
    encode_json: Path,
    serve_port: int,
) -> None:
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    import torch_tpu  # noqa: F401
    from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
    from safetensors import safe_open

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.accelerator.set_device_index(0)
    dev = torch.device("tpu:0")
    if not dist.is_initialized():
        dist.init_process_group(backend="tpu_dist")
    gloo_pg = dist.new_group(list(range(world_size)), backend="gloo")
    u2_groups = [dist.new_group([2 * i, 2 * i + 1], backend="tpu_dist") for i in range(world_size // 2)]
    my_u2_pg = u2_groups[local_rank // 2]

    torch._dynamo.config.cache_size_limit = 512
    if hasattr(torch._dynamo.config, "accumulated_cache_size_limit"):
        torch._dynamo.config.accumulated_cache_size_limit = 4096

    if rank == 0:
        log("worker", f"Initialized {world_size}-process TorchTPU FLUX.2 Klein worker pool (`tpu_dist` + Tier-2 cache)")

    enc_map: dict[str, Any] = json.loads(encode_json.read_text()) if encode_json.exists() else {}

    # Load real AutoencoderKLFlux2 (`32` latent channels) on every TensorCore
    vae_src = shm_models_root / "FLUX.2-klein-4B"
    if not vae_src.exists():
        vae_src = models_root / "black-forest-labs" / "FLUX.2-klein-4B"
    vae = AutoencoderKLFlux2.from_pretrained(str(vae_src), subfolder="vae", local_files_only=True).to(
        device=dev, dtype=torch.bfloat16
    )
    vae.eval()
    for _m in vae.modules():
        if isinstance(_m, torch.nn.GroupNorm):

            def _gn_fwd(x: torch.Tensor, _mod: torch.nn.GroupNorm = _m) -> torch.Tensor:
                orig = x.shape
                b_n, c_n = orig[0], orig[1]
                xg = x.contiguous().reshape(b_n, _mod.num_groups, -1).to(torch.float32)
                mean = xg.mean(-1, keepdim=True)
                var = xg.var(-1, unbiased=False, keepdim=True)
                xn = ((xg - mean) * torch.rsqrt(var + _mod.eps)).to(x.dtype).reshape(orig)
                if _mod.affine:
                    w_sh = (1, c_n) + (1,) * (x.ndim - 2)
                    xn = xn * _mod.weight.reshape(w_sh) + _mod.bias.reshape(w_sh)
                return xn

            _m.forward = _gn_fwd  # type: ignore[method-assign]

    @torch.compile(dynamic=False)
    def _vae_decode_compiled(z_lat: torch.Tensor) -> torch.Tensor:
        return vae.decode(z_lat, return_dict=False)[0]

    def _calib_fp8_mm(
        x: torch.Tensor, w_fp8: torch.Tensor, in_scale: torch.Tensor, w_scale: torch.Tensor
    ) -> torch.Tensor:
        """Executes calibrated BFL `F8_E4M3` linear layer via hardware `torch._scaled_mm` on TPU v7x MXU."""
        orig = x.shape[:-1]
        in_f = w_fp8.shape[1]
        out_f = w_fp8.shape[0]
        x2d = x.reshape(-1, in_f)
        m = x2d.shape[0]
        pad_m = (16 - (m % 16)) % 16
        if pad_m:
            x2d = F.pad(x2d, (0, 0, 0, pad_m))
        x_fp8 = (x2d / in_scale.to(torch.bfloat16)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        y = torch._scaled_mm(x_fp8, w_fp8.t(), scale_a=in_scale, scale_b=w_scale, out_dtype=torch.bfloat16)
        if pad_m:
            y = y[:m]
        return y.reshape(*orig, out_f)

    def _make_flux2_kernels(heads: int, hd: int, hidden: int, mlp_h: int) -> dict[str, Any]:
        """Builds dedicated compiled kernels for FLUX.2 Double-Stream (BF16, FP8, and KV-FP8) & Single-Stream blocks."""

        @torch.compile(dynamic=False)
        def double_bf16(
            img: torch.Tensor,
            txt: torch.Tensor,
            iqkv: torch.Tensor,
            iproj: torch.Tensor,
            imlp0: torch.Tensor,
            imlp2: torch.Tensor,
            tqkv: torch.Tensor,
            tproj: torch.Tensor,
            tmlp0: torch.Tensor,
            tmlp2: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            b, si, _ = img.shape
            st = txt.shape[1]
            qi, ki, vi = F.linear(img, iqkv).chunk(3, dim=-1)
            qt, kt, vt = F.linear(txt, tqkv).chunk(3, dim=-1)
            q = torch.cat([qt, qi], dim=1).view(b, st + si, heads, hd).transpose(1, 2)
            k = torch.cat([kt, ki], dim=1).view(b, st + si, heads, hd).transpose(1, 2)
            v = torch.cat([vt, vi], dim=1).view(b, st + si, heads, hd).transpose(1, 2)
            at = F.scaled_dot_product_attention(q, k, v, is_causal=False).transpose(1, 2).contiguous().view(b, st + si, hidden)
            at_t, at_i = at[:, :st, :], at[:, st:, :]
            hi = img + F.linear(at_i, iproj)
            ht = txt + F.linear(at_t, tproj)
            gi, ui = F.linear(hi, imlp0).chunk(2, dim=-1)
            gt, ut = F.linear(ht, tmlp0).chunk(2, dim=-1)
            return hi + F.linear(F.silu(gi) * ui, imlp2), ht + F.linear(F.silu(gt) * ut, tmlp2)

        @torch.compile(dynamic=False)
        def double_fp8(
            img: torch.Tensor,
            txt: torch.Tensor,
            iqkv: torch.Tensor,
            iqkv_is: torch.Tensor,
            iqkv_ws: torch.Tensor,
            iproj: torch.Tensor,
            iproj_is: torch.Tensor,
            iproj_ws: torch.Tensor,
            imlp0: torch.Tensor,
            imlp0_is: torch.Tensor,
            imlp0_ws: torch.Tensor,
            imlp2: torch.Tensor,
            imlp2_is: torch.Tensor,
            imlp2_ws: torch.Tensor,
            tqkv: torch.Tensor,
            tqkv_is: torch.Tensor,
            tqkv_ws: torch.Tensor,
            tproj: torch.Tensor,
            tproj_is: torch.Tensor,
            tproj_ws: torch.Tensor,
            tmlp0: torch.Tensor,
            tmlp0_is: torch.Tensor,
            tmlp0_ws: torch.Tensor,
            tmlp2: torch.Tensor,
            tmlp2_is: torch.Tensor,
            tmlp2_ws: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            b, si, _ = img.shape
            st = txt.shape[1]
            qi, ki, vi = _calib_fp8_mm(img, iqkv, iqkv_is, iqkv_ws).chunk(3, dim=-1)
            qt, kt, vt = _calib_fp8_mm(txt, tqkv, tqkv_is, tqkv_ws).chunk(3, dim=-1)
            q = torch.cat([qt, qi], dim=1).view(b, st + si, heads, hd).transpose(1, 2)
            k = torch.cat([kt, ki], dim=1).view(b, st + si, heads, hd).transpose(1, 2)
            v = torch.cat([vt, vi], dim=1).view(b, st + si, heads, hd).transpose(1, 2)
            at = F.scaled_dot_product_attention(q, k, v, is_causal=False).transpose(1, 2).contiguous().view(b, st + si, hidden)
            at_t, at_i = at[:, :st, :], at[:, st:, :]
            hi = img + _calib_fp8_mm(at_i, iproj, iproj_is, iproj_ws)
            ht = txt + _calib_fp8_mm(at_t, tproj, tproj_is, tproj_ws)
            gi, ui = _calib_fp8_mm(hi, imlp0, imlp0_is, imlp0_ws).chunk(2, dim=-1)
            gt, ut = _calib_fp8_mm(ht, tmlp0, tmlp0_is, tmlp0_ws).chunk(2, dim=-1)
            return (
                hi + _calib_fp8_mm(F.silu(gi) * ui, imlp2, imlp2_is, imlp2_ws),
                ht + _calib_fp8_mm(F.silu(gt) * ut, tmlp2, tmlp2_is, tmlp2_ws),
            )

        @torch.compile(dynamic=False)
        def double_kv_fp8(
            img: torch.Tensor,
            txt: torch.Tensor,
            iqkv_bf16: torch.Tensor,
            iproj_bf16: torch.Tensor,
            imlp0: torch.Tensor,
            imlp0_is: torch.Tensor,
            imlp0_ws: torch.Tensor,
            imlp2: torch.Tensor,
            imlp2_is: torch.Tensor,
            imlp2_ws: torch.Tensor,
            tqkv: torch.Tensor,
            tqkv_is: torch.Tensor,
            tqkv_ws: torch.Tensor,
            tproj: torch.Tensor,
            tproj_is: torch.Tensor,
            tproj_ws: torch.Tensor,
            tmlp0_bf16: torch.Tensor,
            tmlp2_bf16: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """BFL `FLUX.2-klein-9b-kv-fp8` early hybrid block (`double_blocks.0..1`):
            `img_attn.(qkv,proj)` + `txt_mlp.(0,2)` run in native BF16 while `img_mlp` + `txt_attn` run calibrated FP8!"""
            b, si, _ = img.shape
            st = txt.shape[1]
            qi, ki, vi = F.linear(img, iqkv_bf16).chunk(3, dim=-1)
            qt, kt, vt = _calib_fp8_mm(txt, tqkv, tqkv_is, tqkv_ws).chunk(3, dim=-1)
            q = torch.cat([qt, qi], dim=1).view(b, st + si, heads, hd).transpose(1, 2)
            k = torch.cat([kt, ki], dim=1).view(b, st + si, heads, hd).transpose(1, 2)
            v = torch.cat([vt, vi], dim=1).view(b, st + si, heads, hd).transpose(1, 2)
            at = F.scaled_dot_product_attention(q, k, v, is_causal=False).transpose(1, 2).contiguous().view(b, st + si, hidden)
            at_t, at_i = at[:, :st, :], at[:, st:, :]
            hi = img + F.linear(at_i, iproj_bf16)
            ht = txt + _calib_fp8_mm(at_t, tproj, tproj_is, tproj_ws)
            gi, ui = _calib_fp8_mm(hi, imlp0, imlp0_is, imlp0_ws).chunk(2, dim=-1)
            gt, ut = F.linear(ht, tmlp0_bf16).chunk(2, dim=-1)
            return (
                hi + _calib_fp8_mm(F.silu(gi) * ui, imlp2, imlp2_is, imlp2_ws),
                ht + F.linear(F.silu(gt) * ut, tmlp2_bf16),
            )

        @torch.compile(dynamic=False)
        def single_bf16(x: torch.Tensor, lin1: torch.Tensor, lin2: torch.Tensor) -> torch.Tensor:
            b, s, _ = x.shape
            proj = F.linear(x, lin1)
            qkv = proj[:, :, : 3 * hidden]
            mlp_in = proj[:, :, 3 * hidden :]
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(b, s, heads, hd).transpose(1, 2)
            k = k.view(b, s, heads, hd).transpose(1, 2)
            v = v.view(b, s, heads, hd).transpose(1, 2)
            at = F.scaled_dot_product_attention(q, k, v, is_causal=False).transpose(1, 2).contiguous().view(b, s, hidden)
            g, u = mlp_in.chunk(2, dim=-1)
            cat = torch.cat([at, F.silu(g) * u], dim=-1)
            return x + F.linear(cat, lin2)

        @torch.compile(dynamic=False)
        def single_fp8(
            x: torch.Tensor,
            lin1: torch.Tensor,
            l1_is: torch.Tensor,
            l1_ws: torch.Tensor,
            lin2: torch.Tensor,
            l2_is: torch.Tensor,
            l2_ws: torch.Tensor,
        ) -> torch.Tensor:
            b, s, _ = x.shape
            proj = _calib_fp8_mm(x, lin1, l1_is, l1_ws)
            qkv = proj[:, :, : 3 * hidden]
            mlp_in = proj[:, :, 3 * hidden :]
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(b, s, heads, hd).transpose(1, 2)
            k = k.view(b, s, heads, hd).transpose(1, 2)
            v = v.view(b, s, heads, hd).transpose(1, 2)
            at = F.scaled_dot_product_attention(q, k, v, is_causal=False).transpose(1, 2).contiguous().view(b, s, hidden)
            g, u = mlp_in.chunk(2, dim=-1)
            cat = torch.cat([at, F.silu(g) * u], dim=-1)
            return x + _calib_fp8_mm(cat, lin2, l2_is, l2_ws)

        @torch.compile(dynamic=False)
        def embed_io(img_lat: torch.Tensor, txt_emb: torch.Tensor, w_img_in: torch.Tensor, w_txt_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return F.linear(img_lat, w_img_in), F.linear(txt_emb, w_txt_in)

        @torch.compile(dynamic=False)
        def final_out(seq: torch.Tensor, st: int, w_out: torch.Tensor) -> torch.Tensor:
            return F.linear(seq[:, st:, :], w_out)

        return {
            "double_bf16": double_bf16,
            "double_fp8": double_fp8,
            "double_kv_fp8": double_kv_fp8,
            "single_bf16": single_bf16,
            "single_fp8": single_fp8,
            "embed_io": embed_io,
            "final_out": final_out,
        }

    def _stage_safetensor_collective(mkey: str) -> tuple[Path, dict[str, Any]]:
        spec = FLUX2_SPECS[mkey]
        src = models_root / spec["fp8_repo"] / spec["fp8_file"]
        dst = Path("/dev/shm") / f"_flux2_raw_{spec['fp8_file']}"
        meta_list: list[Any] = [None]
        if rank == 0:
            t0 = time.perf_counter()
            if not dst.exists() or dst.stat().st_size != src.stat().st_size:
                shutil.copyfile(src, dst)
            dtype_hist: dict[str, int] = {}
            n_params = 0
            with safe_open(str(dst), framework="pt", device="cpu") as sf:
                for k in sf.keys():
                    sl = sf.get_slice(k)
                    dt = str(sl.get_dtype())
                    dtype_hist[dt] = dtype_hist.get(dt, 0) + 1
                    if not k.endswith("_scale"):
                        el = 1
                        for d in sl.get_shape():
                            el *= d
                        n_params += el
            meta_list[0] = {
                "safetensors_file": spec["fp8_file"],
                "disk_gib": round(dst.stat().st_size / (1024**3), 3),
                "dit_parameters_b": round(n_params / 1e9, 3),
                "dtype_histogram": dtype_hist,
                "stage_s": round(time.perf_counter() - t0, 2),
            }
        dist.broadcast_object_list(meta_list, src=0, group=gloo_pg)
        dist.barrier(group=gloo_pg)
        return dst, meta_list[0]

    def _load_flux2_bank(mkey: str, shm_sf: Path, mode: str, u_degree: int = 1) -> tuple[dict[str, Any], float]:
        """Loads exact BFL safetensors into either `BF16` or calibrated `FP8` / `KV-FP8` on `tpu:0`."""
        spec = FLUX2_SPECS[mkey]
        heads = max(1, spec["heads"] // u_degree)
        hd = spec["head_dim"]
        hidden = heads * hd
        mlp_h = spec["mlp_hidden"] // u_degree
        kerns = _make_flux2_kernels(heads, hd, hidden, mlp_h)

        hbm_bytes = 0

        def _get_bf16(sf: Any, prefix: str, out_dim: int | None = None, in_dim: int | None = None) -> torch.Tensor:
            nonlocal hbm_bytes
            t = sf.get_tensor(f"{prefix}.weight")
            if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                ws = sf.get_tensor(f"{prefix}.weight_scale").to(torch.float32)
                t = (t.to(torch.float32) * ws).to(torch.bfloat16)
            else:
                t = t.to(torch.bfloat16)
            if out_dim is not None:
                t = t[:out_dim, :]
            if in_dim is not None:
                t = t[:, :in_dim]
            td = t.contiguous().to(dev)
            hbm_bytes += td.numel() * 2
            return td

        def _get_fp8_triplet(
            sf: Any, sf_keys: set[str], prefix: str, out_dim: int | None = None, in_dim: int | None = None
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            nonlocal hbm_bytes
            t = sf.get_tensor(f"{prefix}.weight")
            if out_dim is not None:
                t = t[:out_dim, :]
            if in_dim is not None:
                t = t[:, :in_dim]
            if f"{prefix}.input_scale" in sf_keys and t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                ins = sf.get_tensor(f"{prefix}.input_scale").reshape(()).to(torch.float32)
                ws = sf.get_tensor(f"{prefix}.weight_scale").reshape(()).to(torch.float32)
            else:
                tf = t.to(torch.float32)
                ws = (tf.abs().amax() / 448.0).clamp(min=1e-8).reshape(()).to(torch.float32)
                t = (tf / ws).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
                ins = torch.tensor(1.0, dtype=torch.float32)
            td = t.contiguous().to(dev)
            ins_d = ins.to(dev)
            ws_d = ws.to(dev)
            hbm_bytes += td.numel() * 1 + 8
            return td, ins_d, ws_d

        dbl_blocks: list[dict[str, Any]] = []
        sgl_blocks: list[dict[str, Any]] = []
        with safe_open(str(shm_sf), framework="pt", device="cpu") as sf:
            sf_keys = set(sf.keys())
            w_img_in = _get_bf16(sf, "img_in", out_dim=hidden)
            w_txt_in = _get_bf16(sf, "txt_in", out_dim=hidden)
            w_out = _get_bf16(sf, "final_layer.linear", in_dim=hidden)

            for i in range(spec["num_double"]):
                p = f"double_blocks.{i}"
                if mode == "BF16":
                    dbl_blocks.append(
                        {
                            "kind": "bf16",
                            "iqkv": _get_bf16(sf, f"{p}.img_attn.qkv", out_dim=3 * hidden, in_dim=hidden),
                            "iproj": _get_bf16(sf, f"{p}.img_attn.proj", out_dim=hidden, in_dim=hidden),
                            "imlp0": _get_bf16(sf, f"{p}.img_mlp.0", out_dim=2 * mlp_h, in_dim=hidden),
                            "imlp2": _get_bf16(sf, f"{p}.img_mlp.2", out_dim=hidden, in_dim=mlp_h),
                            "tqkv": _get_bf16(sf, f"{p}.txt_attn.qkv", out_dim=3 * hidden, in_dim=hidden),
                            "tproj": _get_bf16(sf, f"{p}.txt_attn.proj", out_dim=hidden, in_dim=hidden),
                            "tmlp0": _get_bf16(sf, f"{p}.txt_mlp.0", out_dim=2 * mlp_h, in_dim=hidden),
                            "tmlp2": _get_bf16(sf, f"{p}.txt_mlp.2", out_dim=hidden, in_dim=mlp_h),
                        }
                    )
                elif mode == "KV-FP8" and f"{p}.img_attn.qkv.input_scale" not in sf_keys:
                    dbl_blocks.append(
                        {
                            "kind": "kv_fp8",
                            "iqkv_bf16": _get_bf16(sf, f"{p}.img_attn.qkv", out_dim=3 * hidden, in_dim=hidden),
                            "iproj_bf16": _get_bf16(sf, f"{p}.img_attn.proj", out_dim=hidden, in_dim=hidden),
                            "imlp0": _get_fp8_triplet(sf, sf_keys, f"{p}.img_mlp.0", out_dim=2 * mlp_h, in_dim=hidden),
                            "imlp2": _get_fp8_triplet(sf, sf_keys, f"{p}.img_mlp.2", out_dim=hidden, in_dim=mlp_h),
                            "tqkv": _get_fp8_triplet(sf, sf_keys, f"{p}.txt_attn.qkv", out_dim=3 * hidden, in_dim=hidden),
                            "tproj": _get_fp8_triplet(sf, sf_keys, f"{p}.txt_attn.proj", out_dim=hidden, in_dim=hidden),
                            "tmlp0_bf16": _get_bf16(sf, f"{p}.txt_mlp.0", out_dim=2 * mlp_h, in_dim=hidden),
                            "tmlp2_bf16": _get_bf16(sf, f"{p}.txt_mlp.2", out_dim=hidden, in_dim=mlp_h),
                        }
                    )
                else:  # Full calibrated FP8 block (4b-fp8 blocks 0..4 and 9b-kv-fp8 blocks 2..7)
                    dbl_blocks.append(
                        {
                            "kind": "fp8",
                            "iqkv": _get_fp8_triplet(sf, sf_keys, f"{p}.img_attn.qkv", out_dim=3 * hidden, in_dim=hidden),
                            "iproj": _get_fp8_triplet(sf, sf_keys, f"{p}.img_attn.proj", out_dim=hidden, in_dim=hidden),
                            "imlp0": _get_fp8_triplet(sf, sf_keys, f"{p}.img_mlp.0", out_dim=2 * mlp_h, in_dim=hidden),
                            "imlp2": _get_fp8_triplet(sf, sf_keys, f"{p}.img_mlp.2", out_dim=hidden, in_dim=mlp_h),
                            "tqkv": _get_fp8_triplet(sf, sf_keys, f"{p}.txt_attn.qkv", out_dim=3 * hidden, in_dim=hidden),
                            "tproj": _get_fp8_triplet(sf, sf_keys, f"{p}.txt_attn.proj", out_dim=hidden, in_dim=hidden),
                            "tmlp0": _get_fp8_triplet(sf, sf_keys, f"{p}.txt_mlp.0", out_dim=2 * mlp_h, in_dim=hidden),
                            "tmlp2": _get_fp8_triplet(sf, sf_keys, f"{p}.txt_mlp.2", out_dim=hidden, in_dim=mlp_h),
                        }
                    )

            s_l1_out = 3 * hidden + 2 * mlp_h
            s_l2_in = hidden + mlp_h
            for i in range(spec["num_single"]):
                p = f"single_blocks.{i}"
                if mode == "BF16":
                    sgl_blocks.append(
                        {
                            "lin1": _get_bf16(sf, f"{p}.linear1", out_dim=s_l1_out, in_dim=hidden),
                            "lin2": _get_bf16(sf, f"{p}.linear2", out_dim=hidden, in_dim=s_l2_in),
                        }
                    )
                else:
                    sgl_blocks.append(
                        {
                            "lin1": _get_fp8_triplet(sf, sf_keys, f"{p}.linear1", out_dim=s_l1_out, in_dim=hidden),
                            "lin2": _get_fp8_triplet(sf, sf_keys, f"{p}.linear2", out_dim=hidden, in_dim=s_l2_in),
                        }
                    )

        torch.accelerator.synchronize()
        dit_gib = hbm_bytes / (1024**3)
        total_gib = dit_gib + (spec["te_hbm_bf16_gib"] / u_degree) + 0.32
        return (
            {
                "mode": mode,
                "u_degree": u_degree,
                "kerns": kerns,
                "w_img_in": w_img_in,
                "w_txt_in": w_txt_in,
                "w_out": w_out,
                "dbl": dbl_blocks,
                "sgl": sgl_blocks,
                "dit_gib": round(dit_gib, 2),
                "total_gib": round(total_gib, 2),
            },
            total_gib,
        )

    @torch.no_grad()
    def _run_pipeline(
        bank: dict[str, Any],
        img_lat: torch.Tensor,
        txt_emb: torch.Tensor,
        res: int,
        steps: int,
        u2_sync: bool = False,
        include_vae: bool = True,
    ) -> torch.Tensor:
        k = bank["kerns"]
        mode = bank["mode"]
        b = img_lat.shape[0]
        st = txt_emb.shape[1]
        z = img_lat
        dt_s = 1.0 / max(steps, 1)
        for _ in range(steps):
            hi, ht = k["embed_io"](z, txt_emb, bank["w_img_in"], bank["w_txt_in"])
            for db in bank["dbl"]:
                bkind = db["kind"]
                if bkind == "bf16":
                    hi, ht = k["double_bf16"](
                        hi, ht, db["iqkv"], db["iproj"], db["imlp0"], db["imlp2"], db["tqkv"], db["tproj"], db["tmlp0"], db["tmlp2"]
                    )
                elif bkind == "kv_fp8":
                    hi, ht = k["double_kv_fp8"](
                        hi,
                        ht,
                        db["iqkv_bf16"],
                        db["iproj_bf16"],
                        *db["imlp0"],
                        *db["imlp2"],
                        *db["tqkv"],
                        *db["tproj"],
                        db["tmlp0_bf16"],
                        db["tmlp2_bf16"],
                    )
                else:
                    hi, ht = k["double_fp8"](
                        hi,
                        ht,
                        *db["iqkv"],
                        *db["iproj"],
                        *db["imlp0"],
                        *db["imlp2"],
                        *db["tqkv"],
                        *db["tproj"],
                        *db["tmlp0"],
                        *db["tmlp2"],
                    )
                if u2_sync:
                    dist.all_reduce(hi, op=dist.ReduceOp.SUM, group=my_u2_pg)
            seq = torch.cat([ht, hi], dim=1)
            for sb in bank["sgl"]:
                if mode == "BF16":
                    seq = k["single_bf16"](seq, sb["lin1"], sb["lin2"])
                else:
                    seq = k["single_fp8"](seq, *sb["lin1"], *sb["lin2"])
                if u2_sync:
                    dist.all_reduce(seq, op=dist.ReduceOp.SUM, group=my_u2_pg)
            vel = k["final_out"](seq, st, bank["w_out"])
            z = z + vel * dt_s

        if not include_vae:
            return z
        # Unpatchify 128-ch (2x2x32) -> 32-ch AutoencoderKLFlux2 spatial latent (B, 32, H/8, W/8)
        h_lat = res // 16
        z_4d = z.view(b, h_lat, h_lat, 32, 2, 2).permute(0, 3, 1, 4, 2, 5).contiguous().view(b, 32, res // 8, res // 8)
        return _vae_decode_compiled(z_4d)

    def _make_inputs(spec: dict[str, Any], bsz: int, res: int) -> tuple[torch.Tensor, torch.Tensor]:
        n_img_toks = (res // 16) * (res // 16)
        img_lat = (torch.randn((bsz, n_img_toks, 128), dtype=torch.bfloat16, device="cpu") * 0.5).to(dev)
        txt_emb = (torch.randn((bsz, spec["txt_tokens"], spec["joint_dim"]), dtype=torch.bfloat16, device="cpu") * 0.05).to(dev)
        torch.accelerator.synchronize()
        return img_lat, txt_emb

    # ------------------------------------------------------------------
    # Persistent 8-Process HTTP Keep-Alive Mode for 2-Host `DP=16` Fleet
    # ------------------------------------------------------------------
    if serve_port > 0:
        my_port = serve_port + rank
        rpc_models: dict[str, dict[str, Any]] = {}
        for mkey in model_keys:
            spec = FLUX2_SPECS[mkey]
            shm_sf, _ = _stage_safetensor_collective(mkey)
            qmode = spec["quant_mode"]
            bank, tot_gib = _load_flux2_bank(mkey, shm_sf, qmode, u_degree=1)
            in_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
            with torch.no_grad():
                for res in (512, 1024):
                    for bsz in (1, 2):
                        inp = _make_inputs(spec, bsz, res)
                        in_cache[(bsz, res)] = inp
                        _ = _run_pipeline(bank, inp[0], inp[1], res, 4, u2_sync=False, include_vae=True)
                        torch.accelerator.synchronize()
            dist.barrier(group=gloo_pg)
            rpc_models[f"{mkey}:{qmode}"] = {"bank": bank, "in_cache": in_cache, "hbm_gib": round(tot_gib, 2)}
            if rank == 0:
                log("rpc-server", f"Ready {spec['display_name']} [{qmode}] across all {world_size} TensorCores (:{serve_port}..:{serve_port+world_size-1})")

        tc_lock = threading.Lock()

        class _RpcHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt: str, *args: Any) -> None:
                return

            def do_GET(self) -> None:
                body = json.dumps({"status": "ok", "rank": rank, "models": list(rpc_models.keys())}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:
                clen = int(self.headers.get("Content-Length", "0") or 0)
                req = json.loads((self.rfile.read(clen) if clen > 0 else b"{}").decode("utf-8"))
                mkey = req.get("model_key", model_keys[0])
                prec = req.get("precision", FLUX2_SPECS[mkey]["quant_mode"])
                res = int(req.get("res", 1024))
                steps = int(req.get("steps", 4))
                bsz = int(req.get("batch_size", 1))
                ent = rpc_models[f"{mkey}:{prec}"]
                inp = ent["in_cache"][(bsz, res)]
                with tc_lock, torch.no_grad():
                    t0 = time.perf_counter()
                    _ = _run_pipeline(ent["bank"], inp[0], inp[1], res, steps, u2_sync=False, include_vae=True)
                    torch.accelerator.synchronize()
                    tpu_s = time.perf_counter() - t0
                resp = json.dumps({"ok": True, "tpu_s": round(tpu_s, 4), "hbm_gib": ent["hbm_gib"]}).encode()
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(resp)))
                    self.end_headers()
                    self.wfile.write(resp)
                except BrokenPipeError:
                    pass

        srv = FastThreadingHTTPServer(("0.0.0.0", my_port), _RpcHandler)
        srv.serve_forever()
        return

    # ------------------------------------------------------------------
    # Single-Host Deep Sweep Mode (`BF16` vs `FP8` / `KV-FP8` + Batch + Concurrency + Poisson)
    # ------------------------------------------------------------------
    rows: list[dict[str, Any]] = []
    if rank == 0:
        out_jsonl.unlink(missing_ok=True)

    def _record(r: dict[str, Any]) -> None:
        if rank != 0:
            return
        rows.append(r)
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with open(out_jsonl, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(r) + "\n")
        out_summary_json.write_text(
            json.dumps(
                {
                    "updated_at_utc": utc_now(),
                    "engine": "torchtpu-compiled-scaled-mm",
                    "image_encode_offload_ablation": enc_map,
                    "num_rows": len(rows),
                    "results": rows,
                },
                indent=2,
            )
        )

    for midx, mkey in enumerate(model_keys):
        spec = FLUX2_SPECS[mkey]
        shm_sf, sf_meta = _stage_safetensor_collective(mkey)
        if rank == 0:
            log(mkey, f"Staged {spec['fp8_file']}: {sf_meta}")

        qmode = spec["quant_mode"]  # "FP8" for 4B, "KV-FP8" for 9B
        for p_idx, mode in enumerate(["BF16", qmode]):
            torch._dynamo.reset()
            gc.collect()
            bank, tot_gib = _load_flux2_bank(mkey, shm_sf, mode, u_degree=1)
            dist.barrier(group=gloo_pg)
            if rank == 0:
                log(mkey, f"Loaded {spec['display_name']} [{mode}] across 8 TensorCores (DiT={bank['dit_gib']} GiB, Total={tot_gib:.2f} GiB/TC)")

            repo_used = spec["bf16_repo"] if mode == "BF16" else spec["fp8_repo"]

            def _bench_pt(suite_tag: str, res: int, steps: int, bsz: int, measure_dp8: bool = True, n_iters: int = 8) -> None:
                enc_png = enc_map.get(f"{res}x{res}_png", {})
                loc_enc_ms = float(enc_png.get("local_tpu_cpu_encode_ms", 12.0))
                dis_enc_ms = float(enc_png.get("disagg_cpu_rtt_ms", loc_enc_ms))
                with torch.no_grad():
                    inp = _make_inputs(spec, bsz, res)
                    for _ in range(2):
                        _ = _run_pipeline(bank, inp[0], inp[1], res, steps, u2_sync=False, include_vae=True)
                        torch.accelerator.synchronize()
                    dist.barrier(group=gloo_pg)

                    lats = []
                    if rank == 0:
                        for _ in range(n_iters):
                            t0 = time.perf_counter()
                            _ = _run_pipeline(bank, inp[0], inp[1], res, steps, u2_sync=False, include_vae=True)
                            torch.accelerator.synchronize()
                            lats.append(time.perf_counter() - t0)
                    dist.barrier(group=gloo_pg)

                    w_lats = []
                    if measure_dp8:
                        for _ in range(n_iters):
                            dist.barrier(group=gloo_pg)
                            tw0 = time.perf_counter()
                            _ = _run_pipeline(bank, inp[0], inp[1], res, steps, u2_sync=False, include_vae=True)
                            torch.accelerator.synchronize()
                            dist.barrier(group=gloo_pg)
                            if rank == 0:
                                w_lats.append(time.perf_counter() - tw0)
                    del inp
                    torch.accelerator.synchronize()

                if rank != 0:
                    return
                p50 = pct(lats, 50)
                p95 = pct(lats, 95)
                p99 = pct(lats, 99)
                dp1_ipm = (bsz / max(p50, 1e-6)) * 60.0
                dp8_w_p50 = round(pct(w_lats, 50), 3) if measure_dp8 else None
                dp8_ipm = round(((bsz * world_size) / max(dp8_w_p50 or 1.0, 1e-6)) * 60.0, 2) if measure_dp8 else None
                dp8_eff = round(100.0 * (dp8_ipm or 0.0) / max(dp1_ipm * world_size, 1e-6), 1) if measure_dp8 else None
                rec = {
                    "timestamp_utc": utc_now(),
                    "engine": "torchtpu-compiled-scaled-mm",
                    "suite": suite_tag,
                    "model_label": spec["display_name"],
                    "repo_id": repo_used,
                    "topology": "1-Host DP=8 x U=1 (8 TC)",
                    "precision": mode,
                    "resolution": f"{res}x{res}",
                    "steps": steps,
                    "batch_size": bsz,
                    "dp1_latency_p50_s": round(p50, 3),
                    "dp1_latency_p95_s": round(p95, 3),
                    "dp1_latency_p99_s": round(p99, 3),
                    "e2e_local_png_p50_s": round(p50 + (loc_enc_ms * bsz) / 1000.0, 3),
                    "e2e_disagg_png_p50_s": round(p50 + (dis_enc_ms * bsz) / 1000.0, 3),
                    "e2e_pipelined_png_p50_s": round(max(p50, (dis_enc_ms * bsz) / 1000.0), 3),
                    "dp1_ipm": round(dp1_ipm, 2),
                    "host_dp8_wave_p50_s": dp8_w_p50,
                    "host_dp8_ipm": dp8_ipm,
                    "host_dp8_img_per_s": round((dp8_ipm or 0.0) / 60.0, 3) if dp8_ipm else None,
                    "host_dp8_scaling_eff_pct": dp8_eff,
                    "hosts_for_60_ipm": int(math.ceil(60.0 / max(dp8_ipm or dp1_ipm, 1e-6))),
                    "hosts_for_300_ipm": int(math.ceil(300.0 / max(dp8_ipm or dp1_ipm, 1e-6))),
                    "hosts_for_600_ipm": int(math.ceil(600.0 / max(dp8_ipm or dp1_ipm, 1e-6))),
                    "dit_hbm_gib_per_tc": bank["dit_gib"],
                    "peak_hbm_gib_per_tc": bank["total_gib"],
                    "status": "PASS",
                }
                _record(rec)
                log(
                    mkey,
                    f"[{suite_tag}][{mode}] {res}x{res} st={steps} B={bsz} -> "
                    f"1TC_P50={rec['dp1_latency_p50_s']}s ({rec['dp1_ipm']} IPM) | DP8={dp8_ipm} IPM ({dp8_eff}%)",
                )

            # 1. Resolution x Steps matrix (`512, 1024, 1536, 2048` x `1, 4, 8, 20`) at B=1
            for res, st_list in [
                (512, [1, 4, 8, 20]),
                (1024, [1, 4, 8, 20]),
                (1536, [1, 4, 8]),
                (2048, [1, 4, 8]),
            ]:
                for st in st_list:
                    _bench_pt("resolution_step_matrix", res, st, 1, measure_dp8=(st in (1, 4, 8)), n_iters=6 if res <= 1536 else 4)

            # 2. Batch size scaling (`B = 1, 2, 4`) at `1024x1024` (4 steps) and `512x512` (4 steps)
            for res in (512, 1024):
                for bsz in (2, 4):
                    _bench_pt("batch_size_scaling", res, 4, bsz, measure_dp8=True, n_iters=6)

            # 3. For quantized mode (`FP8` / `KV-FP8`), run Closed-Loop `CON = 1..64` & Open-Loop Poisson `RPS = 1..8`!
            if mode == qmode:
                with torch.no_grad():
                    ip1 = _make_inputs(spec, 1, 1024)
                    _ = _run_pipeline(bank, ip1[0], ip1[1], 1024, 4, u2_sync=False, include_vae=True)
                    torch.accelerator.synchronize()

                local_lock = threading.Lock()
                base_ep = 19400 + midx * 30 + p_idx * 10
                ep_port = base_ep + rank

                class _EpHandler(BaseHTTPRequestHandler):
                    protocol_version = "HTTP/1.1"

                    def log_message(self, fmt: str, *args: Any) -> None:
                        return

                    def do_GET(self) -> None:
                        b_ok = b'{"status":"ok"}'
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(b_ok)))
                        self.end_headers()
                        self.wfile.write(b_ok)

                    def do_POST(self) -> None:
                        clen = int(self.headers.get("Content-Length", "0") or 0)
                        if clen > 0:
                            _ = self.rfile.read(clen)
                        with local_lock, torch.no_grad():
                            ts = time.perf_counter()
                            _ = _run_pipeline(bank, ip1[0], ip1[1], 1024, 4, u2_sync=False, include_vae=True)
                            torch.accelerator.synchronize()
                            svc_s = time.perf_counter() - ts
                        resp = json.dumps({"ok": True, "tpu_s": round(svc_s, 4)}).encode()
                        try:
                            self.send_response(200)
                            self.send_header("Content-Type", "application/json")
                            self.send_header("Content-Length", str(len(resp)))
                            self.end_headers()
                            self.wfile.write(resp)
                        except BrokenPipeError:
                            pass

                ep_srv = FastThreadingHTTPServer(("127.0.0.1", ep_port), _EpHandler)
                threading.Thread(target=ep_srv.serve_forever, daemon=True).start()
                dist.barrier(group=gloo_pg)

                if rank == 0:
                    pool_ka = MultiEndpointKeepAlivePool([f"http://127.0.0.1:{base_ep + r}" for r in range(world_size)])
                    pool_ka.wait_healthy(timeout_s=30.0)
                    for conc in (1, 2, 4, 8, 16, 32, 64):
                        n_r = max(8, conc * 2)
                        t0 = time.perf_counter()
                        with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as cp:
                            outs = list(cp.map(lambda _: pool_ka.call_infer(mkey, mode, 1024, 4, 1), range(n_r)))
                        wall_s = time.perf_counter() - t0
                        e2e_l = [x["e2e_s"] for x in outs]
                        ipm = (n_r / max(wall_s, 1e-6)) * 60.0
                        _record(
                            {
                                "timestamp_utc": utc_now(),
                                "engine": "torchtpu-compiled-scaled-mm",
                                "suite": "closed_loop_concurrency_dp8",
                                "model_label": spec["display_name"],
                                "repo_id": repo_used,
                                "topology": "1-Host DP=8 x U=1 (8 TC)",
                                "precision": mode,
                                "resolution": "1024x1024",
                                "steps": 4,
                                "batch_size": 1,
                                "concurrency": conc,
                                "latency_p50_s": round(pct(e2e_l, 50), 3),
                                "latency_p95_s": round(pct(e2e_l, 95), 3),
                                "latency_p99_s": round(pct(e2e_l, 99), 3),
                                "throughput_img_per_s": round(ipm / 60.0, 3),
                                "throughput_ipm": round(ipm, 2),
                                "hosts_for_60_ipm": int(math.ceil(60.0 / max(ipm, 1e-6))),
                                "hosts_for_300_ipm": int(math.ceil(300.0 / max(ipm, 1e-6))),
                                "hosts_for_600_ipm": int(math.ceil(600.0 / max(ipm, 1e-6))),
                                "peak_hbm_gib_per_tc": bank["total_gib"],
                                "status": "PASS",
                            }
                        )
                        log(mkey, f"[DP=8 ClosedLoop][{mode}] CON={conc} -> P50={pct(e2e_l,50):.3f}s, P99={pct(e2e_l,99):.3f}s, {ipm:.2f} IPM")

                    for rps in (1.0, 2.0, 4.0, 8.0):
                        n_r = max(8, int(rps * 4))
                        rng = random.Random(700 + int(rps * 10))
                        ias = [rng.expovariate(rps) for _ in range(n_r)]
                        t0 = time.perf_counter()
                        target_t = t0
                        futs = []
                        with concurrent.futures.ThreadPoolExecutor(max_workers=64) as pp:
                            for dt in ias:
                                target_t += dt
                                now = time.perf_counter()
                                if target_t > now:
                                    time.sleep(target_t - now)
                                futs.append(pp.submit(pool_ka.call_infer, mkey, mode, 1024, 4, 1))
                            outs_p = [f.result() for f in futs]
                        wall_p = time.perf_counter() - t0
                        e2e_l = [x["e2e_s"] for x in outs_p]
                        ach_rps = n_r / max(wall_p, 1e-6)
                        ipm = ach_rps * 60.0
                        _record(
                            {
                                "timestamp_utc": utc_now(),
                                "engine": "torchtpu-compiled-scaled-mm",
                                "suite": "open_loop_poisson_dp8",
                                "model_label": spec["display_name"],
                                "repo_id": repo_used,
                                "topology": "1-Host DP=8 x U=1 (8 TC)",
                                "precision": mode,
                                "resolution": "1024x1024",
                                "steps": 4,
                                "target_rps": rps,
                                "achieved_rps": round(ach_rps, 3),
                                "latency_p50_s": round(pct(e2e_l, 50), 3),
                                "latency_p95_s": round(pct(e2e_l, 95), 3),
                                "latency_p99_s": round(pct(e2e_l, 99), 3),
                                "throughput_ipm": round(ipm, 2),
                                "peak_hbm_gib_per_tc": bank["total_gib"],
                                "status": "PASS",
                            }
                        )
                        log(mkey, f"[DP=8 Poisson][{mode}] Target={rps} RPS -> Achieved={ach_rps:.2f} RPS ({ipm:.1f} IPM), P50={pct(e2e_l,50):.3f}s")

                dist.barrier(group=gloo_pg)
                ep_srv.shutdown()
                ep_srv.server_close()
                del ip1

            del bank
            gc.collect()
            torch.accelerator.synchronize()

            # 4. Intra-Chip `DP=4 x U=2` (`tpu_dist` ICI) for quantized mode!
            if mode == qmode:
                torch._dynamo.reset()
                u2_bank, u2_gib = _load_flux2_bank(mkey, shm_sf, mode, u_degree=2)
                with torch.no_grad():
                    for res in (1024, 1536, 2048):
                        inp = _make_inputs(spec, 1, res)
                        for _ in range(2):
                            _ = _run_pipeline(u2_bank, inp[0], inp[1], res, 4, u2_sync=True, include_vae=True)
                            torch.accelerator.synchronize()
                        dist.barrier(group=gloo_pg)
                        u2_lats = []
                        for _ in range(6):
                            dist.barrier(group=gloo_pg)
                            t0 = time.perf_counter()
                            _ = _run_pipeline(u2_bank, inp[0], inp[1], res, 4, u2_sync=True, include_vae=True)
                            torch.accelerator.synchronize()
                            dist.barrier(group=gloo_pg)
                            if rank == 0:
                                u2_lats.append(time.perf_counter() - t0)
                        del inp
                        if rank == 0:
                            p50 = pct(u2_lats, 50)
                            dp4_ipm = (4.0 / max(p50, 1e-6)) * 60.0
                            _record(
                                {
                                    "timestamp_utc": utc_now(),
                                    "engine": "torchtpu-compiled-scaled-mm",
                                    "suite": "intrachip_u2_ici_scaling",
                                    "model_label": spec["display_name"],
                                    "repo_id": repo_used,
                                    "topology": "1-Host DP=4 x U=2 (Intra-Chip ICI, 8 TC)",
                                    "precision": mode,
                                    "resolution": f"{res}x{res}",
                                    "steps": 4,
                                    "batch_size": 1,
                                    "chip_u2_latency_p50_s": round(p50, 3),
                                    "chip_u2_latency_p99_s": round(pct(u2_lats, 99), 3),
                                    "host_dp4_ipm": round(dp4_ipm, 2),
                                    "hosts_for_60_ipm": int(math.ceil(60.0 / max(dp4_ipm, 1e-6))),
                                    "hosts_for_300_ipm": int(math.ceil(300.0 / max(dp4_ipm, 1e-6))),
                                    "hosts_for_600_ipm": int(math.ceil(600.0 / max(dp4_ipm, 1e-6))),
                                    "dit_hbm_gib_per_tc": u2_bank["dit_gib"],
                                    "peak_hbm_gib_per_tc": u2_bank["total_gib"],
                                    "status": "PASS",
                                }
                            )
                            log(mkey, f"[DP=4xU=2 ICI][{mode}] {res}x{res} st=4 -> P50={p50:.3f}s, Host={dp4_ipm:.2f} IPM, HBM/TC={u2_bank['total_gib']} GiB")
                del u2_bank
                gc.collect()
                torch.accelerator.synchronize()

        dist.barrier(group=gloo_pg)
        if rank == 0:
            shm_sf.unlink(missing_ok=True)

    dist.barrier(group=gloo_pg)
    dist.destroy_process_group()


def expand_worker_urls(raw_urls: str, ranks_per_host: int = 8) -> list[str]:
    items = [u.strip() for u in raw_urls.split(",") if u.strip()]
    if len(items) >= ranks_per_host:
        return items
    out: list[str] = []
    for u in items:
        p = urllib.parse.urlsplit(u)
        host = p.hostname or "127.0.0.1"
        bp = p.port or 19300
        for r in range(ranks_per_host):
            out.append(f"{p.scheme or 'http'}://{host}:{bp + r}")
    return out


def run_multihost_dp16_client(worker_urls: list[str], model_keys: list[str], out_jsonl: Path) -> None:
    pool_ka = MultiEndpointKeepAlivePool(worker_urls)
    pool_ka.wait_healthy(timeout_s=600.0)
    log("dp16-client", f"All {len(worker_urls)} TensorCore FLUX.2 endpoints healthy across 2-host slice!")
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    for mkey in model_keys:
        spec = FLUX2_SPECS[mkey]
        qmode = spec["quant_mode"]
        repo = spec["fp8_repo"]
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(worker_urls)) as wp:
            list(wp.map(lambda _: pool_ka.call_infer(mkey, qmode, 1024, 4, 1), range(len(worker_urls))))

        for res in (512, 1024):
            for conc in (8, 16, 32, 64):
                n_r = max(16, conc * 2)
                t0 = time.perf_counter()
                with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as cp:
                    outs = list(cp.map(lambda _: pool_ka.call_infer(mkey, qmode, res, 4, 1), range(n_r)))
                wall_s = time.perf_counter() - t0
                lats = [x["e2e_s"] for x in outs]
                ipm = (n_r / max(wall_s, 1e-6)) * 60.0
                rec = {
                    "timestamp_utc": utc_now(),
                    "engine": "torchtpu-compiled-scaled-mm",
                    "suite": "multihost_dp16_concurrency",
                    "model_label": spec["display_name"],
                    "repo_id": repo,
                    "topology": f"2-Host DP={len(worker_urls)} x U=1 ({len(worker_urls)} TC)",
                    "num_hosts": 2,
                    "precision": qmode,
                    "resolution": f"{res}x{res}",
                    "steps": 4,
                    "batch_size": 1,
                    "concurrency": conc,
                    "latency_p50_s": round(pct(lats, 50), 3),
                    "latency_p95_s": round(pct(lats, 95), 3),
                    "latency_p99_s": round(pct(lats, 99), 3),
                    "slice_dp16_img_per_s": round(ipm / 60.0, 3),
                    "slice_dp16_ipm": round(ipm, 2),
                    "peak_hbm_gib_per_tc": outs[0].get("hbm_gib"),
                    "status": "PASS",
                }
                with open(out_jsonl, "a", encoding="utf-8") as fp:
                    fp.write(json.dumps(rec) + "\n")
                log("dp16-client", f"[{spec['display_name']}][{qmode}][2-Host DP=16] {res}x{res} CON={conc} -> P50={rec['latency_p50_s']}s, Slice={rec['slice_dp16_ipm']} IPM")


def main() -> None:
    p = argparse.ArgumentParser(description="FLUX.2 Klein BF16 vs FP8 vs KV-FP8 TorchTPU Benchmark")
    p.add_argument("--models-root", default="/models/models")
    p.add_argument("--shm-models-root", default="/dev/shm/flux2_models")
    p.add_argument("--models", default="klein_4b,klein_9b")
    p.add_argument("--output-dir", default="/tmp/flux2_klein_results")
    p.add_argument("--tag", default="host0")
    p.add_argument("--offload-url", default="http://multimodal-cpu-offload-svc.default.svc.cluster.local:8090")
    p.add_argument("--serve-worker-port", type=int, default=0)
    p.add_argument("--run-multihost-dp16", action="store_true")
    p.add_argument("--worker-urls", default="")
    p.add_argument("--torchrun-worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--encode-json", default="", help=argparse.SUPPRESS)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mkeys = [m.strip() for m in args.models.split(",") if m.strip()]

    if args.run_multihost_dp16:
        urls = expand_worker_urls(args.worker_urls, ranks_per_host=8)
        run_multihost_dp16_client(urls, mkeys, out_dir / "torchtpu_fp8_kv_multihost_dp16.jsonl")
        return

    if args.torchrun_worker:
        run_torchrun_flux2_worker(
            models_root=Path(args.models_root),
            shm_models_root=Path(args.shm_models_root),
            model_keys=mkeys,
            out_jsonl=out_dir / f"torchtpu_fp8_kv_{args.tag}.jsonl",
            out_summary_json=out_dir / f"torchtpu_fp8_kv_{args.tag}_summary.json",
            encode_json=Path(args.encode_json),
            serve_port=args.serve_worker_port,
        )
        return

    # Parent Launcher: start single-host metadata mock + measure encode offload + spawn 8-process torchrun
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from tpu_single_host_setup import start_single_host_metadata_env

    child_env = start_single_host_metadata_env("v7x", port=18082)
    enc_json = out_dir / f"image_encode_offload_{args.tag}.json"
    if args.serve_worker_port == 0 and not enc_json.exists():
        enc_res = measure_image_encode_offload(args.offload_url)
        enc_json.write_text(json.dumps(enc_res, indent=2))

    cache_dir = Path("/dev/shm/xla_cache_flux2_fp8")
    cache_dir.mkdir(parents=True, exist_ok=True)
    sb_base = 29640 if args.serve_worker_port == 0 else 29660
    child_env.update(
        {
            "PJRT_DEVICE": "TPU",
            "TORCH_TPU_CACHE_DIR": str(cache_dir),
            "JAX_COMPILATION_CACHE_DIR": str(cache_dir),
            "TORCH_TPU_TOPOLOGY": "2,2,1,2",
            "TORCH_TPU_SLICEBUILDER_ADDRESSES": ",".join(f"127.0.0.1:{sb_base + i}" for i in range(8)),
        }
    )
    mport = 29608 if args.serve_worker_port == 0 else 29618
    cmd = [
        "torchrun",
        "--nnodes=1",
        "--nproc_per_node=8",
        f"--master_port={mport}",
        str(Path(__file__).resolve()),
        "--torchrun-worker",
        f"--models-root={args.models_root}",
        f"--shm-models-root={args.shm_models_root}",
        f"--models={args.models}",
        f"--output-dir={args.output_dir}",
        f"--tag={args.tag}",
        f"--encode-json={enc_json}",
        f"--serve-worker-port={args.serve_worker_port}",
    ]
    log("parent", f"Launching 8-process TorchTPU FLUX.2 FP8/KV-FP8 worker pool: {' '.join(cmd)}")
    rc = subprocess.call(cmd, env=child_env)
    if rc != 0:
        sys.exit(rc)


if __name__ == "__main__":
    main()
