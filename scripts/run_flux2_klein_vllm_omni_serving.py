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
"""FLUX.2 Klein (4B & 9B) + FLUX.1-schnell + Qwen-Image-2.1-Turbo vLLM-Omni Serving Benchmark on TPU v7x.

Benchmarks production OpenAI `/v1/images/generations` multi-replica serving on TPU v7x
(`vllm serve --omni` with sub-host VFIO rank pinning + cross-host 16-TensorCore fleet routing):
  - Models:
      1. `black-forest-labs/FLUX.2-klein-4B` (`Flux2KleinPipeline`, 5 double + 20 single blocks, Qwen3-4B)
      2. `Comfy-Org/Flux-2-Klein-9B` (`Flux2KleinPipeline`, 8 double + 24 single blocks, Qwen3-8B)
      3. `black-forest-labs/FLUX.1-schnell` (`FluxPipeline`, 19 double + 38 single blocks, 12B baseline)
      4. `Qwen/Qwen-Image-2-1-Turbo` (`QwenImage21Pipeline`, 8-step distilled multimodal baseline)
  - Topologies:
      * Single-Host (4 physical v7x chips = 8 TensorCores):
        `DP=8 x U=1`, `DP=4 x U=2` (Intra-Chip zero-hop), `DP=2 x U=4`, `DP=1 x U=8`
      * Two-Host Slice (8 physical v7x chips = 16 TensorCores):
        `2-Host DP=16 x U=1` and `2-Host DP=8 x U=2`
  - Resolutions & Steps: `512x512`, `1024x1024` (1MP), `1536x1536` (2.3MP), `2048x2048` (4MP) x `1, 4, 8, 20` steps
  - Closed-Loop Concurrency (`CON = 1, 2, 4, 8, 16, 32, 64`) & Open-Loop Poisson (`RPS = 1, 2, 4, 8`)
  - Saves real generated sample PNGs under `<output-dir>/samples/`.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import py_compile
import random
import re
import signal
import socket
import statistics
import subprocess
import threading
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image

PROMPTS = [
    "A hyper-detailed architectural photograph of a futuristic glass conservatory nestled in misty Nordic fjords at sunrise, golden rim light, crisp reflections on calm water, 8k editorial photography.",
    "Close-up studio portrait of an ornate mechanical clockwork owl perched on moss-covered volcanic stone, brass gears, emerald glass eyes, shallow depth of field, dramatic rim lighting.",
    "Aerial view of vibrant bioluminescent waves washing onto a black sand beach in Iceland under the aurora borealis, long exposure, ultra-sharp stars and sea spray.",
    "A cozy sunlit artisan bakery in Kyoto with freshly baked croissants, steam rising from ceramic pour-over coffee, warm cedar woodwork, natural morning window light.",
    "An astronaut botanist tending glowing hydroponic orchids inside a geodesic dome on Mars, red desert canyon visible through panoramic windows, cinematic sci-fi realism.",
    "Macro wildlife photograph of a translucent glass frog resting on a raindrop-covered heliconia leaf in a Costa Rican cloud forest, vivid colors, razor-sharp focus.",
    "A sleek vintage silver streamliner train crossing a stone viaduct over an autumn alpine valley, turquoise glacial river below, crisp mountain air, medium format film look.",
    "Isometric cutaway illustration of a bustling subterranean solarpunk library with spiral staircases, hanging gardens, warm lanterns, and stained-glass skylights.",
]

METADATA_HOOK_C = r"""
#define _GNU_SOURCE
#include <sys/socket.h>
#include <sys/types.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <dlfcn.h>
#include <string.h>

static int (*real_connect)(int, const struct sockaddr*, socklen_t) = NULL;

int connect(int sockfd, const struct sockaddr *addr, socklen_t addrlen) {
    if (!real_connect) {
        real_connect = dlsym(RTLD_NEXT, "connect");
    }
    if (addr && addr->sa_family == AF_INET) {
        const struct sockaddr_in *addr_in = (const struct sockaddr_in *)addr;
        char ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &(addr_in->sin_addr), ip, INET_ADDRSTRLEN);
        if (strcmp(ip, "169.254.169.254") == 0) {
            struct sockaddr_in redirected = *addr_in;
            redirected.sin_addr.s_addr = inet_addr("127.0.0.1");
            redirected.sin_port = htons(80);
            return real_connect(sockfd, (const struct sockaddr *)&redirected, sizeof(redirected));
        }
    }
    return real_connect(sockfd, addr, addrlen);
}
"""

MOCK_METADATA_PY = r"""#!/usr/bin/env python3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import sys
import urllib.error
import urllib.parse
import urllib.request


class MockMetadataHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def forward_to_real_metadata(self):
        real_url = f"http://169.254.169.254{self.path}"
        req = urllib.request.Request(real_url)
        for key, val in self.headers.items():
            if key.lower() != "host":
                req.add_header(key, val)
        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                self.send_response(response.status)
                for key, val in response.headers.items():
                    self.send_header(key, val)
                self.end_headers()
                self.wfile.write(response.read())
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            for key, val in e.headers.items():
                self.send_header(key, val)
            self.end_headers()
            self.wfile.write(e.read())
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(e).encode())

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        if self.headers.get("Metadata-Flavor") != "Google":
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Missing Metadata-Flavor header")
            return

        response_body = None
        tpu_env_keys = {
            "TPU_WORKER_ID": "0",
            "ALT": "false",
            "CHIPS_PER_HOST_BOUNDS": "2,2,1",
            "CONSUMER_PROJECT_ID": "tpu-benchmark",
            "ENABLE_ICI_RESILIENCY": "false",
            "ENABLE_IMPROVED_REROUTE_ALLREDUCE_STRATEGY": "false",
            "HOST_BOUNDS": "1,1,1",
            "INJECT_SLICE_BUILDER_FAULT": "",
            "NODE_ID": "0",
            "TOPOLOGY": "2x2x1",
            "TPU_ACCELERATOR_TYPE": "tpu7x-8",
            "WRAP": "false,false,false",
            "ZONE": "us-central1-c",
        }
        attr = path.rsplit("/attributes/", 1)[-1] if "/instance/attributes/" in path else None
        if attr in tpu_env_keys:
            response_body = tpu_env_keys[attr].encode()
        elif path.endswith("/computeMetadata/v1/instance/attributes/tpu-env"):
            response_body = (
                "TPU_WORKER_ID: '0'\n"
                "ALT: 'false'\n"
                "CHIPS_PER_HOST_BOUNDS: '2,2,1'\n"
                "CONSUMER_PROJECT_ID: 'tpu-benchmark'\n"
                "ENABLE_ICI_RESILIENCY: 'false'\n"
                "ENABLE_IMPROVED_REROUTE_ALLREDUCE_STRATEGY: 'false'\n"
                "HOST_BOUNDS: '1,1,1'\n"
                "INJECT_SLICE_BUILDER_FAULT: ''\n"
                "NODE_ID: '0'\n"
                "TOPOLOGY: '2x2x1'\n"
                "TPU_ACCELERATOR_TYPE: 'tpu7x-8'\n"
                "WRAP: 'false,false,false'\n"
                "ZONE: 'us-central1-c'\n"
            ).encode()
        elif path.endswith("/computeMetadata/v1/instance/attributes/worker-network-endpoints") or path.endswith(
            "/computeMetadata/v1/instance/attributes/instance-id"
        ):
            response_body = b"tpu-v7x-worker-0"
        elif path.endswith("/computeMetadata/v1/instance/attributes/accelerator-type"):
            response_body = b"tpu7x-8"
        elif path.endswith("/computeMetadata/v1/instance/attributes/agent-worker-number"):
            response_body = b"0"
        elif path.endswith("/computeMetadata/v1/project/project-id"):
            response_body = b"tpu-benchmark"
        elif path.endswith("/computeMetadata/v1/project/numeric-project-id"):
            response_body = b"1234567890"
        elif path.endswith("/computeMetadata/v1/universe/universe-domain") or path.endswith("/universe-domain"):
            response_body = b"googleapis.com"

        if response_body is not None:
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(response_body)
        else:
            self.forward_to_real_metadata()


if __name__ == "__main__":
    httpd = ThreadingHTTPServer(("127.0.0.1", 80), MockMetadataHandler)
    print("[mock_metadata] Serving on 127.0.0.1:80", file=sys.stderr, flush=True)
    httpd.serve_forever()
"""

PATCHED_SET_DEVICE_METHOD = """    @classmethod
    def set_device(cls, device: torch.device | int) -> None:
        world_size = int(os.environ.get("WORLD_SIZE", "1") or "1")
        raw_lrs = os.environ.get("VLLM_OMNI_LOCAL_RANKS", "").strip()
        raw_vis = os.environ.get("TPU_VISIBLE_CHIPS", "").strip()
        ranks: list[str] = []
        if raw_lrs:
            ranks = [x.strip() for x in raw_lrs.split(",") if x.strip()]
        elif raw_vis:
            vis_list = [x.strip() for x in raw_vis.split(",") if x.strip()]
            if len(vis_list) == world_size and [int(c) for c in vis_list] != list(range(world_size)):
                ranks = vis_list

        phys_rank: str | None = None
        sub_topo: str | None = None
        custom_sb = os.environ.get("VLLM_OMNI_SLICEBUILDER_ADDRESSES", "").strip()

        if ranks:
            for _k in (
                "TPU_CHIPS_PER_HOST_BOUNDS",
                "TPU_HOST_BOUNDS",
                "TPU_WORKER_ID",
                "TPU_WORKER_HOSTNAMES",
                "CLOUD_TPU_TASK_ID",
            ):
                os.environ.pop(_k, None)
            w_rank = int(os.environ.get("RANK", "0") or "0")
            phys_rank = ranks[w_rank % len(ranks)]
            vllm_topo = os.environ.get("VLLM_OMNI_TOPOLOGY", "").strip()
            if vllm_topo:
                sub_topo = vllm_topo
            elif world_size == 1:
                sub_topo = ""
            elif world_size == 2:
                sub_topo = "1,1,1,2"
            elif world_size == 4:
                sub_topo = "1,2,1,2"
            else:
                sub_topo = os.environ.get("TORCH_TPU_TOPOLOGY", "2,2,1,2")

            os.environ["LOCAL_RANK"] = phys_rank
            if sub_topo:
                os.environ["TORCH_TPU_TOPOLOGY"] = sub_topo
            else:
                os.environ.pop("TORCH_TPU_TOPOLOGY", None)

        if world_size > 1:
            cls._worker_world_size = world_size
            if custom_sb:
                os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = custom_sb
            else:
                cls._set_shared_slicebuilder_addresses(world_size)
            cls._offload_dense_all_to_all()
        elif custom_sb:
            os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = custom_sb

        cls._prepare_singlehost_tpu_env(world_size)

        if phys_rank is not None:
            os.environ["LOCAL_RANK"] = phys_rank
            os.environ["TPU_VISIBLE_CHIPS"] = phys_rank
            if sub_topo:
                os.environ["TORCH_TPU_TOPOLOGY"] = sub_topo
            else:
                os.environ.pop("TORCH_TPU_TOPOLOGY", None)
        if custom_sb:
            os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = custom_sb

        torch.accelerator.set_device_index(0)
        _ = torch.empty(1, device=torch.device("tpu"))
        torch.accelerator.synchronize()
"""

_MOCK_META_PROC: subprocess.Popen[Any] | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"[{utc_now()}][flux2-klein-omni] {msg}", flush=True)


def pct(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = min(len(s) - 1, max(0, int(math.ceil((q / 100.0) * len(s))) - 1))
    return float(s[idx])


def ensure_subhost_isolation_ready() -> None:
    global _MOCK_META_PROC
    hook_c = Path("/tmp/metadata_hook.c")
    hook_so = Path("/tmp/metadata_hook.so")
    if not hook_so.exists():
        hook_c.write_text(METADATA_HOOK_C)
        subprocess.run(["gcc", "-O2", "-shared", "-fPIC", str(hook_c), "-o", str(hook_so), "-ldl"], check=True)
        log("Compiled /tmp/metadata_hook.so for GCE metadata interception")

    port80_up = False
    try:
        with socket.create_connection(("127.0.0.1", 80), timeout=0.5):
            port80_up = True
    except Exception:
        port80_up = False

    if not port80_up:
        mock_py = Path("/tmp/mock_metadata.py")
        mock_py.write_text(MOCK_METADATA_PY)
        _MOCK_META_PROC = subprocess.Popen(
            ["python3", "-u", str(mock_py)],
            stdout=open("/tmp/mock_metadata.log", "a"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        for _ in range(25):
            try:
                req = urllib.request.Request(
                    "http://127.0.0.1:80/computeMetadata/v1/instance/attributes/tpu-env",
                    headers={"Metadata-Flavor": "Google"},
                )
                with urllib.request.urlopen(req, timeout=1.0) as r:
                    if r.status == 200:
                        log("Started mock_metadata.py on 127.0.0.1:80")
                        break
            except Exception:
                time.sleep(0.2)

    for p_cand in (
        Path("/workspace/vllm-torchtpu/src/vllm_torchtpu/omni/platform.py"),
        Path("/workspace/vllm-torchtpu/vllm_torchtpu/omni/platform.py"),
    ):
        if not p_cand.exists():
            continue
        txt = p_cand.read_text()
        if "VLLM_OMNI_LOCAL_RANKS" in txt:
            continue
        start_marker = "    @classmethod\n    def set_device(cls, device: torch.device | int) -> None:\n"
        end_marker = "\n    @staticmethod\n    def _offload_dense_all_to_all() -> None:"
        if start_marker in txt and end_marker in txt:
            pre, rest = txt.split(start_marker, 1)
            _, post = rest.split(end_marker, 1)
            p_cand.write_text(pre + PATCHED_SET_DEVICE_METHOD + end_marker + post)
            py_compile.compile(str(p_cand), doraise=True)
            log(f"Patched OmniTpuPlatform.set_device in {p_cand}")


class LibtpuHbmMonitor:
    def __init__(self) -> None:
        self.ports = [8431 + i for i in range(8)]
        self.peak_bytes: dict[int, float] = {}
        self._stop = threading.Event()
        self._thr: threading.Thread | None = None

    def _poll_once(self) -> None:
        for idx, port in enumerate(self.ports):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=0.6) as r:
                    txt = r.read().decode("utf-8", errors="ignore")
                for line in txt.splitlines():
                    if line.startswith("#"):
                        continue
                    low = line.lower()
                    if ("hbm" in low or "memory" in low) and ("usage" in low or "used" in low or "allocated" in low or "bytes" in low):
                        if any(x in low for x in ("total", "limit", "capacity")):
                            continue
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            val = float(parts[-1])
                            if val > 1024 * 1024 and val > self.peak_bytes.get(idx, 0.0):
                                self.peak_bytes[idx] = val
            except Exception:
                pass

    def start(self) -> None:
        self._stop.clear()
        self.peak_bytes.clear()

        def _loop() -> None:
            while not self._stop.is_set():
                self._poll_once()
                self._stop.wait(1.5)

        self._thr = threading.Thread(target=_loop, daemon=True)
        self._thr.start()

    def snapshot_gib(self) -> float | None:
        self._poll_once()
        if not self.peak_bytes:
            return None
        return round(max(self.peak_bytes.values()) / (1024.0**3), 2)

    def stop(self) -> float | None:
        self._stop.set()
        if self._thr is not None:
            self._thr.join(timeout=2.0)
        return self.snapshot_gib()


def find_vfio_holder_pids() -> list[int]:
    my_pid = os.getpid()
    mock_pid = _MOCK_META_PROC.pid if _MOCK_META_PROC is not None else -1
    holders: set[int] = set()
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        pid = int(p.name)
        if pid in (my_pid, mock_pid, 1):
            continue
        try:
            for fd in (p / "fd").iterdir():
                try:
                    if "/dev/vfio/" in os.readlink(fd):
                        holders.add(pid)
                        break
                except Exception:
                    pass
        except Exception:
            pass
    return sorted(holders)


def stop_cluster_cleanly(procs: list[subprocess.Popen[Any]] | None = None) -> None:
    if procs:
        for proc in procs:
            if proc is not None and proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except Exception:
                    pass
        time.sleep(1.5)
        for proc in procs:
            if proc is not None and proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    pass

    my_pid = os.getpid()
    parent_pid = os.getppid()
    mock_pid = _MOCK_META_PROC.pid if _MOCK_META_PROC is not None else -1
    protected_pids = {1, my_pid, parent_pid, mock_pid}
    kill_tokens = ("/usr/local/bin/vllm", "vllm serve", "DiffusionWorker", "StageDiffusionProc", "multiprocessing.spawn")
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        pid = int(p.name)
        if pid in protected_pids:
            continue
        try:
            cmdline = (p / "cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", errors="ignore")
            if "run_flux2_klein_vllm_omni_serving.py" in cmdline or "run_phaseb" in cmdline:
                continue
            if any(tok in cmdline for tok in kill_tokens):
                os.kill(pid, signal.SIGKILL)
        except Exception:
            pass
    for lf_pat in ("/tmp/libtpu_lockfile",):
        Path(lf_pat).unlink(missing_ok=True)
    for d_root, glob_pat in (("/tmp/tpu_logs", "**/*.lock"), ("/dev/shm", "vllm_omni_*"), ("/dev/shm", "psm_*")):
        base_d = Path(d_root)
        if base_d.exists():
            for item in base_d.glob(glob_pat):
                try:
                    if item.is_file() or item.is_symlink():
                        item.unlink(missing_ok=True)
                except Exception:
                    pass
    for _ in range(12):
        holders = find_vfio_holder_pids()
        if not holders:
            break
        for pid in holders:
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass
        time.sleep(0.8)
    time.sleep(1.0)


def generate_one_image(
    endpoint: str,
    prompt: str,
    size: str,
    steps: int,
    guidance_scale: float,
    seed: int,
    timeout_s: float = 900.0,
) -> dict[str, Any]:
    exp_w, exp_h = (int(x) for x in size.split("x"))
    payload: dict[str, Any] = {
        "prompt": prompt,
        "size": size,
        "n": 1,
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "generator_device": "cpu",
        "response_format": "b64_json",
        "output_format": "png",
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{endpoint}/v1/images/generations",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw_bytes = resp.read()
        e2e_s = time.perf_counter() - t0
        data = json.loads(raw_bytes.decode("utf-8"))
        items = data.get("data") or []
        if not items or not items[0].get("b64_json"):
            return {"ok": False, "e2e_s": round(e2e_s, 4), "error": f"Missing b64_json keys={list(data.keys())}"}
        png_bytes = base64.b64decode(items[0]["b64_json"])
        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        w, h = img.size
        arr = np.asarray(img, dtype=np.float32)
        finite = bool(np.isfinite(arr).all())
        p_mean = float(np.mean(arr))
        p_std = float(np.std(arr))
        valid = w == exp_w and h == exp_h and finite and p_std > 3.0
        return {
            "ok": valid,
            "e2e_s": round(e2e_s, 4),
            "width": w,
            "height": h,
            "pixel_mean": round(p_mean, 2),
            "pixel_std": round(p_std, 2),
            "png_bytes": png_bytes,
            "error": None if valid else f"Invalid image dims=({w},{h}) std={p_std:.2f}",
        }
    except urllib.error.HTTPError as he:
        err_txt = ""
        try:
            err_txt = he.read().decode("utf-8", errors="ignore")[:500]
        except Exception:
            pass
        return {"ok": False, "e2e_s": round(time.perf_counter() - t0, 4), "error": f"HTTP {he.code}: {err_txt}"}
    except Exception as exc:
        return {"ok": False, "e2e_s": round(time.perf_counter() - t0, 4), "error": repr(exc)}


class VllmOmniCluster:
    def __init__(
        self,
        procs: list[subprocess.Popen[Any]],
        endpoints: list[str],
        log_paths: list[Path],
        boot_s: float,
        mode: str,
    ) -> None:
        self.procs = procs
        self.endpoints = endpoints
        self.log_paths = log_paths
        self.boot_s = boot_s
        self.mode = mode
        self._lock = threading.Lock()
        self._inflight = [0] * len(endpoints)
        self._rr = 0

    def _acquire(self) -> tuple[int, str]:
        with self._lock:
            n = len(self.endpoints)
            idx = min(range(n), key=lambda i: (self._inflight[i], (i - self._rr) % n))
            self._inflight[idx] += 1
            self._rr = (idx + 1) % n
            return idx, self.endpoints[idx]

    def _release(self, idx: int) -> None:
        with self._lock:
            self._inflight[idx] = max(0, self._inflight[idx] - 1)

    def generate_single(
        self,
        prompt: str,
        size: str,
        steps: int,
        gs: float,
        seed: int,
        endpoint_override: str | None = None,
    ) -> dict[str, Any]:
        if endpoint_override is not None:
            return generate_one_image(endpoint_override, prompt, size, steps, gs, seed)
        idx, ep = self._acquire()
        try:
            return generate_one_image(ep, prompt, size, steps, gs, seed)
        finally:
            self._release(idx)

    def warm_all_endpoints(self, size: str, steps: int, gs: float) -> float:
        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.endpoints)) as pool:
            futs = [
                pool.submit(self.generate_single, PROMPTS[i % len(PROMPTS)], size, steps, gs, 100 + i, ep)
                for i, ep in enumerate(self.endpoints)
            ]
            outs = [f.result() for f in futs]
        bad = [x for x in outs if not x.get("ok")]
        if bad:
            raise RuntimeError(f"Warmup failed on {len(bad)}/{len(outs)} endpoints: {bad[0].get('error')}")
        return round(time.perf_counter() - t0, 2)

    def run_closed_loop(
        self, conc: int, n_req: int, size: str, steps: int, gs: float, seed_base: int
    ) -> tuple[list[dict[str, Any]], float]:
        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as pool:
            futs = [
                pool.submit(self.generate_single, PROMPTS[i % len(PROMPTS)], size, steps, gs, seed_base + i)
                for i in range(n_req)
            ]
            outs = [f.result() for f in futs]
        return outs, time.perf_counter() - t0

    def run_poisson_rps(
        self, target_rps: float, n_req: int, size: str, steps: int, gs: float, seed_base: int
    ) -> tuple[list[dict[str, Any]], float]:
        rng = random.Random(int(target_rps * 100) + seed_base)
        ia = [rng.expovariate(target_rps) for _ in range(n_req)]
        t0 = time.perf_counter()
        target_t = t0
        futs = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(16, n_req)) as pool:
            for i, dt in enumerate(ia):
                target_t += dt
                now = time.perf_counter()
                if target_t > now:
                    time.sleep(target_t - now)
                futs.append(pool.submit(self.generate_single, PROMPTS[i % len(PROMPTS)], size, steps, gs, seed_base + i))
            outs = [f.result() for f in futs]
        return outs, time.perf_counter() - t0

    def stop(self) -> None:
        stop_cluster_cleanly(self.procs)


def _write_deploy_yaml(
    cfg_path: Path,
    ulysses_degree: int,
    max_num_seqs: int = 1,
    quantization: str | None = None,
    num_replicas: int = 1,
    devices: str | None = None,
) -> None:
    stage_cfg: dict[str, Any] = {
        "stage_id": 0,
        "ulysses_degree": ulysses_degree,
        "max_num_seqs": max_num_seqs,
        "max_generated_image_size": 16777216,
        "enable_diffusion_pipeline_profiler": True,
        "engine_extras": {
            "num_gpus": ulysses_degree,
            "request_batch_max_wait_ms": 20.0 if max_num_seqs > 1 else 0.0,
        },
    }
    if num_replicas > 1:
        stage_cfg["num_replicas"] = num_replicas
    if devices is not None:
        stage_cfg["devices"] = devices
    elif ulysses_degree == 8:
        stage_cfg["devices"] = "0,1,2,3,4,5,6,7"
    deploy_cfg: dict[str, Any] = {"stages": [stage_cfg]}
    if quantization:
        deploy_cfg["quantization"] = quantization
    cfg_path.write_text(json.dumps(deploy_cfg, indent=2) + "\n")


def _spawn_vllm_omni_proc(
    model_path: str,
    port: int,
    deploy_cfg_path: Path,
    log_path: Path,
    extra_env: dict[str, str] | None = None,
) -> subprocess.Popen[Any]:
    env = os.environ.copy()
    env.update(
        {
            "PJRT_DEVICE": "TPU",
            "TPU_SKIP_MDS_QUERY": "1",
            "TPU_PREMAPPED_BUFFER_SIZE": str(512 * 1024**2),
            "JAX_COMPILATION_CACHE_DIR": "/dev/shm/xla_cache_flux2_omni",
            "TORCH_TPU_CACHE_DIR": "/dev/shm/xla_cache_flux2_omni",
            "VLLM_DISABLE_LOG_LOGO": "1",
            "TPU_VISIBLE_CHIPS": "0,1,2,3,4,5,6,7",
        }
    )
    if Path("/tmp/metadata_hook.so").exists():
        env["LD_PRELOAD"] = "/tmp/metadata_hook.so"
    if extra_env:
        env.update(extra_env)

    cmd = [
        "vllm",
        "serve",
        model_path,
        "--omni",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--deploy-config",
        str(deploy_cfg_path),
        "--max-generated-image-size",
        "16777216",
    ]
    log(f"Spawning vllm-omni :{port} (ranks={env.get('VLLM_OMNI_LOCAL_RANKS','all-8')}): {' '.join(cmd)}")
    lf = open(log_path, "w")
    return subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)


def extract_clean_error_tail(log_path: Path, max_chars: int = 2500) -> str:
    if not log_path.exists():
        return ""
    lines = [
        ln
        for ln in log_path.read_text(errors="ignore").splitlines()
        if "coredump_hook.cc" not in ln and "Coroner client" not in ln and "process_state.cc" not in ln
    ]
    return "\n".join(lines)[-max_chars:]


def _wait_all_healthy(
    procs: list[subprocess.Popen[Any]],
    ports: list[int],
    log_paths: list[Path],
    boot_timeout_s: float = 720.0,
) -> float:
    t0 = time.time()
    ready = [False] * len(ports)
    while time.time() - t0 < boot_timeout_s:
        for idx, (proc, port, lp) in enumerate(zip(procs, ports, log_paths)):
            if proc.poll() is not None:
                raise RuntimeError(f"vllm-omni :{port} exited early (rc={proc.returncode}):\n{extract_clean_error_tail(lp)}")
            if not ready[idx]:
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2.5) as r:
                        if r.status == 200:
                            ready[idx] = True
                            log(f"vllm-omni replica {idx} healthy on :{port} in {time.time() - t0:.1f}s")
                except Exception:
                    pass
        if all(ready):
            return round(time.time() - t0, 2)
        time.sleep(2.0)
    raise TimeoutError(f"Timed out waiting for ports {[ports[i] for i, ok in enumerate(ready) if not ok]}")


def start_vllm_omni_cluster(
    model_path: str,
    base_port: int,
    dp_size: int,
    ulysses_degree: int,
    group_id: str,
    max_num_seqs: int = 1,
    quantization: str | None = None,
) -> VllmOmniCluster:
    stop_cluster_cleanly(None)
    Path("/dev/shm/xla_cache_flux2_omni").mkdir(parents=True, exist_ok=True)

    if dp_size == 8 and ulysses_degree == 1:
        procs, ports, log_paths = [], [], []
        try:
            for r in range(8):
                p_http = base_port + r * 20
                tmp_r = Path(f"/tmp/vo_f2_u1_rep{r}")
                tmp_r.mkdir(parents=True, exist_ok=True)
                cfg_p = Path(f"/tmp/deploy_{group_id}_rep{r}.yaml")
                _write_deploy_yaml(cfg_p, ulysses_degree=1, max_num_seqs=max_num_seqs, quantization=quantization, devices=str(r))
                lp = Path(f"/tmp/{group_id}_rep{r}.log")
                extra_env = {
                    "VLLM_OMNI_LOCAL_RANKS": str(r),
                    "VLLM_OMNI_SLICEBUILDER_ADDRESSES": f"localhost:{29100 + r * 10}",
                    "MASTER_PORT": str(32100 + r * 30),
                    "OMNI_MASTER_PORT": str(42100 + r * 20),
                    "TPU_RUNTIME_METRICS_PORTS": str(8431 + r),
                    "TMPDIR": str(tmp_r),
                    "TPU_VISIBLE_CHIPS": str(r),
                }
                procs.append(_spawn_vllm_omni_proc(model_path, p_http, cfg_p, lp, extra_env))
                ports.append(p_http)
                log_paths.append(lp)
                time.sleep(1.2)
            boot_s = _wait_all_healthy(procs, ports, log_paths, boot_timeout_s=600.0)
            return VllmOmniCluster(procs, [f"http://127.0.0.1:{p}" for p in ports], log_paths, boot_s, "subhost_8x_u1")
        except Exception as exc:
            log(f"subhost_8x_u1 fallback to num_replicas=8 ({str(exc)[:180]})")
            stop_cluster_cleanly(procs)
        cfg_p = Path(f"/tmp/deploy_{group_id}_nr8.yaml")
        _write_deploy_yaml(cfg_p, ulysses_degree=1, max_num_seqs=max_num_seqs, quantization=quantization, num_replicas=8, devices="0,1,2,3,4,5,6,7")
        lp = Path(f"/tmp/{group_id}_nr8.log")
        proc = _spawn_vllm_omni_proc(model_path, base_port, cfg_p, lp, {"TPU_RUNTIME_METRICS_PORTS": "8431,8432,8433,8434,8435,8436,8437,8438"})
        boot_s = _wait_all_healthy([proc], [base_port], [lp], boot_timeout_s=720.0)
        return VllmOmniCluster([proc], [f"http://127.0.0.1:{base_port}"], [lp], boot_s, "single_server_nr8_u1")

    if dp_size == 4 and ulysses_degree == 2:
        procs, ports, log_paths = [], [], []
        try:
            for r in range(4):
                p_http = base_port + r * 20
                sb0 = 29300 + r * 10
                tmp_r = Path(f"/tmp/vo_f2_u2_rep{r}")
                tmp_r.mkdir(parents=True, exist_ok=True)
                cfg_p = Path(f"/tmp/deploy_{group_id}_rep{r}.yaml")
                _write_deploy_yaml(cfg_p, ulysses_degree=2, max_num_seqs=max_num_seqs, quantization=quantization, devices=f"{2*r},{2*r+1}")
                lp = Path(f"/tmp/{group_id}_rep{r}.log")
                extra_env = {
                    "VLLM_OMNI_LOCAL_RANKS": f"{2*r},{2*r+1}",
                    "VLLM_OMNI_TOPOLOGY": "1,1,1,2",
                    "VLLM_OMNI_SLICEBUILDER_ADDRESSES": f"localhost:{sb0},localhost:{sb0+1}",
                    "MASTER_PORT": str(32300 + r * 30),
                    "OMNI_MASTER_PORT": str(42300 + r * 20),
                    "TPU_RUNTIME_METRICS_PORTS": f"{8431 + 2*r},{8432 + 2*r}",
                    "TMPDIR": str(tmp_r),
                    "TPU_VISIBLE_CHIPS": f"{2*r},{2*r+1}",
                }
                procs.append(_spawn_vllm_omni_proc(model_path, p_http, cfg_p, lp, extra_env))
                ports.append(p_http)
                log_paths.append(lp)
                time.sleep(1.5)
            boot_s = _wait_all_healthy(procs, ports, log_paths, boot_timeout_s=600.0)
            return VllmOmniCluster(procs, [f"http://127.0.0.1:{p}" for p in ports], log_paths, boot_s, "subhost_4x_u2")
        except Exception as exc:
            log(f"subhost_4x_u2 fallback to num_replicas=4 ({str(exc)[:180]})")
            stop_cluster_cleanly(procs)
        cfg_p = Path(f"/tmp/deploy_{group_id}_nr4.yaml")
        _write_deploy_yaml(cfg_p, ulysses_degree=2, max_num_seqs=max_num_seqs, quantization=quantization, num_replicas=4, devices="0,1,2,3,4,5,6,7")
        lp = Path(f"/tmp/{group_id}_nr4.log")
        proc = _spawn_vllm_omni_proc(model_path, base_port, cfg_p, lp, {"TPU_RUNTIME_METRICS_PORTS": "8431,8432,8433,8434,8435,8436,8437,8438"})
        boot_s = _wait_all_healthy([proc], [base_port], [lp], boot_timeout_s=720.0)
        return VllmOmniCluster([proc], [f"http://127.0.0.1:{base_port}"], [lp], boot_s, "single_server_nr4_u2")

    if dp_size == 2 and ulysses_degree == 4:
        procs, ports, log_paths = [], [], []
        try:
            for r in range(2):
                p_http = base_port + r * 20
                sb0 = 29400 + r * 10
                ranks_str = ",".join(str(4 * r + k) for k in range(4))
                sb_str = ",".join(f"localhost:{sb0 + k}" for k in range(4))
                mports_str = ",".join(str(8431 + 4 * r + k) for k in range(4))
                tmp_r = Path(f"/tmp/vo_f2_u4_rep{r}")
                tmp_r.mkdir(parents=True, exist_ok=True)
                cfg_p = Path(f"/tmp/deploy_{group_id}_rep{r}.yaml")
                _write_deploy_yaml(cfg_p, ulysses_degree=4, max_num_seqs=max_num_seqs, quantization=quantization, devices=ranks_str)
                lp = Path(f"/tmp/{group_id}_rep{r}.log")
                extra_env = {
                    "VLLM_OMNI_LOCAL_RANKS": ranks_str,
                    "VLLM_OMNI_TOPOLOGY": "1,2,1,2",
                    "VLLM_OMNI_SLICEBUILDER_ADDRESSES": sb_str,
                    "MASTER_PORT": str(32400 + r * 30),
                    "OMNI_MASTER_PORT": str(42400 + r * 20),
                    "TPU_RUNTIME_METRICS_PORTS": mports_str,
                    "TMPDIR": str(tmp_r),
                    "TPU_VISIBLE_CHIPS": ranks_str,
                }
                procs.append(_spawn_vllm_omni_proc(model_path, p_http, cfg_p, lp, extra_env))
                ports.append(p_http)
                log_paths.append(lp)
                time.sleep(1.5)
            boot_s = _wait_all_healthy(procs, ports, log_paths, boot_timeout_s=600.0)
            return VllmOmniCluster(procs, [f"http://127.0.0.1:{p}" for p in ports], log_paths, boot_s, "subhost_2x_u4")
        except Exception as exc:
            log(f"subhost_2x_u4 fallback to num_replicas=2 ({str(exc)[:180]})")
            stop_cluster_cleanly(procs)
        cfg_p = Path(f"/tmp/deploy_{group_id}_nr2.yaml")
        _write_deploy_yaml(cfg_p, ulysses_degree=4, max_num_seqs=max_num_seqs, quantization=quantization, num_replicas=2, devices="0,1,2,3,4,5,6,7")
        lp = Path(f"/tmp/{group_id}_nr2.log")
        proc = _spawn_vllm_omni_proc(model_path, base_port, cfg_p, lp, {"TPU_RUNTIME_METRICS_PORTS": "8431,8432,8433,8434,8435,8436,8437,8438"})
        boot_s = _wait_all_healthy([proc], [base_port], [lp], boot_timeout_s=720.0)
        return VllmOmniCluster([proc], [f"http://127.0.0.1:{base_port}"], [lp], boot_s, "single_server_nr2_u4")

    if dp_size == 1 and ulysses_degree == 8:
        cfg_p = Path(f"/tmp/deploy_{group_id}_u8.yaml")
        _write_deploy_yaml(cfg_p, ulysses_degree=8, max_num_seqs=max_num_seqs, quantization=quantization, devices="0,1,2,3,4,5,6,7")
        lp = Path(f"/tmp/{group_id}_u8.log")
        sb_str = ",".join(f"localhost:{29500 + k}" for k in range(8))
        extra_env = {
            "VLLM_OMNI_LOCAL_RANKS": "0,1,2,3,4,5,6,7",
            "VLLM_OMNI_TOPOLOGY": "2,2,1,2",
            "VLLM_OMNI_SLICEBUILDER_ADDRESSES": sb_str,
            "MASTER_PORT": "32500",
            "OMNI_MASTER_PORT": "42500",
            "TPU_RUNTIME_METRICS_PORTS": "8431,8432,8433,8434,8435,8436,8437,8438",
        }
        proc = _spawn_vllm_omni_proc(model_path, base_port, cfg_p, lp, extra_env)
        boot_s = _wait_all_healthy([proc], [base_port], [lp], boot_timeout_s=720.0)
        return VllmOmniCluster([proc], [f"http://127.0.0.1:{base_port}"], [lp], boot_s, "single_server_1x_u8")

    raise ValueError(f"Unsupported dp_size={dp_size} x ulysses_degree={ulysses_degree}")


def run_benchmark_point(
    cluster: VllmOmniCluster,
    hbm_mon: LibtpuHbmMonitor | None,
    out_jsonl: Path,
    samples_dir: Path,
    saved_samples: set[str],
    *,
    model_label: str,
    repo_id: str,
    topology_label: str,
    dp_size: int,
    ulysses_degree: int,
    precision: str,
    size: str,
    steps: int,
    gs: float,
    concurrencies: list[int],
    poisson_rps_list: list[float] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cold_s = cluster.warm_all_endpoints(size, steps, gs)

    # Save one verified sample PNG per (model, size, steps)
    sample_slug = f"{model_label.lower().replace('.', '_').replace('-', '_')}_{size}_{steps}step"
    if sample_slug not in saved_samples:
        s_out = cluster.generate_single(PROMPTS[0], size, steps, gs, 2026)
        if s_out.get("ok") and s_out.get("png_bytes"):
            samples_dir.mkdir(parents=True, exist_ok=True)
            sp = samples_dir / f"{sample_slug}.png"
            sp.write_bytes(s_out["png_bytes"])
            saved_samples.add(sample_slug)
            log(f"Saved real verified sample image -> {sp} (mean={s_out['pixel_mean']}, std={s_out['pixel_std']})")

    for conc in concurrencies:
        n_req = max(conc * 2, 4) if conc >= 4 else max(conc * 3, 3)
        if size == "2048x2048" or steps >= 20:
            n_req = max(conc, 2)
        outs, wall_s = cluster.run_closed_loop(conc, n_req, size, steps, gs, 500 + conc * 10)
        ok_outs = [x for x in outs if x.get("ok")]
        if not ok_outs:
            log(f"WARN {model_label} {topology_label} {size} {steps}st c={conc} failed: {outs[0].get('error')}")
            continue
        lats = [x["e2e_s"] for x in ok_outs]
        img_s = len(ok_outs) / max(wall_s, 1e-6)
        ipm = img_s * 60.0
        p50 = pct(lats, 50)
        p95 = pct(lats, 95)
        p99 = pct(lats, 99)
        hbm_gib = hbm_mon.snapshot_gib() if hbm_mon else None
        num_hosts = 2 if "2-Host" in topology_label else 1
        ipm_per_host = ipm / num_hosts
        rec = {
            "timestamp_utc": utc_now(),
            "engine": "vllm-omni",
            "cluster_mode": cluster.mode,
            "workload_mode": "closed_loop",
            "model_label": model_label,
            "repo_id": repo_id,
            "topology": topology_label,
            "num_hosts": num_hosts,
            "dp_size": dp_size,
            "ulysses_degree": ulysses_degree,
            "precision": precision,
            "resolution": size,
            "steps": steps,
            "guidance_scale": gs,
            "concurrency": conc,
            "num_requests": len(ok_outs),
            "cold_compile_plus_infer_s": cold_s,
            "latency_p50_s": round(p50, 3),
            "latency_p95_s": round(p95, 3),
            "latency_p99_s": round(p99, 3),
            "ms_per_step_p50": round((p50 * 1000.0) / max(steps, 1), 2),
            "throughput_img_per_s": round(img_s, 3),
            "throughput_ipm": round(ipm, 2),
            "hosts_for_60_ipm": int(math.ceil(60.0 / max(ipm_per_host, 1e-6))),
            "hosts_for_300_ipm": int(math.ceil(300.0 / max(ipm_per_host, 1e-6))),
            "hosts_for_600_ipm": int(math.ceil(600.0 / max(ipm_per_host, 1e-6))),
            "peak_hbm_gib_per_tc": hbm_gib,
            "pixel_mean": ok_outs[0]["pixel_mean"],
            "pixel_std": ok_outs[0]["pixel_std"],
            "status": "PASS",
        }
        rows.append(rec)
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with open(out_jsonl, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(rec) + "\n")
        log(
            f"[{model_label}][{topology_label}][{size} {steps}st] c={conc} -> "
            f"P50={rec['latency_p50_s']}s, P99={rec['latency_p99_s']}s, {rec['throughput_ipm']} IPM (HBM={hbm_gib} GiB)"
        )

    if poisson_rps_list:
        for rps in poisson_rps_list:
            n_req = max(6, int(rps * 4))
            outs, wall_s = cluster.run_poisson_rps(rps, n_req, size, steps, gs, 900 + int(rps * 10))
            ok_outs = [x for x in outs if x.get("ok")]
            if not ok_outs:
                continue
            lats = [x["e2e_s"] for x in ok_outs]
            ach_rps = len(ok_outs) / max(wall_s, 1e-6)
            ipm = ach_rps * 60.0
            rec = {
                "timestamp_utc": utc_now(),
                "engine": "vllm-omni",
                "cluster_mode": cluster.mode,
                "workload_mode": "open_loop_poisson",
                "model_label": model_label,
                "repo_id": repo_id,
                "topology": topology_label,
                "num_hosts": 2 if "2-Host" in topology_label else 1,
                "dp_size": dp_size,
                "ulysses_degree": ulysses_degree,
                "precision": precision,
                "resolution": size,
                "steps": steps,
                "target_rps": rps,
                "achieved_rps": round(ach_rps, 3),
                "latency_p50_s": round(pct(lats, 50), 3),
                "latency_p95_s": round(pct(lats, 95), 3),
                "latency_p99_s": round(pct(lats, 99), 3),
                "throughput_ipm": round(ipm, 2),
                "peak_hbm_gib_per_tc": hbm_mon.snapshot_gib() if hbm_mon else None,
                "status": "PASS",
            }
            rows.append(rec)
            with open(out_jsonl, "a", encoding="utf-8") as fp:
                fp.write(json.dumps(rec) + "\n")
            log(
                f"[{model_label}][{topology_label}][Poisson RPS={rps}] -> "
                f"Achieved={rec['achieved_rps']} img/s ({rec['throughput_ipm']} IPM), P50={rec['latency_p50_s']}s, P99={rec['latency_p99_s']}s"
            )
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="vLLM-Omni FLUX.2 Klein & Diffusion Serving Benchmark on TPU v7x")
    p.add_argument(
        "--suite",
        choices=["host0-main", "host1-baselines", "serve-only", "multihost-client"],
        required=True,
    )
    p.add_argument("--models-root", default="/models/models")
    p.add_argument("--shm-models-root", default="/dev/shm/flux2_models")
    p.add_argument("--output-dir", default="/tmp/flux2_klein_results")
    p.add_argument("--serve-model-path", default="")
    p.add_argument("--serve-dp", type=int, default=4)
    p.add_argument("--serve-ulysses", type=int, default=2)
    p.add_argument("--serve-base-port", type=int, default=8100)
    p.add_argument("--remote-endpoints", default="", help="Comma-separated http://ip:port endpoints across 2 hosts")
    p.add_argument("--model-label", default="FLUX.2-klein-4B")
    p.add_argument("--repo-id", default="black-forest-labs/FLUX.2-klein-4B")
    p.add_argument("--topology-label", default="2-Host DP=8 x U=2 (16 TC)")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = out_dir / "samples"
    saved_samples: set[str] = set()

    if args.suite == "multihost-client":
        eps = [x.strip() for x in args.remote_endpoints.split(",") if x.strip()]
        for ep in eps:
            deadline = time.time() + 600.0
            while time.time() < deadline:
                try:
                    with urllib.request.urlopen(f"{ep}/health", timeout=2.5) as r:
                        if r.status == 200:
                            break
                except Exception:
                    time.sleep(1.0)
            else:
                raise TimeoutError(f"Timed out waiting for {ep}/health")
        cluster = VllmOmniCluster([], eps, [], 0.0, f"multihost_{len(eps)}ep")
        out_jsonl = out_dir / "vllm_omni_multihost_results.jsonl"
        dp_sz = len(eps)
        u_deg = 16 // max(dp_sz, 1)
        run_benchmark_point(
            cluster,
            None,
            out_jsonl,
            samples_dir,
            saved_samples,
            model_label=args.model_label,
            repo_id=args.repo_id,
            topology_label=args.topology_label,
            dp_size=dp_sz,
            ulysses_degree=u_deg,
            precision="BF16",
            size="1024x1024",
            steps=4,
            gs=0.0,
            concurrencies=[1, 4, 8, 16, 32, 64],
            poisson_rps_list=[1.0, 2.0, 4.0, 8.0],
        )
        run_benchmark_point(
            cluster,
            None,
            out_jsonl,
            samples_dir,
            saved_samples,
            model_label=args.model_label,
            repo_id=args.repo_id,
            topology_label=args.topology_label,
            dp_size=dp_sz,
            ulysses_degree=u_deg,
            precision="BF16",
            size="512x512",
            steps=4,
            gs=0.0,
            concurrencies=[8, 16, 32],
        )
        return

    ensure_subhost_isolation_ready()

    if args.suite == "serve-only":
        cluster = start_vllm_omni_cluster(
            args.serve_model_path,
            args.serve_base_port,
            args.serve_dp,
            args.serve_ulysses,
            group_id=f"serve_d{args.serve_dp}_u{args.serve_ulysses}",
        )
        log(f"Serving persistently on endpoints: {cluster.endpoints}")
        while True:
            time.sleep(10.0)

    if args.suite == "host0-main":
        # Host 0: FLUX.2-klein-4B full topology x resolution x step matrix + Flux-2-Klein-9B BF16!
        out_jsonl = out_dir / "vllm_omni_host0_results.jsonl"
        out_jsonl.unlink(missing_ok=True)
        m4b = str(Path(args.shm_models_root) / "FLUX.2-klein-4B")
        m9b = str(Path(args.shm_models_root) / "Flux-2-Klein-9B")

        # 1. FLUX.2-klein-4B across DP=4xU=2 (Intra-Chip), DP=8xU=1 (Max-DP), DP=2xU=4, DP=1xU=8
        for dp_s, u_s, topo_lbl in [
            (4, 2, "1-Host DP=4 x U=2 (Intra-Chip, 8 TC)"),
            (8, 1, "1-Host DP=8 x U=1 (Max-DP, 8 TC)"),
            (2, 4, "1-Host DP=2 x U=4 (2-Chip, 8 TC)"),
            (1, 8, "1-Host DP=1 x U=8 (Low-Lat, 8 TC)"),
        ]:
            hbm = LibtpuHbmMonitor()
            hbm.start()
            cluster = start_vllm_omni_cluster(m4b, 8100, dp_s, u_s, f"k4b_d{dp_s}u{u_s}")
            try:
                if (dp_s, u_s) == (4, 2):
                    # Full resolution & step sweep on Intra-Chip sweet spot
                    for sz, st_list, concs, rps_l in [
                        ("512x512", [1, 4], [1, 4, 8, 16], None),
                        ("1024x1024", [1, 4, 8, 20], [1, 2, 4, 8, 16, 32], [1.0, 2.0, 4.0, 8.0]),
                        ("1536x1536", [4, 8], [1, 4, 8], None),
                        ("2048x2048", [4], [1, 4, 8], None),
                    ]:
                        for st in st_list:
                            c_use = concs if (sz == "1024x1024" and st == 4) else [c for c in concs if c <= 16]
                            r_use = rps_l if (sz == "1024x1024" and st == 4) else None
                            run_benchmark_point(
                                cluster,
                                hbm,
                                out_jsonl,
                                samples_dir,
                                saved_samples,
                                model_label="FLUX.2-klein-4B",
                                repo_id="black-forest-labs/FLUX.2-klein-4B",
                                topology_label=topo_lbl,
                                dp_size=dp_s,
                                ulysses_degree=u_s,
                                precision="BF16",
                                size=sz,
                                steps=st,
                                gs=0.0,
                                concurrencies=c_use,
                                poisson_rps_list=r_use,
                            )
                elif (dp_s, u_s) == (8, 1):
                    for sz, st, concs in [
                        ("512x512", 4, [1, 8, 16, 32]),
                        ("1024x1024", 1, [1, 8, 16, 32]),
                        ("1024x1024", 4, [1, 4, 8, 16, 32, 64]),
                        ("1536x1536", 4, [1, 8, 16]),
                    ]:
                        run_benchmark_point(
                            cluster,
                            hbm,
                            out_jsonl,
                            samples_dir,
                            saved_samples,
                            model_label="FLUX.2-klein-4B",
                            repo_id="black-forest-labs/FLUX.2-klein-4B",
                            topology_label=topo_lbl,
                            dp_size=dp_s,
                            ulysses_degree=u_s,
                            precision="BF16",
                            size=sz,
                            steps=st,
                            gs=0.0,
                            concurrencies=concs,
                        )
                else:
                    for sz, st, concs in [
                        ("512x512", 1, [1]),
                        ("1024x1024", 1, [1]),
                        ("1024x1024", 4, [1, 2, 4, 8] if dp_s == 2 else [1, 2, 4]),
                        ("2048x2048", 4, [1, 2] if dp_s == 2 else [1]),
                    ]:
                        run_benchmark_point(
                            cluster,
                            hbm,
                            out_jsonl,
                            samples_dir,
                            saved_samples,
                            model_label="FLUX.2-klein-4B",
                            repo_id="black-forest-labs/FLUX.2-klein-4B",
                            topology_label=topo_lbl,
                            dp_size=dp_s,
                            ulysses_degree=u_s,
                            precision="BF16",
                            size=sz,
                            steps=st,
                            gs=0.0,
                            concurrencies=concs,
                        )
            finally:
                hbm.stop()
                cluster.stop()

        # 2. Flux-2-Klein-9B BF16 on DP=4xU=2 and DP=1xU=8
        for dp_s, u_s, topo_lbl in [
            (4, 2, "1-Host DP=4 x U=2 (Intra-Chip, 8 TC)"),
            (1, 8, "1-Host DP=1 x U=8 (Low-Lat, 8 TC)"),
        ]:
            hbm = LibtpuHbmMonitor()
            hbm.start()
            cluster = start_vllm_omni_cluster(m9b, 8200, dp_s, u_s, f"k9b_d{dp_s}u{u_s}")
            try:
                pts = (
                    [
                        ("512x512", 4, [1, 4, 8]),
                        ("1024x1024", 1, [1, 4, 8]),
                        ("1024x1024", 4, [1, 4, 8, 16]),
                        ("1024x1024", 8, [1, 4, 8]),
                        ("1536x1536", 4, [1, 4]),
                        ("2048x2048", 4, [1, 4]),
                    ]
                    if dp_s == 4
                    else [
                        ("1024x1024", 1, [1]),
                        ("1024x1024", 4, [1, 4]),
                        ("2048x2048", 4, [1]),
                    ]
                )
                for sz, st, concs in pts:
                    run_benchmark_point(
                        cluster,
                        hbm,
                        out_jsonl,
                        samples_dir,
                        saved_samples,
                        model_label="Flux-2-Klein-9B",
                        repo_id="Comfy-Org/Flux-2-Klein-9B",
                        topology_label=topo_lbl,
                        dp_size=dp_s,
                        ulysses_degree=u_s,
                        precision="BF16",
                        size=sz,
                        steps=st,
                        gs=0.0,
                        concurrencies=concs,
                        poisson_rps_list=[1.0, 2.0, 4.0] if (dp_s == 4 and sz == "1024x1024" and st == 4) else None,
                    )
            finally:
                hbm.stop()
                cluster.stop()
        return

    if args.suite == "host1-baselines":
        # Host 1 runs Flux-2-Klein-9B DP=8xU=1 + Head-to-Head Baselines (FLUX.1-schnell & Qwen-Image-2-1-Turbo) in parallel!
        out_jsonl = out_dir / "vllm_omni_host1_results.jsonl"
        out_jsonl.unlink(missing_ok=True)
        m9b = str(Path(args.shm_models_root) / "Flux-2-Klein-9B")
        m_schnell = str(Path(args.models_root) / "black-forest-labs" / "FLUX.1-schnell")
        m_qwen21 = str(Path(args.models_root) / "Qwen" / "Qwen-Image-2-1-Turbo")

        # 1. Flux-2-Klein-9B DP=8xU=1 (fits easily: ~26 GiB < 95 GiB/TC!)
        hbm = LibtpuHbmMonitor()
        hbm.start()
        try:
            cluster = start_vllm_omni_cluster(m9b, 8300, 8, 1, "k9b_d8u1")
            try:
                for sz, st, concs in [
                    ("1024x1024", 4, [1, 8, 16, 32]),
                    ("1536x1536", 4, [1, 8]),
                ]:
                    run_benchmark_point(
                        cluster,
                        hbm,
                        out_jsonl,
                        samples_dir,
                        saved_samples,
                        model_label="Flux-2-Klein-9B",
                        repo_id="Comfy-Org/Flux-2-Klein-9B",
                        topology_label="1-Host DP=8 x U=1 (Max-DP, 8 TC)",
                        dp_size=8,
                        ulysses_degree=1,
                        precision="BF16",
                        size=sz,
                        steps=st,
                        gs=0.0,
                        concurrencies=concs,
                    )
            finally:
                cluster.stop()
        except Exception as exc:
            log(f"Flux-2-Klein-9B DP=8xU=1 note: {exc}")
        finally:
            hbm.stop()

        # 2. Head-to-head baseline: FLUX.1-schnell (12B) on DP=4xU=2
        hbm = LibtpuHbmMonitor()
        hbm.start()
        try:
            cluster = start_vllm_omni_cluster(m_schnell, 8400, 4, 2, "schnell_d4u2")
            try:
                for sz, st, concs in [
                    ("512x512", 4, [1, 4, 8]),
                    ("1024x1024", 1, [1, 4, 8]),
                    ("1024x1024", 4, [1, 4, 8, 16]),
                    ("1536x1536", 4, [1, 4]),
                    ("2048x2048", 4, [1, 4]),
                ]:
                    run_benchmark_point(
                        cluster,
                        hbm,
                        out_jsonl,
                        samples_dir,
                        saved_samples,
                        model_label="FLUX.1-schnell",
                        repo_id="black-forest-labs/FLUX.1-schnell",
                        topology_label="1-Host DP=4 x U=2 (Intra-Chip, 8 TC)",
                        dp_size=4,
                        ulysses_degree=2,
                        precision="BF16",
                        size=sz,
                        steps=st,
                        gs=0.0,
                        concurrencies=concs,
                    )
            finally:
                cluster.stop()
        finally:
            hbm.stop()

        # 3. Head-to-head baseline: Qwen/Qwen-Image-2-1-Turbo (8-step distilled) on DP=4xU=2
        # Stage to /dev/shm first to eliminate 8-way GCS FUSE shard read contention across 750+ tensor slices.
        from prepare_flux2_klein_9b_diffusers import fast_copy_tree

        shm_qwen21 = Path(args.shm_models_root) / "Qwen-Image-2-1-Turbo"
        fast_copy_tree(Path(m_qwen21), shm_qwen21)
        hbm = LibtpuHbmMonitor()
        hbm.start()
        try:
            cluster = start_vllm_omni_cluster(str(shm_qwen21), 8500, 4, 2, "qwen21_d4u2")
            try:
                for sz, st, concs in [
                    ("512x512", 8, [1, 4, 8]),
                    ("1024x1024", 4, [1, 4, 8]),
                    ("1024x1024", 8, [1, 4, 8, 16]),
                ]:
                    run_benchmark_point(
                        cluster,
                        hbm,
                        out_jsonl,
                        samples_dir,
                        saved_samples,
                        model_label="Qwen-Image-2.1-Turbo",
                        repo_id="Qwen/Qwen-Image-2-1-Turbo",
                        topology_label="1-Host DP=4 x U=2 (Intra-Chip, 8 TC)",
                        dp_size=4,
                        ulysses_degree=2,
                        precision="BF16",
                        size=sz,
                        steps=st,
                        gs=0.0,
                        concurrencies=concs,
                    )
            finally:
                cluster.stop()
        except Exception as exc:
            log(f"Qwen-Image-2.1-Turbo vllm-omni note: {exc}")
        finally:
            hbm.stop()
        return


if __name__ == "__main__":
    main()
