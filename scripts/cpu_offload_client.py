#!/usr/bin/env python3
"""Client helper for `multimodal-cpu-offload-svc` (`omni-cpu-offload-pool` / `video-encode-cpu-pool`).

Provides both:
  1. Disaggregated remote AVX-512 CPU offload (`POST /encode`, `/encode_image`, `/encode_wav`, `/preprocess_vision`)
  2. Local TPU-host CPU baseline timing (for clean A/B ablation tables in `APPENDIX.md`)
"""

from __future__ import annotations

import io
import json
import os
import struct
import subprocess
import tempfile
import time
import urllib.request
import wave
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

DEFAULT_OFFLOAD_BASE_URL = os.environ.get(
    "MULTIMODAL_CPU_OFFLOAD_URL",
    "http://multimodal-cpu-offload-svc.default.svc.cluster.local:8090",
).rstrip("/")


import http.client
import socket
import threading
import urllib.parse

ADAPTIVE_OFFLOAD_MIN_BYTES = int(os.environ.get("ADAPTIVE_OFFLOAD_MIN_BYTES", "262144"))
_TLS_CONN = threading.local()


def _get_keepalive_conn(host: str, port: int, timeout: float) -> http.client.HTTPConnection:
    key = f"{host}:{port}"
    conns: dict[str, http.client.HTTPConnection] = getattr(_TLS_CONN, "conns", None) or {}
    _TLS_CONN.conns = conns
    conn = conns.get(key)
    if conn is None or conn.sock is None:
        conn = http.client.HTTPConnection(host, port, timeout=timeout)
        conn.connect()
        if conn.sock is not None:
            conn.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        conns[key] = conn
    return conn


def _post_framed(endpoint: str, meta: dict[str, Any], payload: bytes | memoryview, timeout: float = 120.0) -> tuple[bytes, dict[str, str], float]:
    parsed = urllib.parse.urlsplit(endpoint)
    host = parsed.hostname or "localhost"
    port = parsed.port or 80
    path = parsed.path or "/"
    meta_b = json.dumps(meta).encode("utf-8")
    total = 4 + len(meta_b) + len(payload)
    buf = bytearray(total)
    struct.pack_into(">I", buf, 0, len(meta_b))
    buf[4 : 4 + len(meta_b)] = meta_b
    buf[4 + len(meta_b) :] = payload
    hdrs_out = {
        "Content-Type": "application/octet-stream",
        "Content-Length": str(total),
        "Connection": "keep-alive",
    }
    t0 = time.perf_counter()
    for attempt in range(2):
        try:
            conn = _get_keepalive_conn(host, port, timeout)
            conn.request("POST", path, body=memoryview(buf), headers=hdrs_out)
            resp = conn.getresponse()
            out = resp.read()
            if resp.status >= 400:
                raise RuntimeError(f"HTTP {resp.status}: {out[:200]!r}")
            hdrs = {k.lower(): v for k, v in resp.getheaders()}
            return out, hdrs, time.perf_counter() - t0
        except Exception:
            conns = getattr(_TLS_CONN, "conns", {})
            old = conns.pop(f"{host}:{port}", None)
            if old is not None:
                try:
                    old.close()
                except Exception:
                    pass
            if attempt == 1:
                raise
    raise RuntimeError("unreachable")


def check_offload_health(base_url: str = DEFAULT_OFFLOAD_BASE_URL) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=3.0) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return {"status": "unreachable", "error": str(exc)[:200]}


def benchmark_video_encode_ablation(
    width: int,
    height: int,
    num_frames: int,
    fps: int = 24,
    with_stereo_audio: bool = True,
    audio_sr: int = 48000,
    base_url: str = DEFAULT_OFFLOAD_BASE_URL,
) -> dict[str, Any]:
    """Measures local TPU-host CPU ffmpeg encode vs disaggregated `multimodal-cpu-offload-svc` `/encode`."""
    rng = np.random.default_rng(42)
    vid = rng.integers(0, 255, size=(num_frames, height, width, 3), dtype=np.uint8)
    vid_bytes = vid.tobytes()
    aud_bytes = b""
    if with_stereo_audio:
        n_samp = int((num_frames / max(fps, 1)) * audio_sr)
        aud = (rng.standard_normal((n_samp, 2), dtype=np.float32) * 0.1).astype(np.float32)
        aud_bytes = aud.tobytes()

    # 1. Local on-host CPU ffmpeg encode (constrained to 4 threads typical of busy TPU worker pod)
    t_loc0 = time.perf_counter()
    with tempfile.TemporaryDirectory(dir="/dev/shm", prefix="loc_venc_") as td:
        mp4_p = os.path.join(td, "out.mp4")
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0",
        ]
        if aud_bytes:
            af_p = os.path.join(td, "aud.f32le")
            Path(af_p).write_bytes(aud_bytes)
            cmd.extend(["-f", "f32le", "-ar", str(audio_sr), "-ac", "2", "-i", af_p])
        cmd.extend(["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-threads", "4"])
        if aud_bytes:
            cmd.extend(["-c:a", "aac", "-b:a", "192k", "-shortest"])
        cmd.append(mp4_p)
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        assert p.stdin is not None
        p.communicate(input=vid_bytes, timeout=180)
    local_ms = (time.perf_counter() - t_loc0) * 1000.0

    # 2. Remote disaggregated AVX-512 CPU offload (`c3-standard-44`)
    meta = {
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "fps": fps,
        "preset": "ultrafast",
        "crf": 20,
        "threads": 16,
        "audio_sr": audio_sr if aud_bytes else 0,
        "audio_channels": 2,
        "audio_bytes": len(aud_bytes),
        "return_mp4": False,
    }
    try:
        resp_b, _, rtt_s = _post_framed(f"{base_url}/encode", meta, vid_bytes + aud_bytes, timeout=180.0)
        info = json.loads(resp_b.decode("utf-8"))
        return {
            "offload_ok": True,
            "local_cpu_encode_ms": round(local_ms, 2),
            "disagg_cpu_rtt_ms": round(rtt_s * 1000.0, 2),
            "disagg_cpu_ffmpeg_ms": info.get("encode_ms"),
            "disagg_encode_fps": info.get("encode_fps"),
            "encoder_host": info.get("hostname"),
            "mp4_bytes": info.get("mp4_bytes"),
            "cpu_offload_speedup_x": round(local_ms / max(rtt_s * 1000.0, 1e-3), 2),
        }
    except Exception as exc:
        return {
            "offload_ok": False,
            "local_cpu_encode_ms": round(local_ms, 2),
            "error": str(exc)[:240],
        }


def benchmark_image_encode_ablation(
    width: int,
    height: int,
    batch: int = 1,
    fmt: str = "png",
    base_url: str = DEFAULT_OFFLOAD_BASE_URL,
) -> dict[str, Any]:
    rng = np.random.default_rng(7)
    arr = rng.integers(0, 255, size=(batch, height, width, 3), dtype=np.uint8)
    raw = arr.tobytes()

    t0 = time.perf_counter()
    for i in range(batch):
        im = Image.fromarray(arr[i], mode="RGB")
        bio = io.BytesIO()
        im.save(bio, format="PNG")
    local_ms = (time.perf_counter() - t0) * 1000.0

    try:
        resp_b, _, rtt_s = _post_framed(
            f"{base_url}/encode_image",
            {"width": width, "height": height, "batch": batch, "fmt": fmt, "return_bytes": False},
            raw,
            timeout=60.0,
        )
        info = json.loads(resp_b.decode("utf-8"))
        return {
            "offload_ok": True,
            "local_cpu_png_ms": round(local_ms, 2),
            "disagg_cpu_rtt_ms": round(rtt_s * 1000.0, 2),
            "disagg_cpu_encode_ms": info.get("encode_ms"),
            "disagg_images_per_s": info.get("images_per_s"),
            "encoder_host": info.get("hostname"),
            "cpu_offload_speedup_x": round(local_ms / max(rtt_s * 1000.0, 1e-3), 2),
        }
    except Exception as exc:
        return {"offload_ok": False, "local_cpu_png_ms": round(local_ms, 2), "error": str(exc)[:240]}


def benchmark_audio_wav_ablation(
    audio_seconds: float = 2.0,
    batch: int = 1,
    sample_rate: int = 24000,
    base_url: str = DEFAULT_OFFLOAD_BASE_URL,
) -> dict[str, Any]:
    n_samp = int(audio_seconds * sample_rate * batch)
    pcm = (np.random.default_rng(11).standard_normal(n_samp, dtype=np.float32) * 0.1).tobytes()

    t0 = time.perf_counter()
    s16 = np.clip(np.frombuffer(pcm, dtype=np.float32) * 32767.0, -32768.0, 32767.0).astype(np.int16)
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(s16.tobytes())
    local_ms = (time.perf_counter() - t0) * 1000.0

    try:
        resp_b, _, rtt_s = _post_framed(
            f"{base_url}/encode_wav",
            {"sample_rate": sample_rate, "channels": 1, "return_bytes": False},
            pcm,
            timeout=30.0,
        )
        info = json.loads(resp_b.decode("utf-8"))
        return {
            "offload_ok": True,
            "local_cpu_wav_ms": round(local_ms, 3),
            "disagg_cpu_rtt_ms": round(rtt_s * 1000.0, 3),
            "disagg_cpu_encode_ms": info.get("encode_ms"),
            "encoder_host": info.get("hostname"),
        }
    except Exception as exc:
        return {"offload_ok": False, "local_cpu_wav_ms": round(local_ms, 3), "error": str(exc)[:240]}


def benchmark_vision_preproc_ablation(
    num_cams: int = 2,
    src_w: int = 640,
    src_h: int = 480,
    dst_w: int = 256,
    dst_h: int = 256,
    base_url: str = DEFAULT_OFFLOAD_BASE_URL,
) -> dict[str, Any]:
    rng = np.random.default_rng(19)
    arr = rng.integers(0, 255, size=(num_cams, src_h, src_w, 3), dtype=np.uint8)
    raw = arr.tobytes()

    t0 = time.perf_counter()
    outs = []
    for i in range(num_cams):
        im = Image.fromarray(arr[i], mode="RGB").resize((dst_w, dst_h), Image.Resampling.BICUBIC)
        f32 = (np.asarray(im, dtype=np.float32) / 127.5) - 1.0
        outs.append(np.transpose(f32, (2, 0, 1)).astype(np.float16))
    _ = np.stack(outs, axis=0)
    local_ms = (time.perf_counter() - t0) * 1000.0

    try:
        _, hdrs, rtt_s = _post_framed(
            f"{base_url}/preprocess_vision",
            {"num_cams": num_cams, "src_w": src_w, "src_h": src_h, "dst_w": dst_w, "dst_h": dst_h, "return_tensor": True},
            raw,
            timeout=30.0,
        )
        return {
            "offload_ok": True,
            "local_cpu_preproc_ms": round(local_ms, 2),
            "disagg_cpu_rtt_ms": round(rtt_s * 1000.0, 2),
            "disagg_cpu_preproc_ms": float(hdrs.get("x-preproc-time-ms", "0") or 0),
            "encoder_host": hdrs.get("x-encoder-host", ""),
            "cpu_offload_speedup_x": round(local_ms / max(rtt_s * 1000.0, 1e-3), 2),
        }
    except Exception as exc:
        return {"offload_ok": False, "local_cpu_preproc_ms": round(local_ms, 2), "error": str(exc)[:240]}
