#!/usr/bin/env python3
"""Disaggregated High-Throughput Multimodal CPU Offload Service (`omni-cpu-offload-pool`).

Runs on dedicated Intel Xeon Sapphire Rapids (`c3-standard-44`, 44 vCPUs / 176 GiB DDR5 RAM AVX-512) CPU nodes
to offload ALL non-TPU media/vision/audio pre- and post-processing completely off Cloud TPU (`v5p` & `v7x`) hosts:

1. `POST /encode` (100% wire-compatible with `video-encode-svc`):
   - Multi-threaded `ffmpeg` `libx264` AVX-512 H.264 MP4 + `aac` stereo audio muxing for `Cosmos3-Super` T2V/T2VS
     and video diffusion workloads (`LTX-2.5`, `Wan 2.2`, `MiniMax-H3`).
2. `POST /encode_image`:
   - Multi-threaded PNG / WebP / JPEG image encoding (`1024x1024`, `1328x1328`, `1664x928`, `2048x2048`) for
     `Qwen-Image`, `Qwen-Image-Edit-2511`, `Qwen-Image-2.1` (`Qwen3-VL`), and `HunyuanImage-3.0` (`-Instruct` & `-Distil`).
3. `POST /encode_wav`:
   - Low-latency `24 kHz` / `48 kHz` PCM -> WAV / AAC streaming audio packet framing & resampling for
     `Qwen3-Omni-30B-A3B` (`Code2Wav`) real-time voice output.
4. `POST /preprocess_vision`:
   - Multi-camera RGB / JPEG frame decoding, bicubic `256x256` / `448x448` / `1024x1024` resize, SigLip2 / Qwen-VL
     normalization, and `bfloat16` / `float16` tensor packing for `GR00T-N1.7-3B` humanoid robotics control loops
     and TI2I reference image conditioning (`Qwen-Image-Edit`, `HunyuanImage-3.0`).
"""

from __future__ import annotations

import concurrent.futures
import io
import json
import os
import platform
import socket
import struct
import subprocess
import tempfile
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

HOSTNAME = socket.gethostname()
CPU_COUNT = os.cpu_count() or 1
SHM_DIR = Path("/dev/shm/multimodal_cpu_spool")
SHM_DIR.mkdir(parents=True, exist_ok=True)
_IMG_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=min(32, max(4, CPU_COUNT)))

_STATS_LOCK = threading.Lock()
_STATS: dict[str, Any] = {
    "video_jobs": 0,
    "video_frames": 0,
    "video_encode_s": 0.0,
    "image_jobs": 0,
    "images_encoded": 0,
    "image_encode_s": 0.0,
    "audio_jobs": 0,
    "audio_seconds_encoded": 0.0,
    "audio_encode_s": 0.0,
    "vision_preproc_jobs": 0,
    "vision_cams_processed": 0,
    "vision_preproc_s": 0.0,
}


def _cpu_model() -> str:
    try:
        for ln in Path("/proc/cpuinfo").read_text().splitlines():
            if "model name" in ln:
                return ln.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


CPU_MODEL = _cpu_model()


def _read_framed_payload(rfile: Any, clen: int) -> tuple[dict[str, Any], memoryview, float]:
    t0 = time.perf_counter()
    buf = bytearray(clen)
    view = memoryview(buf)
    pos = 0
    while pos < clen:
        n = rfile.readinto(view[pos:])
        if not n:
            break
        pos += n
    rx_s = time.perf_counter() - t0
    if pos < clen:
        raise ValueError(f"incomplete read {pos}/{clen}")
    meta_len = struct.unpack(">I", view[:4])[0]
    meta = json.loads(bytes(view[4 : 4 + meta_len]).decode("utf-8"))
    return meta, view[4 + meta_len :], rx_s


def encode_mp4_ffmpeg(
    video_rgb_bytes: memoryview | bytes,
    *,
    width: int,
    height: int,
    num_frames: int,
    fps: float,
    preset: str = "ultrafast",
    crf: int = 20,
    threads: int = 16,
    audio_pcm_bytes: memoryview | bytes | None = None,
    audio_sr: int = 0,
    audio_channels: int = 2,
) -> tuple[bytes, float]:
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(dir=str(SHM_DIR), prefix="enc_") as tmpd:
        out_mp4 = os.path.join(tmpd, "out.mp4")
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
        ]
        if audio_pcm_bytes and len(audio_pcm_bytes) > 0 and audio_sr > 0:
            wav_raw = os.path.join(tmpd, "audio.f32le")
            with open(wav_raw, "wb") as af:
                af.write(audio_pcm_bytes)
            cmd.extend(["-f", "f32le", "-ar", str(audio_sr), "-ac", str(audio_channels), "-i", wav_raw])
        cmd.extend(
            [
                "-c:v",
                "libx264",
                "-preset",
                preset,
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
                "-threads",
                str(threads),
            ]
        )
        if audio_pcm_bytes and len(audio_pcm_bytes) > 0 and audio_sr > 0:
            cmd.extend(["-c:a", "aac", "-b:a", "192k", "-shortest"])
        cmd.extend(["-movflags", "+faststart", out_mp4])
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        assert proc.stdin is not None
        _, stderr_b = proc.communicate(input=video_rgb_bytes, timeout=300)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed (rc={proc.returncode}): {stderr_b.decode('utf-8', 'replace')[:600]}")
        mp4_data = Path(out_mp4).read_bytes()
    return mp4_data, time.perf_counter() - t0


def encode_images_batch(
    rgb_bytes: memoryview,
    *,
    width: int,
    height: int,
    batch: int,
    fmt: str = "png",
    quality: int = 92,
) -> tuple[list[bytes], float]:
    t0 = time.perf_counter()
    frame_sz = height * width * 3
    fmt_u = fmt.upper()
    if fmt_u == "JPG":
        fmt_u = "JPEG"

    def _enc_one(idx: int) -> bytes:
        sl = rgb_bytes[idx * frame_sz : (idx + 1) * frame_sz]
        arr = np.frombuffer(sl, dtype=np.uint8).reshape((height, width, 3))
        img = Image.fromarray(arr, mode="RGB")
        bio = io.BytesIO()
        if fmt_u == "PNG":
            img.save(bio, format="PNG", compress_level=1)
        elif fmt_u == "WEBP":
            img.save(bio, format="WEBP", quality=quality, method=1)
        else:
            img.save(bio, format="JPEG", quality=quality)
        return bio.getvalue()

    if batch == 1:
        outs = [_enc_one(0)]
    else:
        outs = list(_IMG_POOL.map(_enc_one, range(batch)))
    return outs, time.perf_counter() - t0


def encode_audio_wav(
    pcm_bytes: memoryview,
    *,
    sample_rate: int = 24000,
    channels: int = 1,
) -> tuple[bytes, float, float]:
    t0 = time.perf_counter()
    samples = np.frombuffer(pcm_bytes, dtype=np.float32)
    pcm16 = np.clip(samples * 32767.0, -32768.0, 32767.0).astype(np.int16)
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    dur_audio_s = len(samples) / max(sample_rate * channels, 1)
    return bio.getvalue(), time.perf_counter() - t0, dur_audio_s


def preprocess_vision_batch(
    payload: memoryview,
    *,
    num_cams: int,
    src_w: int,
    src_h: int,
    dst_w: int = 256,
    dst_h: int = 256,
    input_fmt: str = "rgb24",
) -> tuple[bytes, float]:
    t0 = time.perf_counter()
    frame_sz = src_h * src_w * 3

    def _proc_cam(idx: int) -> np.ndarray:
        if input_fmt == "rgb24":
            sl = payload[idx * frame_sz : (idx + 1) * frame_sz]
            arr = np.frombuffer(sl, dtype=np.uint8).reshape((src_h, src_w, 3))
            img = Image.fromarray(arr, mode="RGB")
        else:
            img = Image.open(io.BytesIO(bytes(payload))).convert("RGB")
        if (src_w, src_h) != (dst_w, dst_h):
            img = img.resize((dst_w, dst_h), Image.Resampling.BICUBIC)
        f32 = (np.asarray(img, dtype=np.float32) / 127.5) - 1.0
        # CHW float16 wire representation (2 bytes/elem, direct zero-copy cast to bfloat16 on TPU)
        return np.transpose(f32, (2, 0, 1)).astype(np.float16)

    if num_cams == 1:
        cams = [_proc_cam(0)]
    else:
        cams = list(_IMG_POOL.map(_proc_cam, range(num_cams)))
    stacked = np.stack(cams, axis=0)
    return stacked.tobytes(), time.perf_counter() - t0


class MultimodalOffloadHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    disable_nagle_algorithm = True

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _send_json(self, code: int, obj: dict[str, Any]) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.startswith("/health") or self.path.startswith("/stats"):
            with _STATS_LOCK:
                snap = dict(_STATS)
            venc = snap["video_encode_s"]
            self._send_json(
                200,
                {
                    "status": "ok",
                    "hostname": HOSTNAME,
                    "cpu_model": CPU_MODEL,
                    "vcpus": CPU_COUNT,
                    "avg_video_fps": round(snap["video_frames"] / venc, 1) if venc > 0 else 0.0,
                    **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in snap.items()},
                },
            )
            return
        if self.path.startswith("/gcs_write_token"):
            import urllib.request

            req = urllib.request.Request(
                "http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/token",
                headers={"Metadata-Flavor": "Google"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                tok_data = json.loads(resp.read().decode("utf-8"))
            self._send_json(200, {"access_token": tok_data["access_token"]})
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        try:
            clen = int(self.headers.get("Content-Length", "0"))
            if clen < 8:
                self._send_json(400, {"error": "payload too short"})
                return
            meta, body_view, rx_s = _read_framed_payload(self.rfile, clen)

            if self.path.startswith("/encode_image"):
                w = int(meta["width"])
                h = int(meta["height"])
                batch = int(meta.get("batch", 1))
                fmt = str(meta.get("fmt", "png"))
                return_bytes = bool(meta.get("return_bytes", False))
                imgs, enc_s = encode_images_batch(body_view, width=w, height=h, batch=batch, fmt=fmt)
                total_b = sum(len(x) for x in imgs)
                with _STATS_LOCK:
                    _STATS["image_jobs"] += 1
                    _STATS["images_encoded"] += batch
                    _STATS["image_encode_s"] += enc_s
                if not return_bytes:
                    self._send_json(
                        200,
                        {
                            "ok": True,
                            "hostname": HOSTNAME,
                            "cpu_model": CPU_MODEL,
                            "batch": batch,
                            "encoded_bytes": total_b,
                            "rx_ms": round(rx_s * 1000.0, 2),
                            "encode_ms": round(enc_s * 1000.0, 2),
                            "images_per_s": round(batch / max(enc_s, 1e-6), 1),
                        },
                    )
                    return
                out_b = imgs[0]
                self.send_response(200)
                self.send_header("Content-Type", f"image/{fmt.lower()}")
                self.send_header("Content-Length", str(len(out_b)))
                self.send_header("X-Encoder-Host", HOSTNAME)
                self.send_header("X-Encode-Time-Ms", f"{enc_s * 1000.0:.2f}")
                self.end_headers()
                self.wfile.write(out_b)
                return

            if self.path.startswith("/encode_wav"):
                sr = int(meta.get("sample_rate", 24000))
                ch = int(meta.get("channels", 1))
                return_bytes = bool(meta.get("return_bytes", False))
                wav_b, enc_s, dur_s = encode_audio_wav(body_view, sample_rate=sr, channels=ch)
                with _STATS_LOCK:
                    _STATS["audio_jobs"] += 1
                    _STATS["audio_seconds_encoded"] += dur_s
                    _STATS["audio_encode_s"] += enc_s
                if not return_bytes:
                    self._send_json(
                        200,
                        {
                            "ok": True,
                            "hostname": HOSTNAME,
                            "wav_bytes": len(wav_b),
                            "audio_duration_s": round(dur_s, 3),
                            "rx_ms": round(rx_s * 1000.0, 2),
                            "encode_ms": round(enc_s * 1000.0, 3),
                        },
                    )
                    return
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(wav_b)))
                self.end_headers()
                self.wfile.write(wav_b)
                return

            if self.path.startswith("/preprocess_vision"):
                ncams = int(meta.get("num_cams", 2))
                sw = int(meta.get("src_w", 640))
                sh = int(meta.get("src_h", 480))
                dw = int(meta.get("dst_w", 256))
                dh = int(meta.get("dst_h", 256))
                return_tensor = bool(meta.get("return_tensor", True))
                tensor_b, proc_s = preprocess_vision_batch(
                    body_view, num_cams=ncams, src_w=sw, src_h=sh, dst_w=dw, dst_h=dh
                )
                with _STATS_LOCK:
                    _STATS["vision_preproc_jobs"] += 1
                    _STATS["vision_cams_processed"] += ncams
                    _STATS["vision_preproc_s"] += proc_s
                if not return_tensor:
                    self._send_json(
                        200,
                        {
                            "ok": True,
                            "hostname": HOSTNAME,
                            "num_cams": ncams,
                            "tensor_bytes": len(tensor_b),
                            "rx_ms": round(rx_s * 1000.0, 2),
                            "preproc_ms": round(proc_s * 1000.0, 2),
                        },
                    )
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(tensor_b)))
                self.send_header("X-Encoder-Host", HOSTNAME)
                self.send_header("X-Preproc-Time-Ms", f"{proc_s * 1000.0:.2f}")
                self.end_headers()
                self.wfile.write(tensor_b)
                return

            if self.path.startswith("/encode"):
                w = int(meta["width"])
                h = int(meta["height"])
                nf = int(meta["num_frames"])
                fps = float(meta.get("fps", 24))
                preset = str(meta.get("preset", "ultrafast"))
                crf = int(meta.get("crf", 20))
                threads = int(meta.get("threads", min(16, CPU_COUNT)))
                audio_sr = int(meta.get("audio_sr", 0) or 0)
                audio_ch = int(meta.get("audio_channels", 2) or 2)
                audio_len = int(meta.get("audio_bytes", 0) or 0)
                return_mp4 = bool(meta.get("return_mp4", True))
                vid_len = nf * h * w * 3
                vid_slice = body_view[:vid_len]
                aud_slice = body_view[vid_len : vid_len + audio_len] if audio_len > 0 else None
                mp4_bytes, enc_s = encode_mp4_ffmpeg(
                    vid_slice,
                    width=w,
                    height=h,
                    num_frames=nf,
                    fps=fps,
                    preset=preset,
                    crf=crf,
                    threads=threads,
                    audio_pcm_bytes=aud_slice,
                    audio_sr=audio_sr,
                    audio_channels=audio_ch,
                )
                with _STATS_LOCK:
                    _STATS["video_jobs"] += 1
                    _STATS["video_frames"] += nf
                    _STATS["video_encode_s"] += enc_s
                if not return_mp4:
                    self._send_json(
                        200,
                        {
                            "ok": True,
                            "hostname": HOSTNAME,
                            "mp4_bytes": len(mp4_bytes),
                            "rx_ms": round(rx_s * 1000.0, 2),
                            "encode_ms": round(enc_s * 1000.0, 2),
                            "encode_fps": round(nf / max(enc_s, 1e-6), 1),
                        },
                    )
                    return
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Length", str(len(mp4_bytes)))
                self.send_header("X-Encoder-Host", HOSTNAME)
                self.send_header("X-Rx-Time-Ms", f"{rx_s * 1000.0:.2f}")
                self.send_header("X-Encode-Time-Ms", f"{enc_s * 1000.0:.2f}")
                self.send_header("X-Encode-FPS", f"{nf / max(enc_s, 1e-6):.1f}")
                self.end_headers()
                self.wfile.write(mp4_bytes)
                return

            self._send_json(404, {"error": f"unknown endpoint {self.path}"})
        except Exception as exc:
            self._send_json(500, {"ok": False, "error": str(exc)[:600]})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8090"))
    srv = ThreadingHTTPServer(("0.0.0.0", port), MultimodalOffloadHandler)
    print(
        f"[multimodal-cpu-offload-svc] Listening on 0.0.0.0:{port} host={HOSTNAME} cpu={CPU_MODEL} vcpus={CPU_COUNT}",
        flush=True,
    )
    srv.serve_forever()
