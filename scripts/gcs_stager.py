#!/usr/bin/env python3
"""Cooperative Multi-Host Zero-Disk HuggingFace -> Regional GCS Model Stager + Safe /dev/shm Loader."""

from __future__ import annotations

import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from huggingface_hub import HfApi, hf_hub_download
import requests


def utc_iso() -> str:
    import datetime as dt
    return dt.datetime.now(dt.timezone.utc).isoformat()


def vm_access_token() -> str:
    req = urllib.request.Request(
        "http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/token",
        headers={"Metadata-Flavor": "Google"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))["access_token"]


def vm_write_token() -> str:
    """Returns a GCS write-capable token, bridging via multimodal-cpu-offload-svc if host VM has storage-ro scope."""
    try:
        sreq = urllib.request.Request(
            "http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/scopes",
            headers={"Metadata-Flavor": "Google"},
        )
        with urllib.request.urlopen(sreq, timeout=5) as sresp:
            scopes = sresp.read().decode("utf-8")
        if any(w in scopes for w in ("cloud-platform", "devstorage.read_write", "devstorage.full_control")):
            return vm_access_token()
    except Exception:
        pass

    offload_base = os.environ.get(
        "MULTIMODAL_CPU_OFFLOAD_URL",
        "http://multimodal-cpu-offload-svc.default.svc.cluster.local:8090",
    ).rstrip("/")
    try:
        treq = urllib.request.Request(f"{offload_base}/gcs_write_token")
        with urllib.request.urlopen(treq, timeout=10) as tresp:
            return json.loads(tresp.read().decode("utf-8"))["access_token"]
    except Exception:
        return vm_access_token()


def gcs_get_metadata(bucket: str, obj: str, tok: str | None = None) -> dict[str, Any] | None:
    tok = tok or vm_access_token()
    url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{urllib.parse.quote(obj, safe='')}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def gcs_list_objects(bucket: str, prefix: str) -> dict[str, int]:
    tok = vm_access_token()
    out: dict[str, int] = {}
    page_token = None
    while True:
        q = f"prefix={urllib.parse.quote(prefix, safe='')}&maxResults=1000"
        if page_token:
            q += f"&pageToken={urllib.parse.quote(page_token, safe='')}"
        url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o?{q}"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        for item in data.get("items", []):
            out[item["name"]] = int(item.get("size", 0))
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return out


def gcs_upload_bytes(bucket: str, obj: str, payload: bytes, content_type: str = "application/json") -> None:
    for attempt in range(4):
        try:
            tok = vm_write_token()
            url = (
                f"https://storage.googleapis.com/upload/storage/v1/b/{bucket}/o"
                f"?uploadType=media&name={urllib.parse.quote(obj, safe='')}"
            )
            req = urllib.request.Request(
                url,
                data=payload,
                method="POST",
                headers={
                    "Authorization": f"Bearer {tok}",
                    "Content-Type": content_type,
                    "Content-Length": str(len(payload)),
                },
            )
            with urllib.request.urlopen(req, timeout=120):
                return
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))


def _stream_hf_file_to_gcs(
    repo_id: str,
    rfilename: str,
    expected_size: int,
    bucket: str,
    dst_prefix: str,
    hf_token: str | None,
) -> tuple[str, str, int]:
    obj_name = f"{dst_prefix.rstrip('/')}/{rfilename}"
    meta = gcs_get_metadata(bucket, obj_name)
    if meta is not None and (expected_size <= 0 or int(meta.get("size", -1)) == expected_size):
        return (rfilename, "SKIPPED_EXISTS", int(meta.get("size", 0)))

    stage_root = Path("/dev/shm/_hf_stage_tmp") if Path("/dev/shm").exists() else Path("/tmp/_hf_stage_tmp")
    repo_slug = repo_id.replace("/", "__")
    worker_tmp = stage_root / repo_slug
    worker_tmp.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for attempt in range(1, 5):
        local_file: Path | None = None
        try:
            dl_path = hf_hub_download(
                repo_id=repo_id,
                filename=rfilename,
                repo_type="model",
                local_dir=str(worker_tmp),
                token=hf_token or None,
            )
            local_file = Path(dl_path)
            exact_size = local_file.stat().st_size

            tok = vm_write_token()
            if exact_size <= 8 * 1024 * 1024:
                payload = local_file.read_bytes()
                gcs_upload_bytes(bucket, obj_name, payload, content_type="application/octet-stream")
                written = exact_size
            else:
                init_url = (
                    f"https://storage.googleapis.com/upload/storage/v1/b/{bucket}/o"
                    f"?uploadType=resumable&name={urllib.parse.quote(obj_name, safe='')}"
                )
                init_resp = requests.post(
                    init_url,
                    headers={
                        "Authorization": f"Bearer {tok}",
                        "Content-Type": "application/json; charset=UTF-8",
                        "X-Upload-Content-Type": "application/octet-stream",
                        "X-Upload-Content-Length": str(exact_size),
                    },
                    data=json.dumps({"name": obj_name}),
                    timeout=60,
                )
                init_resp.raise_for_status()
                session_uri = init_resp.headers["Location"]

                with open(local_file, "rb") as fp:
                    put_resp = requests.put(
                        session_uri,
                        data=fp,
                        headers={
                            "Content-Type": "application/octet-stream",
                            "Content-Length": str(exact_size),
                        },
                        timeout=1800,
                    )
                put_resp.raise_for_status()
                meta_out = put_resp.json()
                written = int(meta_out.get("size", exact_size))

            dur = max(time.time() - t0, 1e-3)
            mbps = (written / (1024 * 1024)) / dur
            print(
                f"[stager] UPLOADED {repo_id}/{rfilename} -> gs://{bucket}/{obj_name} "
                f"({written/1e9:.2f} GB in {dur:.1f}s = {mbps:.1f} MiB/s)",
                flush=True,
            )
            return (rfilename, "UPLOADED", written)
        except Exception as exc:
            print(f"[stager] WARN attempt {attempt}/4 failed for {repo_id}/{rfilename}: {exc}", flush=True)
            if attempt == 4:
                raise
            time.sleep(4 * attempt)
        finally:
            if local_file is not None and local_file.exists():
                try:
                    local_file.unlink()
                except OSError:
                    pass
    return (rfilename, "FAILED", 0)


def ensure_model_staged_to_gcs(
    repo_id: str,
    bucket: str,
    dst_prefix: str,
    worker_rank: int = 0,
    num_workers: int = 1,
    threads_per_worker: int = 8,
    wait_all: bool = True,
    timeout_s: int = 3600,
) -> dict[str, Any]:
    """Cooperatively stages `repo_id` into `gs://{bucket}/{dst_prefix}/` across `num_workers` hosts."""
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    marker_obj = f"{dst_prefix.rstrip('/')}/_STAGE_COMPLETE.json"
    existing_marker = gcs_get_metadata(bucket, marker_obj)
    if existing_marker is not None:
        print(f"[stager][w{worker_rank}/{num_workers}] Marker exists: gs://{bucket}/{marker_obj}", flush=True)
        return {"repo_id": repo_id, "gcs_uri": f"gs://{bucket}/{dst_prefix.rstrip('/')}", "cached": True}

    api = HfApi(token=hf_token)
    info = api.repo_info(repo_id=repo_id, repo_type="model", files_metadata=True)
    siblings = sorted(
        [
            s
            for s in info.siblings
            if not s.rfilename.startswith(".git")
            and not s.rfilename.endswith(".mp4")
            and not s.rfilename.startswith("images/")
        ],
        key=lambda x: x.rfilename,
    )
    existing_objs = gcs_list_objects(bucket, dst_prefix.rstrip("/") + "/")
    missing = []
    for s in siblings:
        obj_name = f"{dst_prefix.rstrip('/')}/{s.rfilename}"
        exp_sz = s.size or 0
        cur_sz = existing_objs.get(obj_name, -1)
        if cur_sz < 0 or (exp_sz > 0 and cur_sz != exp_sz):
            missing.append(s)

    my_files = [s for idx, s in enumerate(missing) if idx % max(num_workers, 1) == worker_rank]
    total_gb = sum((s.size or 0) for s in siblings) / 1e9
    my_gb = sum((s.size or 0) for s in my_files) / 1e9
    print(
        f"[stager][w{worker_rank}/{num_workers}] {repo_id} -> gs://{bucket}/{dst_prefix}: "
        f"total={len(siblings)} files ({total_gb:.2f} GB), missing={len(missing)}, assigned={len(my_files)} ({my_gb:.2f} GB)",
        flush=True,
    )

    t0 = time.time()
    if my_files:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(threads_per_worker, 4)) as pool:
            futs = [
                pool.submit(
                    _stream_hf_file_to_gcs,
                    repo_id,
                    s.rfilename,
                    s.size or 0,
                    bucket,
                    dst_prefix,
                    hf_token,
                )
                for s in my_files
            ]
            for fut in concurrent.futures.as_completed(futs):
                fut.result()
        stage_root = Path("/dev/shm/_hf_stage_tmp") if Path("/dev/shm").exists() else Path("/tmp/_hf_stage_tmp")
        shutil.rmtree(stage_root / repo_id.replace("/", "__"), ignore_errors=True)

    if wait_all:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            cur_objs = gcs_list_objects(bucket, dst_prefix.rstrip("/") + "/")
            still_missing = [
                s.rfilename
                for s in siblings
                if cur_objs.get(f"{dst_prefix.rstrip('/')}/{s.rfilename}", -1) != (s.size or cur_objs.get(f"{dst_prefix.rstrip('/')}/{s.rfilename}", 0))
                and (s.size or 0) > 0
            ]
            if not still_missing:
                if worker_rank == 0:
                    gcs_upload_bytes(
                        bucket,
                        marker_obj,
                        json.dumps(
                            {
                                "repo_id": repo_id,
                                "completed_at_utc": utc_iso(),
                                "num_files": len(siblings),
                                "total_bytes": sum((s.size or 0) for s in siblings),
                            },
                            indent=2,
                        ).encode("utf-8"),
                    )
                print(
                    f"[stager][w{worker_rank}/{num_workers}] ALL {len(siblings)} files ready in "
                    f"{time.time() - t0:.1f}s for {repo_id}",
                    flush=True,
                )
                break
            print(
                f"[stager][w{worker_rank}/{num_workers}] Waiting on {len(still_missing)} files from peer workers "
                f"(e.g. {still_missing[:3]})...",
                flush=True,
            )
            # Pitch in if any straggler files remain after 120s!
            if time.time() - t0 > 120 and worker_rank == 0:
                stragglers = [s for s in siblings if s.rfilename in set(still_missing)]
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as spool:
                    list(
                        spool.map(
                            lambda s: _stream_hf_file_to_gcs(
                                repo_id, s.rfilename, s.size or 0, bucket, dst_prefix, hf_token
                            ),
                            stragglers,
                        )
                    )
            time.sleep(8)

    return {
        "repo_id": repo_id,
        "gcs_uri": f"gs://{bucket}/{dst_prefix.rstrip('/')}",
        "stage_seconds": round(time.time() - t0, 2),
        "cached": False,
    }


def pull_gcs_prefix_to_local(
    bucket: str,
    src_prefix: str,
    local_dir: Path,
    threads: int = 16,
) -> dict[str, Any]:
    """Fast 16-thread regional GCS -> /dev/shm downloader (20+ GB/s on TPU hosts)."""
    t0 = time.time()
    objs = gcs_list_objects(bucket, src_prefix.rstrip("/") + "/")
    items = [(name, sz) for name, sz in objs.items() if not name.endswith("/")]
    local_dir.mkdir(parents=True, exist_ok=True)

    def _dl_one(name_sz: tuple[str, int]) -> int:
        obj_name, sz = name_sz
        rel = obj_name[len(src_prefix.rstrip("/")) + 1 :]
        if not rel:
            return 0
        dst = local_dir / rel
        if dst.exists() and (sz == 0 or dst.stat().st_size == sz):
            return 0
        dst.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(4):
            try:
                tok = vm_access_token()
                url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{urllib.parse.quote(obj_name, safe='')}?alt=media"
                req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}"})
                with urllib.request.urlopen(req, timeout=300) as resp, open(dst, "wb") as f:
                    shutil.copyfileobj(resp, f, length=16 * 1024 * 1024)
                return sz
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))
        return 0

    downloaded = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as pool:
        for b in pool.map(_dl_one, items):
            downloaded += b
    dur = max(time.time() - t0, 1e-3)
    print(
        f"[stager] Local sync gs://{bucket}/{src_prefix} -> {local_dir} "
        f"({downloaded/1e9:.2f} GB new in {dur:.1f}s = {(downloaded/1e9)/dur:.2f} GB/s)",
        flush=True,
    )
    return {"local_dir": str(local_dir), "downloaded_bytes": downloaded, "sync_seconds": round(dur, 2)}
