#!/usr/bin/env python3
"""Single-Host PJRT Metadata Proxy + LD_PRELOAD Hook for TPU v7x (8 TensorCores) & TPU v5p (4 Chips).

Enables any pod on a multi-host GKE TPU slice (`2x2x2`, `2x2x4`, `2x4x4`) to run independent single-host
`vllm serve` (`TP=2/4/8 x DP`) and single-client multi-device `torch_tpu` (`tpu:0..N-1`) subprocesses
without multi-host PJRT coordination hangs or root port-80 requirements, while transparently forwarding
GCE OAuth token requests to the real `169.254.169.254` metadata server!
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

HOOK_C = r"""
#define _GNU_SOURCE
#include <sys/socket.h>
#include <sys/types.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>

static int (*real_connect)(int, const struct sockaddr*, socklen_t) = NULL;

int connect(int sockfd, const struct sockaddr *addr, socklen_t addrlen) {
    if (!real_connect) {
        real_connect = dlsym(RTLD_NEXT, "connect");
    }
    if (addr && addr->sa_family == AF_INET) {
        const char *port_s = getenv("MOCK_METADATA_PORT");
        if (port_s && port_s[0] != '\0') {
            struct sockaddr_in copy_addr;
            memcpy(&copy_addr, addr, sizeof(struct sockaddr_in));
            char ip[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, &(copy_addr.sin_addr), ip, INET_ADDRSTRLEN);
            if (strcmp(ip, "169.254.169.254") == 0) {
                int port = atoi(port_s);
                copy_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
                copy_addr.sin_port = htons(port);
                return real_connect(sockfd, (const struct sockaddr *)&copy_addr, sizeof(struct sockaddr_in));
            }
        }
    }
    return real_connect(sockfd, addr, addrlen);
}
"""


def _real_meta(rel: str, default: str = "") -> str:
    try:
        req = urllib.request.Request(
            f"http://169.254.169.254/computeMetadata/v1/{rel.lstrip('/')}",
            headers={"Metadata-Flavor": "Google"},
        )
        with urllib.request.urlopen(req, timeout=3.0) as r:
            return r.read().decode("utf-8").strip()
    except Exception:
        return default


def _patch_torchtpu_export_multidevice() -> None:
    """Fixes torch_tpu 0.2.0 export.py bug where current_accelerator().index is None or C++ current_device mismatches on tpu:0..N-1."""
    exp_path = Path("/usr/local/lib/python3.12/site-packages/torch_tpu/_internal/export/export.py")
    if not exp_path.exists():
        return
    try:
        src = exp_path.read_text()
        target_line = "      default_generator = device_module.default_generators[default_device_index]"
        old_bad_order = (
            "      try:\n"
            "        device_module.set_device(default_device_index)\n"
            "        torch.accelerator.set_device_index(default_device_index)\n"
        )
        good_order = (
            "      try:\n"
            "        torch.accelerator.set_device_index(default_device_index)\n"
            "        device_module.set_device(default_device_index)\n"
        )
        if old_bad_order in src:
            exp_path.write_text(src.replace(old_bad_order, good_order))
            return
        sync_snippet = (
            "      if argument_tensors and argument_tensors[0].device.index is not None:\n"
            "        default_device_index = argument_tensors[0].device.index\n"
            "      else:\n"
            "        default_device_index = (torch.accelerator.current_device_index() if device.index is None else device.index) or 0\n"
            + good_order
            + "      except Exception:\n"
            "        pass\n"
            "      default_generator = device_module.default_generators[default_device_index]"
        )
        if "torch.accelerator.set_device_index(default_device_index)" not in src and target_line in src:
            src = src.replace("      default_device_index = device.index or 0\n", "")
            src = src.replace(
                "      default_device_index = (torch.accelerator.current_device_index() if device.index is None else device.index) or 0\n",
                "",
            )
            exp_path.write_text(src.replace(target_line, sync_snippet, 1))
    except Exception:
        pass


def start_single_host_metadata_env(chip_type: str, port: int = 18080) -> dict[str, str]:
    """Compiles /dev/shm/metadata_hook.so, starts local proxy on `port` (in parent without LD_PRELOAD), returns child env."""
    _patch_torchtpu_export_multidevice()
    shm = Path("/dev/shm")
    shm.mkdir(parents=True, exist_ok=True)
    c_path = shm / "metadata_hook.c"
    so_path = shm / "metadata_hook.so"
    if not so_path.exists():
        c_path.write_text(HOOK_C)
        subprocess.run(["gcc", "-shared", "-fPIC", "-O2", "-o", str(so_path), str(c_path), "-ldl"], check=True)

    host_short = _real_meta("instance/hostname", f"tpu-{chip_type}-node").split(".")[0]
    zone = _real_meta("instance/zone", "projects/0/zones/us-central1-a").rsplit("/", 1)[-1]
    accel = "tpu7x-8" if chip_type == "v7x" else "v5p-8"

    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
            already_up = True
    except OSError:
        already_up = False

    if not already_up:
        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args) -> None:
                return

            def do_GET(self) -> None:
                if self.headers.get("Metadata-Flavor") != "Google":
                    self.send_response(400)
                    self.end_headers()
                    return
                path = urllib.parse.urlparse(self.path).path
                tpu_kv = {
                    "TPU_WORKER_ID": "0",
                    "TPU_WORKER_HOSTNAMES": "localhost",
                    "ALT": "false",
                    "CHIPS_PER_HOST_BOUNDS": "2,2,1",
                    "CONSUMER_PROJECT_ID": "tpu-bench-project",
                    "ENABLE_ICI_RESILIENCY": "false",
                    "ENABLE_IMPROVED_REROUTE_ALLREDUCE_STRATEGY": "false",
                    "HOST_BOUNDS": "1,1,1",
                    "INJECT_SLICE_BUILDER_FAULT": "",
                    "NODE_ID": "0",
                    "TOPOLOGY": "2x2x1",
                    "TPU_ACCELERATOR_TYPE": accel,
                    "WRAP": "false,false,false",
                    "ZONE": zone,
                }
                body: bytes | None = None
                attr = path.rsplit("/attributes/", 1)[-1] if "/instance/attributes/" in path else None
                if attr in tpu_kv:
                    body = tpu_kv[attr].encode()
                elif path.endswith("/computeMetadata/v1/instance/attributes/tpu-env"):
                    lines = [f"{k}: '{v}'" for k, v in tpu_kv.items()]
                    lines.append("TPU_RUNTIME_METRICS_PORTS: '8431,8432,8433,8434'")
                    body = ("\n".join(lines) + "\n").encode()
                elif path.endswith("/computeMetadata/v1/instance/attributes/worker-network-endpoints"):
                    body = b"localhost:0:127.0.0.1"
                elif path.endswith("/computeMetadata/v1/instance/attributes/accelerator-type"):
                    body = accel.encode()
                elif path.endswith("/computeMetadata/v1/instance/attributes/agent-worker-number"):
                    body = b"0"

                if body is not None:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain")
                    self.end_headers()
                    self.wfile.write(body)
                    return

                # Forward OAuth token & all non-TPU metadata to real 169.254.169.254 (parent has NO MOCK_METADATA_PORT!)
                real_req = urllib.request.Request(f"http://169.254.169.254{self.path}")
                for k, v in self.headers.items():
                    if k.lower() != "host":
                        real_req.add_header(k, v)
                try:
                    with urllib.request.urlopen(real_req, timeout=5.0) as resp:
                        self.send_response(resp.status)
                        for k, v in resp.headers.items():
                            self.send_header(k, v)
                        self.end_headers()
                        self.wfile.write(resp.read())
                except urllib.error.HTTPError as e:
                    self.send_response(e.code)
                    self.end_headers()
                    self.wfile.write(e.read())
                except Exception as e:
                    self.send_response(500)
                    self.end_headers()
                    self.wfile.write(str(e).encode())

        srv = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        time.sleep(0.2)

    child_env = os.environ.copy()
    for strip_k in (
        "JOB_COMPLETION_INDEX",
        "BATCH_JOB_INDEX",
        "TPU_WORKER_ID",
        "TPU_WORKER_HOSTNAMES",
        "TPU_PROCESS_ADDRESSES",
        "TPU_CHIPS_PER_PROCESS_BOUNDS",
        "TPU_PROCESS_BOUNDS",
        "CLOUD_TPU_TASK_ID",
        "MEGASCALE_NUM_SLICES",
        "MEGASCALE_SLICE_ID",
        "TORCH_TPU_SLICEBUILDER_ADDRESSES",
        "TORCH_TPU_TOPOLOGY",
        "NNODES",
        "NODE_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        child_env.pop(strip_k, None)
    tcmalloc = "/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4"
    preload_libs = [str(so_path)]
    if Path(tcmalloc).exists():
        preload_libs.insert(0, tcmalloc)
    child_env.update(
        {
            "LD_PRELOAD": ":".join(preload_libs),
            "MOCK_METADATA_PORT": str(port),
            "PJRT_DEVICE": "TPU",
            "TPU_VISIBLE_CHIPS": "0,1,2,3",
            "TPU_WORKER_ID": "0",
            "TPU_WORKER_HOSTNAMES": "localhost",
            "TPU_CHIPS_PER_HOST_BOUNDS": "2,2,1",
            "CHIPS_PER_HOST_BOUNDS": "2,2,1",
            "TPU_HOST_BOUNDS": "1,1,1",
            "HOST_BOUNDS": "1,1,1",
            "TPU_TOPOLOGY": "2x2x1",
        }
    )
    return child_env
