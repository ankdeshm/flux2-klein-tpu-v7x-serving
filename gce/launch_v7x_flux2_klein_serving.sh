#!/usr/bin/env bash
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
#
# Standalone GCE TPU VM launcher for FLUX.2 Klein (4B & 9B) Serving Benchmark on Cloud TPU v7x.
set -euo pipefail

export GCP_PROJECT_ID="${GCP_PROJECT_ID:-YOUR_GCP_PROJECT_ID}"
export TPU_ZONE="${TPU_ZONE:-us-central1-c}"
export TPU_NAME="${TPU_NAME:-your-tpu-v7x-slice}"
export MODELS_ROOT="${MODELS_ROOT:-/models/models}"
export SHM_ROOT="${SHM_ROOT:-/dev/shm/flux2_models}"
export OUTPUT_DIR="${OUTPUT_DIR:-/tmp/flux2_klein_results}"

echo "[1/3] Staging FLUX.2-klein-4B and assembling Flux-2-Klein-9B Diffusers layout in ${SHM_ROOT}..."
python3 scripts/prepare_flux2_klein_9b_diffusers.py \
  --models-root="${MODELS_ROOT}" \
  --shm-root="${SHM_ROOT}"

echo "[2/3] Running vLLM-Omni OpenAI /v1/images/generations multi-replica topology sweep..."
python3 scripts/run_flux2_klein_vllm_omni_serving.py \
  --suite=host0-main \
  --models-root="${MODELS_ROOT}" \
  --shm-models-root="${SHM_ROOT}" \
  --output-dir="${OUTPUT_DIR}"

echo "[3/3] Running Direct TorchTPU BF16 vs Calibrated FP8 (4B) vs Hybrid KV-FP8 (9B) sweep..."
python3 scripts/run_flux2_klein_torchtpu_fp8_kv_bench.py \
  --models-root="${MODELS_ROOT}" \
  --shm-models-root="${SHM_ROOT}" \
  --models=klein_4b,klein_9b \
  --tag=host0 \
  --output-dir="${OUTPUT_DIR}"

echo "Benchmark complete! Artifacts written to ${OUTPUT_DIR}"
