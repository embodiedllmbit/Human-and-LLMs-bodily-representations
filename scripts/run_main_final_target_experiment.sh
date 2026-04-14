#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
OUT_ROOT="${1:-${PROJECT_ROOT}/outputs/final_target_run}"
TIMEOUT="${TIMEOUT:-1800}"
SAVE_EVERY="${SAVE_EVERY:-1}"

TARGET_MODELS=(
  InternVL3.5
  GLM-4.1V-base
  GLM-4.1V-thinking
  Qwen
  Qwen-Thinking
  RynnBrain-8B
  RynnBrain-CoP
  RoboBrain2.5
  MiMo-Embodied
)

export EMBODIED_GLOBAL_THINK_TEMPERATURE="${EMBODIED_GLOBAL_THINK_TEMPERATURE:-0}"
export EMBODIED_INTERNVL_THINK_TEMPERATURE="${EMBODIED_INTERNVL_THINK_TEMPERATURE:-0}"
export EMBODIED_QWEN_THINK_TEMPERATURE="${EMBODIED_QWEN_THINK_TEMPERATURE:-0}"

mkdir -p "${OUT_ROOT}/non_embodied" "${OUT_ROOT}/embodied"

python -u "${PROJECT_ROOT}/run_experiment.py" \
  --questions "${PROJECT_ROOT}/questions.json" \
  --video-descriptions-csv "${PROJECT_ROOT}/detailprompt.csv" \
  --output "${OUT_ROOT}/non_embodied/final_non_embodied_results.json" \
  --artifacts-root "${OUT_ROOT}/non_embodied/final_non_embodied_artifacts" \
  --models "${TARGET_MODELS[@]}" \
  --prompt-types simple detail \
  --save-every "${SAVE_EVERY}" \
  --timeout "${TIMEOUT}"

python -u "${PROJECT_ROOT}/run_experiment.py" \
  --questions "${PROJECT_ROOT}/questions.json" \
  --video-descriptions-csv "${PROJECT_ROOT}/detailprompt.csv" \
  --output "${OUT_ROOT}/embodied/final_embodied_results.json" \
  --artifacts-root "${OUT_ROOT}/embodied/final_embodied_artifacts" \
  --models "${TARGET_MODELS[@]}" \
  --prompt-types embodied_simple embodied_detail \
  --save-every "${SAVE_EVERY}" \
  --timeout "${TIMEOUT}"
