#!/usr/bin/env bash
set -euo pipefail

# Run a Hydra sweep as separate Python processes while preserving a Hydra-like
# multirun directory layout and explicitly saving per-job .hydra files.
#
# Usage:
#   bash scripts/launcher.sh RUNNER.py [HYDRA_FLAGS...] -- OVERRIDES...
#
# Example:
#   bash scripts/launcher.sh scripts/experiments/pdx_validation.py \
#     --config-name="pdx_validation.local" \
#     -- \
#     dataset.split.id=1,2,3,4,5,6,7,8,9,10 \
#     model.hyper.noise_stddev=0.3
#
# Notes:
#   - Do NOT pass -m.
#   - Args before "--" are passed directly to Hydra as flags.
#   - Args after "--" are Hydra overrides.
#   - Comma-valued scalar overrides are expanded as sweep dimensions.
#   - Hydra list/dict/tuple literals like [random,uniform] are NOT expanded.
#   - Each job runs in a separate Python process.
#   - Each job chdirs into its own output directory.
#   - Each job gets:
#       .hydra/config.yaml
#       .hydra/hydra.yaml
#       .hydra/overrides.yaml
#       launcher.log
#
# Optional:
#   PYTHON_BIN=/path/to/venv/bin/python bash scripts/launcher.sh ...

RUNNER="${1:-}"
if [[ -z "${RUNNER}" ]]; then
  echo "Usage: $0 RUNNER.py [HYDRA_FLAGS...] -- SWEEP_OVERRIDE [OTHER_OVERRIDES...]"
  exit 1
fi
shift

if [[ ! -f "${RUNNER}" ]]; then
  echo "Runner not found: ${RUNNER}"
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

HYDRA_FLAGS=()
ALL_OVERRIDES=()

saw_sep=0
for arg in "$@"; do
  if [[ "${arg}" == "--" ]]; then
    saw_sep=1
    continue
  fi

  if [[ "${saw_sep}" -eq 0 ]]; then
    HYDRA_FLAGS+=("${arg}")
  else
    ALL_OVERRIDES+=("${arg}")
  fi
done

# Backward-compatible mode: no "--" separator means everything after RUNNER is
# treated as a Hydra override.
if [[ "${saw_sep}" -eq 0 ]]; then
  HYDRA_FLAGS=()
  ALL_OVERRIDES=("$@")
fi

if [[ "${#ALL_OVERRIDES[@]}" -eq 0 ]]; then
  echo "At least one override is required, e.g. dataset.split.id=1,2,3"
  exit 1
fi

is_sweep_value() {
  local val="$1"

  # No comma means not a sweep.
  if [[ "${val}" != *","* ]]; then
    return 1
  fi

  # Hydra/OmegaConf collection literals may contain commas but should be passed
  # as one fixed override.
  case "${val}" in
    \[*\]|\{*\}|\(*\))
      return 1
      ;;
  esac

  return 0
}

FIXED_OVERRIDES=()
SWEEP_KEYS=()
SWEEP_VALS=()

for ov in "${ALL_OVERRIDES[@]}"; do
  if [[ "${ov}" != *=* ]]; then
    echo "Invalid override, expected key=value: ${ov}"
    exit 1
  fi

  key="${ov%%=*}"
  val="${ov#*=}"

  if is_sweep_value "${val}"; then
    SWEEP_KEYS+=("${key}")
    SWEEP_VALS+=("${val}")
  else
    FIXED_OVERRIDES+=("${ov}")
  fi
done

if [[ "${#SWEEP_KEYS[@]}" -eq 0 ]]; then
  echo "No sweep override found. At least one scalar override value must contain commas."
  echo "Example: dataset.split.id=1,2,3,4,5,6,7,8,9,10"
  echo "Hydra list literals like [random,uniform] are treated as fixed overrides."
  exit 1
fi

FIRST_SWEEP_OVERRIDES=()
for i in "${!SWEEP_KEYS[@]}"; do
  key="${SWEEP_KEYS[$i]}"
  vals="${SWEEP_VALS[$i]}"
  first="${vals%%,*}"
  FIRST_SWEEP_OVERRIDES+=("${key}=${first}")
done

# Resolve hydra.sweep.dir using the full composed config.
# hydra.sweep.subdir often uses ${hydra:job.num}, which is None outside
# Hydra native multirun, so override subdir only for config inspection.
CFG_ALL_FILE="$(mktemp)"
trap 'rm -f "${CFG_ALL_FILE}"' EXIT

"${PYTHON_BIN}" "${RUNNER}" \
  "${HYDRA_FLAGS[@]}" \
  --cfg all \
  --resolve \
  "${FIXED_OVERRIDES[@]}" \
  "${FIRST_SWEEP_OVERRIDES[@]}" \
  "hydra.sweep.subdir=0" \
  > "${CFG_ALL_FILE}"

SWEEP_DIR="$(
  "${PYTHON_BIN}" - "${CFG_ALL_FILE}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import yaml

path = Path(sys.argv[1])
cfg = yaml.safe_load(path.read_text())

if cfg is None:
    raise SystemExit("Hydra --cfg all produced empty output.")

try:
    sweep_dir = cfg["hydra"]["sweep"]["dir"]
except Exception as e:
    raise SystemExit(
        "Could not read hydra.sweep.dir from resolved --cfg all output: "
        f"{e}"
    )

if not sweep_dir:
    raise SystemExit("Resolved hydra.sweep.dir is empty.")

print(sweep_dir)
PY
)"

mkdir -p "${SWEEP_DIR}"

JOB_OVERRIDES_FILE="${SWEEP_DIR}/.subprocess_sweep_jobs.tsv"

"${PYTHON_BIN}" - "$JOB_OVERRIDES_FILE" "${SWEEP_KEYS[@]}" -- "${SWEEP_VALS[@]}" <<'PY'
from __future__ import annotations

import itertools
import sys
from pathlib import Path

out_path = Path(sys.argv[1])
args = sys.argv[2:]

sep = args.index("--")
keys = args[:sep]
vals = args[sep + 1 :]

split_vals = [v.split(",") for v in vals]

with out_path.open("w") as f:
    for combo in itertools.product(*split_vals):
        overrides = [f"{k}={v}" for k, v in zip(keys, combo)]
        f.write("\t".join(overrides) + "\n")
PY

N_JOBS="$(wc -l < "${JOB_OVERRIDES_FILE}" | tr -d ' ')"

write_job_hydra_files() {
  local job_dir="$1"
  local job_num="$2"
  shift 2

  local job_overrides=("$@")
  local hydra_dir="${job_dir}/.hydra"
  local cfg_all_file="${job_dir}/.hydra_all_resolved.yaml"

  mkdir -p "${hydra_dir}"

  # Compose full resolved config for this exact job without running the task.
  # We set hydra.run.dir to the job dir to match the actual run.
  # We set hydra.sweep.subdir to job_num to avoid ${hydra:job.num}=None during inspection.
  "${PYTHON_BIN}" "${RUNNER}" \
    "${HYDRA_FLAGS[@]}" \
    --cfg all \
    --resolve \
    "${FIXED_OVERRIDES[@]}" \
    "${job_overrides[@]}" \
    "hydra.run.dir=${job_dir}" \
    "hydra.job.chdir=true" \
    "hydra.output_subdir=.hydra" \
    "hydra.sweep.subdir=${job_num}" \
    > "${cfg_all_file}"

  "${PYTHON_BIN}" - \
    "${cfg_all_file}" \
    "${hydra_dir}/config.yaml" \
    "${hydra_dir}/hydra.yaml" \
    "${hydra_dir}/overrides.yaml" \
    "${FIXED_OVERRIDES[@]}" \
    "${job_overrides[@]}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import yaml

cfg_all_path = Path(sys.argv[1])
config_out = Path(sys.argv[2])
hydra_out = Path(sys.argv[3])
overrides_out = Path(sys.argv[4])
overrides = list(sys.argv[5:])

cfg = yaml.safe_load(cfg_all_path.read_text())
if cfg is None:
    raise SystemExit(f"Empty resolved config: {cfg_all_path}")

hydra_cfg = cfg.pop("hydra", None)
if hydra_cfg is None:
    raise SystemExit("Resolved config did not contain a hydra section.")

config_out.write_text(
    yaml.safe_dump(
        cfg,
        sort_keys=False,
        default_flow_style=False,
    )
)

hydra_out.write_text(
    yaml.safe_dump(
        hydra_cfg,
        sort_keys=False,
        default_flow_style=False,
    )
)

overrides_out.write_text(
    yaml.safe_dump(
        overrides,
        sort_keys=False,
        default_flow_style=False,
    )
)
PY

  rm -f "${cfg_all_file}"
}

{
  echo "# Manual subprocess sweep preserving Hydra-style output dirs."
  echo "runner: ${RUNNER}"
  echo "python_bin: ${PYTHON_BIN}"
  echo "created_at: $(date -Iseconds)"
  echo "sweep_dir: ${SWEEP_DIR}"
  echo "num_jobs: ${N_JOBS}"
  echo "hydra_flags:"
  if [[ "${#HYDRA_FLAGS[@]}" -eq 0 ]]; then
    echo "  []"
  else
    for x in "${HYDRA_FLAGS[@]}"; do
      echo "  - ${x}"
    done
  fi
  echo "fixed_overrides:"
  if [[ "${#FIXED_OVERRIDES[@]}" -eq 0 ]]; then
    echo "  []"
  else
    for x in "${FIXED_OVERRIDES[@]}"; do
      echo "  - ${x}"
    done
  fi
  echo "sweep_overrides:"
  for i in "${!SWEEP_KEYS[@]}"; do
    echo "  ${SWEEP_KEYS[$i]}: ${SWEEP_VALS[$i]}"
  done
} > "${SWEEP_DIR}/multirun.yaml"

echo "Python: ${PYTHON_BIN}"
echo "Sweep dir: ${SWEEP_DIR}"
echo "Jobs: ${N_JOBS}"
echo

job_num=0

while IFS=$'\t' read -r -a JOB_OVERRIDES; do
  JOB_DIR="${SWEEP_DIR}/${job_num}"
  mkdir -p "${JOB_DIR}"

  echo "================================================================"
  echo "Job ${job_num}/${N_JOBS}"
  echo "Output dir: ${JOB_DIR}"
  echo "Hydra flags:"
  if [[ "${#HYDRA_FLAGS[@]}" -eq 0 ]]; then
    echo "  []"
  else
    printf '  %s\n' "${HYDRA_FLAGS[@]}"
  fi
  echo "Fixed overrides:"
  if [[ "${#FIXED_OVERRIDES[@]}" -eq 0 ]]; then
    echo "  []"
  else
    printf '  %s\n' "${FIXED_OVERRIDES[@]}"
  fi
  echo "Sweep overrides:"
  printf '  %s\n' "${JOB_OVERRIDES[@]}"
  echo "================================================================"

  # Explicitly save per-job configs before running. This guarantees configs are
  # available even if the task crashes before Hydra writes them.
  write_job_hydra_files "${JOB_DIR}" "${job_num}" "${JOB_OVERRIDES[@]}"

  (
    set -euo pipefail

    "${PYTHON_BIN}" "${RUNNER}" \
      "${HYDRA_FLAGS[@]}" \
      "${FIXED_OVERRIDES[@]}" \
      "${JOB_OVERRIDES[@]}" \
      "hydra.run.dir=${JOB_DIR}" \
      "hydra.job.chdir=true" \
      "hydra.output_subdir=null"
  ) 2>&1 | tee "${JOB_DIR}/launcher.log"

  status="${PIPESTATUS[0]}"
  if [[ "${status}" -ne 0 ]]; then
    echo "Job ${job_num} failed with status ${status}"
    exit "${status}"
  fi

  job_num=$((job_num + 1))
  echo
done < "${JOB_OVERRIDES_FILE}"

echo "Finished all jobs."
echo "Sweep dir: ${SWEEP_DIR}"