#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs/dbpedia"
DATA_DIR="/home/kai3/coco/data/dbpedia_openai1536"
INDEX_DIR="${ROOT_DIR}/build/indexes/dbpedia_ablation"
MAIN_BIN="${ROOT_DIR}/build/main"
LAUNCHER_LOG="${LOG_DIR}/dbpedia_ABCD_rebuild_launcher.log"

mkdir -p "${LOG_DIR}" "${INDEX_DIR}"

if [[ ! -x "${MAIN_BIN}" ]]; then
  echo "Missing executable: ${MAIN_BIN}" >&2
  exit 1
fi

COMMON_ENV=(
  OMP_NUM_THREADS=64
  OMP_PROC_BIND=close
  OMP_PLACES=cores
  RABITQ_FORCE_REBUILD="${RABITQ_FORCE_REBUILD:-1}"
  RABITQ_BUILD_REPORT_EVERY=50000
  RABITQ_DATASET=dbpedia_openai1536
  RABITQ_BASE_PATH="${DATA_DIR}/dbpedia_openai1536_base.fvecs"
  RABITQ_QUERY_PATH="${DATA_DIR}/dbpedia_openai1536_query.fvecs"
  RABITQ_GT_PATH="${DATA_DIR}/dbpedia_openai1536_groundtruth.ivecs"
  RABITQ_INDEX_DIR="${INDEX_DIR}"
  RABITQ_PAPER_PRUNE_COMPARE=active
  RABITQ_PAPER_EPSILON0=1.9
)

run_case() {
  local case_name="$1"
  local build_distance="$2"
  local log_path="${LOG_DIR}/dbpedia_${case_name}_ef_200_M_16.log"

  {
    echo "[$(date --iso-8601=seconds)] start case ${case_name}"
    echo "  log=${log_path}"
    echo "  OMP_NUM_THREADS=64"
    echo "  OMP_PROC_BIND=close"
    echo "  OMP_PLACES=cores"
    echo "  RABITQ_FORCE_REBUILD=${RABITQ_FORCE_REBUILD:-1}"
    echo "  RABITQ_BUILD_REPORT_EVERY=50000"
    echo "  RABITQ_BUILD_DISTANCE=${build_distance}"
  } | tee -a "${LAUNCHER_LOG}"

  env \
    "${COMMON_ENV[@]}" \
    RABITQ_BUILD_DISTANCE="${build_distance}" \
    RABITQ_ABC_ABLATION="${case_name}" \
    "${MAIN_BIN}" > "${log_path}" 2>&1

  echo "[$(date --iso-8601=seconds)] done case ${case_name}" | tee -a "${LAUNCHER_LOG}"
}

if [[ "$#" -eq 0 ]]; then
  : > "${LAUNCHER_LOG}"
  echo "[$(date --iso-8601=seconds)] dbpedia ABCD rebuild started" | tee -a "${LAUNCHER_LOG}"
  set -- A B C D
else
  echo "[$(date --iso-8601=seconds)] dbpedia ABCD rebuild resumed cases=$*" | tee -a "${LAUNCHER_LOG}"
fi

for case_name in "$@"; do
  case "${case_name}" in
    A|B|C)
      run_case "${case_name}" asymmetric4
      ;;
    D)
      run_case D symmetric4
      ;;
    *)
      echo "Unknown case: ${case_name}" >&2
      exit 1
      ;;
  esac
done

echo "[$(date --iso-8601=seconds)] dbpedia ABCD rebuild finished" | tee -a "${LAUNCHER_LOG}"
