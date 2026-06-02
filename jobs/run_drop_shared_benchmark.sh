#!/bin/bash
set -euo pipefail

cd /nfs/roberts/project/pi_lg689/ch2343/multi-omics/benchmark_script

# 1) build nested subset tables
python -u make_shared_feature_subset_tables.py \
  --datasets GSE164378 D22 D23 \
  --outdir resources/drop_shared_tables \
  --seed 10

# 2) build manifest
python -u build_drop_benchmark_manifest.py

N=$(python - <<PY
import pandas as pd
df = pd.read_csv("jobs/drop_shared_manifest.csv")
print(len(df) - 1)
PY
)

echo "[INFO] Manifest size = $((N+1)) jobs"

# 3) integration
INT_JOBID=$(sbatch \
  --parsable \
  --array=0-${N} \
  --export=ALL,MANIFEST=jobs/drop_shared_manifest.csv \
  jobs/run_manifest_integration.sh)

echo "[INFO] Integration array job: ${INT_JOBID}"

# 4) evaluation (runs even if some integration tasks failed; missing files are skipped)
EVAL_JOBID=$(sbatch \
  --parsable \
  --dependency=afterany:${INT_JOBID} \
  --array=0-${N} \
  --export=ALL,MANIFEST=jobs/drop_shared_manifest.csv \
  jobs/run_manifest_evaluation.sh)

echo "[INFO] Evaluation array job: ${EVAL_JOBID}"

# 5) integrated UMAPs only
UMAP_JOBID=$(sbatch \
  --parsable \
  --dependency=afterany:${INT_JOBID} \
  --array=0-${N} \
  --export=ALL,MANIFEST=jobs/drop_shared_manifest.csv \
  jobs/run_manifest_umap.sh)

echo "[INFO] UMAP array job: ${UMAP_JOBID}"

# 6) combine summaries
COMBINE_JOBID=$(sbatch \
  --parsable \
  --partition=scavenge \
  --requeue \
  --dependency=afterany:${EVAL_JOBID} \
  --output=logs/drop_combine_%j.out \
  --error=logs/drop_combine_%j.err \
  --wrap="cd /nfs/roberts/project/pi_lg689/ch2343/multi-omics/benchmark_script && python -u combine_drop_benchmark_summary.py ./results")

echo "[INFO] Combine job: ${COMBINE_JOBID}"
echo "[DONE] Submitted full drop-shared-feature benchmark pipeline."