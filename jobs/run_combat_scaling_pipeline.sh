#!/bin/bash
set -euo pipefail

cd /nfs/roberts/project/pi_lg689/ch2343/multi-omics/benchmark_script

# 1) submit subset generation first
SUBSET_JOBID=$(sbatch --parsable jobs/run_combat_subset_generation.sh)
echo "[INFO] Subset generation job: ${SUBSET_JOBID}"

# 2) build manifest after subsets are done
MANIFEST_JOBID=$(sbatch \
  --parsable \
  --partition=scavenge \
  --requeue \
  --dependency=afterok:${SUBSET_JOBID} \
  --output=logs/combat_manifest_%j.out \
  --error=logs/combat_manifest_%j.err \
  --wrap="cd /nfs/roberts/project/pi_lg689/ch2343/multi-omics/benchmark_script && python -u build_combat_scaling_manifest.py")
echo "[INFO] Manifest job: ${MANIFEST_JOBID}"

# 3) submit integration array after manifest is built
INT_JOBID=$(sbatch \
  --parsable \
  --dependency=afterok:${MANIFEST_JOBID} \
  --wrap='cd /nfs/roberts/project/pi_lg689/ch2343/multi-omics/benchmark_script && N=$(python - <<PY
import pandas as pd
df = pd.read_csv("jobs/combat_scaling_manifest.csv")
print(len(df) - 1)
PY
) && sbatch --parsable --array=0-${N} --export=ALL,MANIFEST=jobs/combat_scaling_manifest.csv jobs/run_manifest_integration_combat_scaling.sh' \
  --output=logs/combat_submit_array_%j.out \
  --error=logs/combat_submit_array_%j.err \
  --partition=scavenge \
  --requeue)
echo "[INFO] Integration submit wrapper job: ${INT_JOBID}"

# 4) combine after all integration work is done
COMBINE_JOBID=$(sbatch \
  --parsable \
  --dependency=afterany:${INT_JOBID} \
  --partition=scavenge \
  --requeue \
  --output=logs/combat_scaling_combine_%j.out \
  --error=logs/combat_scaling_combine_%j.err \
  --wrap="cd /nfs/roberts/project/pi_lg689/ch2343/multi-omics/benchmark_script && python -u combine_combat_scaling_runtime.py")
echo "[INFO] Combine job: ${COMBINE_JOBID}"

echo "[DONE] Submitted COMBAT scaling pipeline."