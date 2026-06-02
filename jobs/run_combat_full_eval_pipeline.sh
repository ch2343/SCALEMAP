#!/bin/bash
set -euo pipefail

EVAL_JOBID=$(sbatch --parsable jobs/run_combat_full_evaluation.sh)
echo "Evaluation job: ${EVAL_JOBID}"

UMAP_JOBID=$(sbatch --parsable jobs/run_combat_full_umap.sh)
echo "UMAP job: ${UMAP_JOBID}"