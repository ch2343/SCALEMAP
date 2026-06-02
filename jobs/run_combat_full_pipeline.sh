#!/bin/bash
set -euo pipefail


INT_JOBID=$(sbatch --parsable jobs/run_combat_full_integration.sh)
echo "Integration job: ${INT_JOBID}"

EVAL_JOBID=$(sbatch --parsable --dependency=afterok:${INT_JOBID} jobs/run_combat_full_evaluation.sh)
echo "Evaluation job: ${EVAL_JOBID}"

UMAP_JOBID=$(sbatch --parsable --dependency=afterok:${INT_JOBID} jobs/run_combat_full_umap.sh)
echo "UMAP job: ${UMAP_JOBID}"