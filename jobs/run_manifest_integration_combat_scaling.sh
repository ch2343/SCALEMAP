#!/bin/bash
#SBATCH --output logs/combat_scaling_integration_%A_%a.out
#SBATCH --error logs/combat_scaling_integration_%A_%a.err
#SBATCH --partition=scavenge
#SBATCH --requeue
#SBATCH --mem=100g
#SBATCH --cpus-per-task=30
#SBATCH --time=14:00:00

MANIFEST=${MANIFEST:-jobs/combat_scaling_manifest.csv}

ROW=$(python - <<PY
import pandas as pd, os
manifest = pd.read_csv("${MANIFEST}")
idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
r = manifest.iloc[idx]
print("\t".join(str(r[c]) for c in ["dataset","method","version","proportion"]))
PY
)

IFS=$'\t' read -r DATASET METHOD VERSION PROP <<< "$ROW"

cd /nfs/roberts/project/pi_lg689/ch2343/multi-omics/benchmark_script

echo "[INFO] dataset=${DATASET} method=${METHOD} version=${VERSION} proportion=${PROP}"

python -u script_1_integration.py \
  --dataset "${DATASET}" \
  --modalities RNA_ADT \
  --method "${METHOD}" \
  --version "${VERSION}" \
  --output_root ./results