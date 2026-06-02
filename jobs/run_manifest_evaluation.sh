#!/bin/bash
#SBATCH --output logs/drop_eval_%A_%a.out
#SBATCH --error logs/drop_eval_%A_%a.err
#SBATCH --partition=scavenge
#SBATCH --requeue
#SBATCH --mem=32g
#SBATCH --cpus-per-task=30
#SBATCH --time=1:00:00

MANIFEST=${MANIFEST:-jobs/drop_shared_manifest.csv}

ROW=$(python - <<PY
import pandas as pd, os
manifest = pd.read_csv("${MANIFEST}")
idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
r = manifest.iloc[idx]
print("\t".join(str(r[c]) for c in ["dataset","method","version"]))
PY
)

IFS=$'\t' read -r DATASET METHOD VERSION <<< "$ROW"

cd /nfs/roberts/project/pi_lg689/ch2343/multi-omics/benchmark_script

INTEGRATED=./results/${DATASET}/${METHOD}/${VERSION}/integrated_${METHOD}_${DATASET}_RNA_ADT_${VERSION}.h5ad
if [ ! -f "${INTEGRATED}" ]; then
  echo "[WARNING] Missing integrated h5ad, skipping evaluation: ${INTEGRATED}"
  exit 0
fi

python -u script_2_evaluation.py \
  --dataset "${DATASET}" \
  --modalities RNA_ADT \
  --method "${METHOD}" \
  --version "${VERSION}" \
  --output_root ./results \
  --batch_key modality \
  --celltype_key celltype \
  --force_shared_obs_names