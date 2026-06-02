#!/bin/bash
#SBATCH --output logs/drop_integration_%A_%a.out
#SBATCH --error logs/drop_integration_%A_%a.err
#SBATCH --partition=scavenge
#SBATCH --requeue
#SBATCH --mem=64g
#SBATCH --cpus-per-task=30
#SBATCH --time=1:00:00

MANIFEST=${MANIFEST:-jobs/drop_shared_manifest.csv}

ROW=$(python - <<PY
import pandas as pd, os
manifest = pd.read_csv("${MANIFEST}")
idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
r = manifest.iloc[idx]
print("\t".join(str(r[c]) for c in ["dataset","method","version","correspondence_path","protein_subset_to_correspondence"]))
PY
)

IFS=$'\t' read -r DATASET METHOD VERSION CORR_PATH PROT_SUBSET <<< "$ROW"

cd /nfs/roberts/project/pi_lg689/ch2343/multi-omics/benchmark_script

CMD=(
  python -u script_1_integration.py
  --dataset "${DATASET}"
  --modalities RNA_ADT
  --method "${METHOD}"
  --version "${VERSION}"
  --output_root ./results
  --correspondence_path "${CORR_PATH}"
)

if [ "${PROT_SUBSET}" = "1" ]; then
  CMD+=(--protein_subset_to_correspondence)
fi

echo "[INFO] Running: ${CMD[*]}"
"${CMD[@]}"