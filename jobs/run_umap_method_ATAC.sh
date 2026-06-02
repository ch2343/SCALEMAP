#!/bin/bash
#SBATCH --output logs/atac_umap-%A_%a.out
#SBATCH --error logs/atac_umap-%A_%a.err
#SBATCH --partition=scavenge
#SBATCH --requeue
#SBATCH --array 0-2
#SBATCH --job-name atac_umap
#SBATCH --mem 32g
#SBATCH --cpus-per-task 16
#SBATCH --time 1:00:00

DATASETS=("D22_ATAC" "D23_ATAC" "tea_seq_ATAC")
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

cd /nfs/roberts/project/pi_lg689/ch2343/multi-omics/benchmark_script

METHOD=${METHOD:?METHOD not set}
VERSION=${VERSION:?VERSION not set}
OUTPUT_ROOT=${OUTPUT_ROOT:-./results}

echo "[INFO] METHOD=${METHOD}"
echo "[INFO] DATASET=${DATASET}"
echo "[INFO] VERSION=${VERSION}"

python -u script_3_umap.py \
  --dataset "${DATASET}" \
  --modalities RNA_ATAC \
  --method "${METHOD}" \
  --version "${VERSION}" \
  --output_root "${OUTPUT_ROOT}" \
  --celltype_key celltype \
  --modality_col modality \
  --skip_raw