#!/bin/bash
#SBATCH --job-name=eval_test
#SBATCH --output=logs/eval_test_%x_%j.out
#SBATCH --error=logs/eval_test_%x_%j.err
#SBATCH --partition=scavenge
#SBATCH --requeue
#SBATCH --mem=32g
#SBATCH --cpus-per-task=30
#SBATCH --time=1:00:00

set -euo pipefail

cd /nfs/roberts/project/pi_lg689/ch2343/multi-omics/benchmark_script

METHOD=${METHOD:?METHOD not set}
VERSION=${VERSION:?VERSION not set}
DATASET=${DATASET:?DATASET not set}
OUTPUT_ROOT=${OUTPUT_ROOT:-./results}

python -u script_2_evaluation.py \
  --dataset "${DATASET}" \
  --modalities RNA_ADT \
  --method "${METHOD}" \
  --version "${VERSION}" \
  --output_root "${OUTPUT_ROOT}" \
  --batch_key modality \
  --celltype_key celltype \
  --force_shared_obs_names \
  --pair_modalities RNA ADT