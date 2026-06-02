#!/bin/bash
#SBATCH --output logs/combat_full_umap_%A_%a.out
#SBATCH --error logs/combat_full_umap_%A_%a.err
#SBATCH --partition=scavenge
#SBATCH --requeue
#SBATCH --array=0
#SBATCH --mem=256g
#SBATCH --cpus-per-task=30
#SBATCH --time=3:00:00

#METHODS=("scalemap" "scmodal" "scmodal")
#VERSIONS=("combat_full_scalemap_noise01_ae10_v1" "combat_full_scmodal_steps10000_v1" "combat_full_scmodal_steps2000_v1")

METHODS=("scalemap")
VERSIONS=("combat_full_scalemap_v3")

METHOD=${METHODS[$SLURM_ARRAY_TASK_ID]}
VERSION=${VERSIONS[$SLURM_ARRAY_TASK_ID]}

cd /nfs/roberts/project/pi_lg689/ch2343/multi-omics/benchmark_script

python -u script_3_umap.py \
  --dataset COMBAT_full \
  --modalities RNA_ADT \
  --method "${METHOD}" \
  --version "${VERSION}" \
  --output_root ./results \
  --skip_raw