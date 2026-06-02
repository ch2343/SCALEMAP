#!/bin/bash
#SBATCH --output logs/combat_subset_gen_%j.out
#SBATCH --error logs/combat_subset_gen_%j.err
#SBATCH --partition=scavenge
#SBATCH --requeue
#SBATCH --mem=150g
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00

cd /nfs/roberts/project/pi_lg689/ch2343/multi-omics/benchmark_script

python -u make_combat_full_subsets.py \
  --rna_path /nfs/roberts/project/pi_lg689/ch2343/multi-omics/datasets/GSE35216673/COMBAT_RNA.h5ad \
  --adt_path /nfs/roberts/project/pi_lg689/ch2343/multi-omics/datasets/GSE35216673/COMBAT_ADT.h5ad \
  --outdir /nfs/roberts/project/pi_lg689/ch2343/multi-omics/datasets/COMBAT_full_subsets \
  --seed 42