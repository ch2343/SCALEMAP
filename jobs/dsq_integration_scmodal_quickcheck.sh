#!/bin/bash
#SBATCH --output logs/scmodal-%A_%a.out
#SBATCH --partition=scavenge
#SBATCH --requeue
#SBATCH --array 0-5
#SBATCH --job-name dsq-scmodal_quickcheck
#SBATCH --error logs/scmodal-%A_%a.err
#SBATCH --mem 64g
#SBATCH --cpus-per-task 30
#SBATCH --time 8:00:00

# DO NOT EDIT LINE BELOW
/vast/palmer/apps/avx2/software/dSQ/1.05/dSQBatch.py \
  --job-file /gpfs/gibbs/project/guan_leying/ch2343/multi-omics/benchmark_script/jobs/scmodal_quickcheck.txt \
  --status-dir /gpfs/gibbs/project/guan_leying/ch2343/multi-omics/benchmark_script/jobs/dsq_scmodal_status