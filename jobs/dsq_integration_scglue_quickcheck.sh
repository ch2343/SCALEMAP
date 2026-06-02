#!/bin/bash
#SBATCH --output logs/scglue-%A_%a.out
#SBATCH --partition=scavenge
#SBATCH --requeue
#SBATCH --array 0-5
#SBATCH --job-name dsq-scglue_quickcheck
#SBATCH --error logs/scglue-%A_%a.err
#SBATCH --mem 64g
#SBATCH --cpus-per-task 30
#SBATCH --time 8:00:00

# DO NOT EDIT LINE BELOW
/vast/palmer/apps/avx2/software/dSQ/1.05/dSQBatch.py \
  --job-file /gpfs/gibbs/project/guan_leying/ch2343/multi-omics/benchmark_script/jobs/scglue_quickcheck.txt \
  --status-dir /gpfs/gibbs/project/guan_leying/ch2343/multi-omics/benchmark_script/jobs/dsq_scglue_status