#!/bin/bash
#SBATCH --output logs/integration-%A_%a.out
#SBATCH --partition=scavenge
#SBATCH --requeue
#SBATCH --array 0-5
#SBATCH --job-name dsq-integration_quickcheck
#SBATCH --error logs/integration-%A_%a.err --mem 64g --cpus-per-task 30 --time 8:00:00


# DO NOT EDIT LINE BELOW
/vast/palmer/apps/avx2/software/dSQ/1.05/dSQBatch.py --job-file /gpfs/gibbs/project/guan_leying/ch2343/multi-omics/benchmark_script/jobs/integration_quickcheck.txt --status-dir /gpfs/gibbs/project/guan_leying/ch2343/multi-omics/benchmark_script/jobs/dsq_integration_status

