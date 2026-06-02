#!/bin/bash
#SBATCH --output logs/evaluation-%A_%a.out
#SBATCH --partition=scavenge
#SBATCH --requeue
#SBATCH --array 0-5
#SBATCH --job-name dsq-evaluation_quickcheck
#SBATCH --error logs/evaluation-%A_%a.err --mem 32g --cpus-per-task 30 --time 02:00:00


# DO NOT EDIT LINE BELOW
/vast/palmer/apps/avx2/software/dSQ/1.05/dSQBatch.py --job-file /gpfs/gibbs/project/guan_leying/ch2343/multi-omics/benchmark_script/jobs/evaluation_quickcheck.txt --status-dir /gpfs/gibbs/project/guan_leying/ch2343/multi-omics/benchmark_script/jobs/dsq_evaluation_status

