#!/bin/bash
#SBATCH --job-name=combine_scalemap_quickcheck
#SBATCH --output=logs/combine-%j.out
#SBATCH --error=logs/combine-%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4g

set -euo pipefail

python combine_quickcheck_metrics.py