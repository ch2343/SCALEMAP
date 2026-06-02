#!/bin/bash
set -euo pipefail

cd /nfs/roberts/project/pi_lg689/ch2343/multi-omics/benchmark_script

OUTPUT_ROOT="./results"
SEEDS=(10 20 42)

# add a suffix for this rerun
RUN_TAG="repeat7"

# ----------------------------------------
# Manually define settings here
# Format:
#   VERSION_NAME|EXTRA_ARGS
# ----------------------------------------
SETTINGS=(

  # -----------------------------
  # 0) Baseline (your current)
  # -----------------------------
  "scalemap_lr_base|--lr1 0.001 --lr2 0.01"

  # =============================
  # 1) Balanced GAN (very important)
  # =============================
  "scalemap_lr_balanced|--lr1 0.001 --lr2 0.001"

  # -----------------------------
  # slightly weaker D
  # -----------------------------
  "scalemap_lr_ratio0.5|--lr1 0.001 --lr2 0.0005"

  # -----------------------------
  # moderate D (sweet spot candidate)
  # -----------------------------
  "scalemap_lr_ratio2|--lr1 0.001 --lr2 0.002"

  "scalemap_lr_ratio5|--lr1 0.001 --lr2 0.005"

  # =============================
  # 2) Stronger G (stability test)
  # =============================
  "scalemap_lr_smallG_balanced|--lr1 0.0005 --lr2 0.0005"

  "scalemap_lr_smallG_ratio2|--lr1 0.0005 --lr2 0.001"

  "scalemap_lr_smallG_ratio5|--lr1 0.0005 --lr2 0.0025"

  # =============================
  # 3) Slightly stronger D (controlled)
  # =============================
  "scalemap_lr_ratio10|--lr1 0.001 --lr2 0.01"   # same as baseline (for comparison)

  "scalemap_lr_ratio15|--lr1 0.001 --lr2 0.015"

)


for SETTING in "${SETTINGS[@]}"; do
  VERSION_STEM="${SETTING%%|*}"
  EXTRA_ARGS="${SETTING#*|}"

  VERSION_BASE="${VERSION_STEM}_${RUN_TAG}"

  echo "Submitting SCALEMAP benchmark for base version=${VERSION_BASE}"
  echo "  extra args: ${EXTRA_ARGS}"

  EVAL_JOBIDS=()

  for SEED in "${SEEDS[@]}"; do
    VERSION="${VERSION_BASE}_seed${SEED}"

    echo "  -> seed=${SEED}, version=${VERSION}"

    INT_JOBID=$(sbatch \
      --parsable \
      --job-name="int_${VERSION}" \
      --export=ALL,VERSION="${VERSION}",OUTPUT_ROOT="${OUTPUT_ROOT}",EXTRA_ARGS="${EXTRA_ARGS}",SEED="${SEED}" \
      jobs/run_integration_scalemap_setting.sh)

    echo "     Integration job: ${INT_JOBID}"

    EVAL_JOBID=$(sbatch \
      --parsable \
      --dependency=afterok:${INT_JOBID} \
      --job-name="eval_${VERSION}" \
      --export=ALL,VERSION="${VERSION}",OUTPUT_ROOT="${OUTPUT_ROOT}" \
      jobs/run_evaluation_scalemap_setting.sh)

    echo "     Evaluation job: ${EVAL_JOBID}"
    EVAL_JOBIDS+=("${EVAL_JOBID}")

    COMBINE_SEED_JOBID=$(sbatch \
      --parsable \
      --dependency=afterok:${EVAL_JOBID} \
      --partition=scavenge \
      --requeue \
      --job-name="combine_${VERSION}" \
      --output="logs/combine_${VERSION}_%j.out" \
      --error="logs/combine_${VERSION}_%j.err" \
      --wrap="cd /nfs/roberts/project/pi_lg689/ch2343/multi-omics/benchmark_script && python -u combine_method_metrics.py scalemap ${VERSION} ${OUTPUT_ROOT}")

    echo "     Seed-level combine job: ${COMBINE_SEED_JOBID}"
  done

  DEP_STR=$(IFS=:; echo "${EVAL_JOBIDS[*]}")

  AVG_COMBINE_JOBID=$(sbatch \
    --parsable \
    --dependency=afterok:${DEP_STR} \
    --partition=scavenge \
    --requeue \
    --job-name="avg_${VERSION_BASE}" \
    --output="logs/avg_${VERSION_BASE}_%j.out" \
    --error="logs/avg_${VERSION_BASE}_%j.err" \
    --wrap="cd /nfs/roberts/project/pi_lg689/ch2343/multi-omics/benchmark_script && python -u combine_method_metrics_across_seeds.py scalemap ${VERSION_BASE} ${OUTPUT_ROOT} --seeds 10 20 42")

  echo "  Averaged combine job: ${AVG_COMBINE_JOBID}"
done

echo "All SCALEMAP multi-seed benchmark pipelines submitted."