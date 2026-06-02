import os
import sys
import pandas as pd


METHOD = sys.argv[1]
VERSION = sys.argv[2]
OUTPUT_ROOT = sys.argv[3] if len(sys.argv) > 3 else "./results"
MODALITIES = "RNA_ATAC"

DATASETS = [
    "D22_ATAC",
    "D23_ATAC",
    "tea_seq_ATAC",
]

METRIC_ORDER = [
    "ASW_label",
    "ARI",
    "NMI",
    "ASW_batch",
    "kBET Accept Rate",
    "pos rate",
    "true pos rate",
    "Label Transfer Accuracy (RNA→ATAC)",
    "Label Transfer Accuracy (ATAC→RNA)",
    "Average Pair Distance",
    "Average FOSCTTM",
    "Peak Memory (GiB)",
    "Training Time (min)",
]


def empty_metric_series(dataset_name: str) -> pd.Series:
    return pd.Series({metric: pd.NA for metric in METRIC_ORDER}, name=dataset_name)


def read_one_eval_file(eval_path: str, dataset_name: str) -> pd.Series:
    if not os.path.exists(eval_path):
        print(f"[WARNING] Missing evaluation file for {dataset_name}: {eval_path}")
        return empty_metric_series(dataset_name)

    if os.path.getsize(eval_path) == 0:
        print(f"[WARNING] Empty evaluation file for {dataset_name}: {eval_path}")
        return empty_metric_series(dataset_name)

    try:
        df = pd.read_csv(eval_path)
    except Exception as e:
        print(f"[WARNING] Failed to read evaluation file for {dataset_name}: {eval_path}")
        print(f"          Reason: {e}")
        return empty_metric_series(dataset_name)

    if df.shape[0] == 0:
        print(f"[WARNING] Evaluation file has no rows for {dataset_name}: {eval_path}")
        return empty_metric_series(dataset_name)

    row = df.iloc[0]
    values = {metric: row[metric] if metric in row.index else pd.NA for metric in METRIC_ORDER}
    return pd.Series(values, name=dataset_name)


def main():
    dataset_series = []

    for dataset in DATASETS:
        eval_path = os.path.join(
            OUTPUT_ROOT,
            dataset,
            METHOD,
            VERSION,
            f"evaluation_{METHOD}_{dataset}_{MODALITIES}_{VERSION}.csv",
        )
        s = read_one_eval_file(eval_path, dataset)
        dataset_series.append(s)

    combined = pd.concat(dataset_series, axis=1)
    combined = combined.reindex(METRIC_ORDER)

    out_dir = os.path.join(OUTPUT_ROOT, "summary")
    os.makedirs(out_dir, exist_ok=True)

    out_csv = os.path.join(out_dir, f"summary_{METHOD}_{MODALITIES}_{VERSION}.csv")
    combined.to_csv(out_csv)

    print(f"[DONE] Wrote summary table: {out_csv}")
    print(combined)


if __name__ == "__main__":
    main()