import os
import argparse
import pandas as pd


DATASETS = [
    "bmcite",
    "D22",
    "D23",
    "GSE164378",
    "COMBAT_subset_005",
    "tea_seq",
]

METRIC_ORDER = [
    "ASW_label",
    "ARI",
    "NMI",
    "ASW_batch",
    "kBET Accept Rate",
    "pos rate",
    "true pos rate",
    "LISI_batch",
    "Label Transfer Accuracy (RNA→ADT)",
    "Label Transfer Accuracy (ADT→RNA)",
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
    values = {}
    for metric in METRIC_ORDER:
        values[metric] = row[metric] if metric in row.index else pd.NA

    s = pd.Series(values, name=dataset_name)

    numeric_vals = pd.to_numeric(s, errors="coerce").dropna()

    if len(numeric_vals) == 0:
        print(f"[WARNING] All metrics are NA for {dataset_name}: {eval_path}")
        return empty_metric_series(dataset_name)

    # if you want to ignore all-zero rows too, keep this block
    if (numeric_vals == 0).all():
        print(f"[WARNING] All metrics are zero for {dataset_name}: {eval_path}")
        return empty_metric_series(dataset_name)

    return s


def build_seed_summary(method: str, version: str, output_root: str) -> pd.DataFrame:
    dataset_series = []

    for dataset in DATASETS:
        eval_path = os.path.join(
            output_root,
            dataset,
            method,
            version,
            f"evaluation_{method}_{dataset}_RNA_ADT_{version}.csv",
        )
        s = read_one_eval_file(eval_path, dataset)
        dataset_series.append(s)

    combined = pd.concat(dataset_series, axis=1)
    combined = combined.reindex(METRIC_ORDER)
    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", type=str)
    parser.add_argument("version_base", type=str)
    parser.add_argument("output_root", nargs="?", default="./results")
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    args = parser.parse_args()

    seed_tables = []
    seed_version_names = []

    out_dir = os.path.join(args.output_root, "summary")
    os.makedirs(out_dir, exist_ok=True)

    for seed in args.seeds:
        version = f"{args.version_base}_seed{seed}"
        print(f"[INFO] Reading seed version: {version}")
        df = build_seed_summary(args.method, version, args.output_root)
        seed_tables.append(df)
        seed_version_names.append(version)

        seed_csv = os.path.join(out_dir, f"summary_{args.method}_RNA_ADT_{version}.csv")
        df.to_csv(seed_csv)

    mean_df = seed_tables[0].copy()
    sd_df = seed_tables[0].copy()

    for col in mean_df.columns:
        stacked = pd.concat(
            [pd.to_numeric(df[col], errors="coerce") for df in seed_tables],
            axis=1
        )

        # optional: do not let exact zeros contribute
        stacked = stacked.mask(stacked == 0)

        mean_df[col] = stacked.mean(axis=1, skipna=True)
        sd_df[col] = stacked.std(axis=1, skipna=True, ddof=1)

    mean_csv = os.path.join(
        out_dir,
        f"summary_{args.method}_RNA_ADT_{args.version_base}_avg_over_seeds.csv"
    )
    sd_csv = os.path.join(
        out_dir,
        f"summary_{args.method}_RNA_ADT_{args.version_base}_sd_over_seeds.csv"
    )

    mean_df.to_csv(mean_csv)
    sd_df.to_csv(sd_csv)

    long_rows = []
    for seed, version, df in zip(args.seeds, seed_version_names, seed_tables):
        for metric in df.index:
            for dataset in df.columns:
                value = pd.to_numeric(df.loc[metric, dataset], errors="coerce")
                if pd.isna(value) or value == 0:
                    continue
                long_rows.append({
                    "method": args.method,
                    "version_base": args.version_base,
                    "version": version,
                    "seed": seed,
                    "metric": metric,
                    "dataset": dataset,
                    "value": value,
                })

    long_df = pd.DataFrame(long_rows)
    long_csv = os.path.join(
        out_dir,
        f"summary_{args.method}_RNA_ADT_{args.version_base}_all_seeds_long.csv"
    )
    long_df.to_csv(long_csv, index=False)

    print(f"[DONE] Mean summary written to: {mean_csv}")
    print(f"[DONE] SD summary written to:   {sd_csv}")
    print(f"[DONE] Long-format written to:  {long_csv}")


if __name__ == "__main__":
    main()