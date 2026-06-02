#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import scanpy as sc

from configs.datasets import get_dataset_config
from utils.correspondence import load_valid_pairs_for_dataset


DATASETS = [
    "bmcite",
    "D22",
    "D23",
    "GSE164378",
    "COMBAT_subset_005",
    "tea_seq",
    "D22_ATAC",
    "D23_ATAC",
    "tea_seq_ATAC",
]


def summarize_one_dataset(dataset_name: str) -> dict:
    cfg = get_dataset_config(dataset_name)
    modalities = cfg["modalities"]
    if len(modalities) != 2:
        raise ValueError(f"{dataset_name} has unexpected modalities: {modalities}")

    mod1, mod2 = modalities
    if mod1 != "RNA":
        raise ValueError(f"{dataset_name}: expected first modality to be RNA, got {mod1}")

    rna_path = cfg["RNA"]
    mod2_path = cfg[mod2]

    print(f"[INFO] Loading {dataset_name}")
    adata_rna = sc.read_h5ad(rna_path)
    adata_mod2 = sc.read_h5ad(mod2_path)

    n_cells_rna = int(adata_rna.n_obs)
    n_cells_mod2 = int(adata_mod2.n_obs)
    n_features_rna = int(adata_rna.n_vars)
    n_features_mod2 = int(adata_mod2.n_vars)

    if mod2 == "ADT":
        corr_path = cfg["correspondence"]
        valid_pairs = load_valid_pairs_for_dataset(
            adata_rna=adata_rna,
            adata_mod2=adata_mod2,
            correspondence_path=corr_path,
        )
        shared_feature_num = int(valid_pairs.shape[0])
        n_unique_rna_shared = int(valid_pairs["RNA name"].nunique()) if len(valid_pairs) > 0 else 0
        n_unique_mod2_shared = int(valid_pairs["Protein name"].nunique()) if len(valid_pairs) > 0 else 0
        shared_definition = "valid RNA-ADT correspondence pairs"
    elif mod2 == "ATAC":
        shared = sorted(set(adata_rna.var_names).intersection(set(adata_mod2.var_names)))
        shared_feature_num = int(len(shared))
        n_unique_rna_shared = shared_feature_num
        n_unique_mod2_shared = shared_feature_num
        shared_definition = "RNA-ATAC shared feature-name intersection"
    else:
        raise ValueError(f"{dataset_name}: unsupported second modality {mod2}")

    row = {
        "dataset": dataset_name,
        "modality_1": mod1,
        "modality_2": mod2,
        "rna_path": rna_path,
        "mod2_path": mod2_path,
        "n_cells_rna": n_cells_rna,
        "n_cells_mod2": n_cells_mod2,
        "n_features_rna_before_filtering": n_features_rna,
        "n_features_mod2_before_filtering": n_features_mod2,
        "shared_feature_num": shared_feature_num,
        "n_unique_rna_shared_features": n_unique_rna_shared,
        "n_unique_mod2_shared_features": n_unique_mod2_shared,
        "shared_definition": shared_definition,
    }
    return row


def main():
    parser = argparse.ArgumentParser(description="Summarize raw dataset dimensions and shared features.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DATASETS,
        help="Dataset names to summarize",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="./results/dataset_dimension_summary.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    rows = []
    for dataset in args.datasets:
        rows.append(summarize_one_dataset(dataset))

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    print("[DONE] Wrote summary to:")
    print(args.output_csv)
    print(df)


if __name__ == "__main__":
    main()