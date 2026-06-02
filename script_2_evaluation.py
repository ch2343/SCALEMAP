import os
import argparse

import scanpy as sc
import pandas as pd

from configs.datasets import list_datasets
from utils.metrics import summarize_all_metrics
from utils.io import load_json_if_exists, extract_runtime_fields, save_dataframe
from configs.datasets import parse_modalities


DEFAULT_METHODS = ["scalemap", "scmodal", "scglue", "maxfuse", "bindsc"]


def build_eval_paths(output_root, method, dataset, modalities, version):
    base_dir = os.path.join(output_root, dataset, method, version)
    return {
        "base_dir": base_dir,
        "integrated_h5ad": os.path.join(
            base_dir,
            f"integrated_{method}_{dataset}_{modalities}_{version}.h5ad"
        ),
        "runtime_json": os.path.join(
            base_dir,
            f"runtime_{method}_{dataset}_{modalities}_{version}.json"
        ),
        "evaluation_csv": os.path.join(
            base_dir,
            f"evaluation_{method}_{dataset}_{modalities}_{version}.csv"
        ),
    }


def evaluate_one_method(
    method,
    dataset,
    modalities,
    version,
    output_root,
    batch_key,
    celltype_key,
    modality_col,
    pair_modalities,
    force_shared_obs_names,
):
    paths = build_eval_paths(output_root, method, dataset, modalities, version)

    if not os.path.exists(paths["integrated_h5ad"]):
        raise FileNotFoundError(f"Integrated h5ad not found: {paths['integrated_h5ad']}")

    adata_integrated = sc.read_h5ad(paths["integrated_h5ad"])
    runtime_dict = load_json_if_exists(paths["runtime_json"])
    peak_memory_bytes, training_time_sec = extract_runtime_fields(runtime_dict)

    summary_df = summarize_all_metrics(
        adata_integrated=adata_integrated,
        method=method,
        dataset=dataset,
        modalities=modalities,
        version=version,
        embed="X_multi",
        batch_key=batch_key,
        celltype_key=celltype_key,
        modality_col=modality_col,
        pair_modalities=pair_modalities,
        peak_memory_bytes=peak_memory_bytes,
        training_time_sec=training_time_sec,
        force_shared_obs_names=force_shared_obs_names,
    )

    save_dataframe(summary_df, paths["evaluation_csv"])
    return summary_df, paths["evaluation_csv"]


def main():
    parser = argparse.ArgumentParser(description="Script II: Evaluation script")
    parser.add_argument("--dataset", type=str, required=True, choices=list_datasets())
    parser.add_argument("--modalities", type=str, default="RNA_ADT")
    parser.add_argument("--method", type=str, required=True, help="Method name or 'All'")
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--output_root", type=str, default="./results")

    parser.add_argument("--batch_key", type=str, default="modality")
    parser.add_argument("--celltype_key", type=str, default="celltype")
    parser.add_argument("--modality_col", type=str, default="modality")
    parser.add_argument("--pair_modalities", nargs=2, default=None)
    parser.add_argument("--force_shared_obs_names", action="store_true")

    args = parser.parse_args()

    if args.pair_modalities is None:
        pair_modalities = tuple(parse_modalities(args.modalities))
    else:
        pair_modalities = tuple(args.pair_modalities)

    if args.method == "All":
        all_results = []
        for method in DEFAULT_METHODS:
            print(f"[INFO] Evaluating method: {method}")
            df, save_path = evaluate_one_method(
                method=method,
                dataset=args.dataset,
                modalities=args.modalities,
                version=args.version,
                output_root=args.output_root,
                batch_key=args.batch_key,
                celltype_key=args.celltype_key,
                modality_col=args.modality_col,
                pair_modalities=pair_modalities,
                force_shared_obs_names=args.force_shared_obs_names,
            )
            print(f"[DONE] Saved: {save_path}")
            all_results.append(df)

        combined = pd.concat(all_results, ignore_index=True)
        combined_path = os.path.join(
            args.output_root,
            args.dataset,
            f"evaluation_All_{args.dataset}_{args.modalities}_{args.version}.csv"
        )
        os.makedirs(os.path.dirname(combined_path), exist_ok=True)
        combined.to_csv(combined_path, index=False)
        print(f"[DONE] Combined evaluation saved to: {combined_path}")

    else:
        method = args.method.lower()
        df, save_path = evaluate_one_method(
            method=method,
            dataset=args.dataset,
            modalities=args.modalities,
            version=args.version,
            output_root=args.output_root,
            batch_key=args.batch_key,
            celltype_key=args.celltype_key,
            modality_col=args.modality_col,
            pair_modalities=pair_modalities,
            force_shared_obs_names=args.force_shared_obs_names,
        )
        print(df)
        print(f"[DONE] Saved: {save_path}")


if __name__ == "__main__":
    main()