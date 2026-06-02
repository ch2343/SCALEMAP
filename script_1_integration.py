import os
import json
import time
import argparse
import tracemalloc
import importlib

import scanpy as sc

from configs.datasets import (
    get_dataset_config,
    list_datasets,
    parse_modalities,
    get_modality_path,
    get_correspondence_path,
)
from methods.registry import METHOD_REGISTRY


def build_output_paths(output_root, method, dataset, modalities, version, model_ext="pt"):
    base_dir = os.path.join(output_root, dataset, method, version)
    os.makedirs(base_dir, exist_ok=True)

    return {
        "base_dir": base_dir,
        "model": os.path.join(
            base_dir,
            f"model_{method}_{dataset}_{modalities}_{version}.{model_ext}"
        ),
        "embedding": os.path.join(
            base_dir,
            f"embedding_{method}_{dataset}_{modalities}_{version}.csv"
        ),
        "integrated_h5ad": os.path.join(
            base_dir,
            f"integrated_{method}_{dataset}_{modalities}_{version}.h5ad"
        ),
        "runtime_json": os.path.join(
            base_dir,
            f"runtime_{method}_{dataset}_{modalities}_{version}.json"
        ),
        "preprocess_json": os.path.join(
            base_dir,
            f"preprocess_{method}_{dataset}_{modalities}_{version}.json"
        ),
    }


def build_common_parser():
    parser = argparse.ArgumentParser(description="Script I: Integration script", add_help=False)
    parser.add_argument("--dataset", type=str, required=True, choices=list_datasets())
    parser.add_argument("--modalities", type=str, default="RNA_ADT")
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--output_root", type=str, default="./results")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--correspondence_path", type=str, default=None)
    parser.add_argument("--protein_subset_to_correspondence", action="store_true")
    parser.add_argument("--preprocess_mode", type=str, default="auto",
                    choices=["auto", "rna_adt", "rna_atac"])
    return parser


def main():
    # -----------------------------
    # Stage 1: parse common args only
    # -----------------------------
    common_parser = build_common_parser()
    common_args, remaining_argv = common_parser.parse_known_args()

    method = common_args.method.lower()
    if method not in METHOD_REGISTRY:
        raise ValueError(
            f"Unsupported method '{method}'. "
            f"Available methods: {list(METHOD_REGISTRY.keys())}"
        )

    # -----------------------------
    # Stage 2: import selected method module
    # -----------------------------
    method_module = importlib.import_module(METHOD_REGISTRY[method])

    # -----------------------------
    # Stage 3: full parser = common + method-specific
    # -----------------------------
    parser = argparse.ArgumentParser(
        description=f"Script I: Integration script for method={method}",
        parents=[common_parser]
    )

    method_module.add_method_args(parser)
    args = parser.parse_args()

    # -----------------------------
    # Dataset loading
    # -----------------------------
    dataset_cfg = get_dataset_config(args.dataset)
    modality_a, modality_b = parse_modalities(args.modalities)

    output_paths = build_output_paths(
        output_root=args.output_root,
        method=method,
        dataset=args.dataset,
        modalities=args.modalities,
        version=args.version,
        model_ext=getattr(method_module, "MODEL_FILE_EXT", "pt"),
    )

    print(f"[INFO] Loading dataset: {args.dataset}")
    adata_mod1 = sc.read_h5ad(get_modality_path(dataset_cfg, modality_a))
    adata_mod2 = sc.read_h5ad(get_modality_path(dataset_cfg, modality_b))

    correspondence_path = args.correspondence_path
    if correspondence_path is None:
        correspondence_path = dataset_cfg.get("correspondence", None)
    
    from utils.correspondence import subset_mod2_by_correspondence

    if args.protein_subset_to_correspondence:
        if correspondence_path is None:
            raise ValueError(
                "--protein_subset_to_correspondence was set, but no correspondence table is available "
                f"for dataset={args.dataset}, modalities={args.modalities}"
            )

        from utils.correspondence import subset_mod2_by_correspondence

        print(f"[INFO] Subsetting {modality_b} to features present in current correspondence table")
        adata_mod2 = subset_mod2_by_correspondence(
            adata_rna=adata_mod1,
            adata_mod2=adata_mod2,
            correspondence_path=correspondence_path,
            modality2_name=modality_b,
        )

    # -----------------------------
    # Method-specific input preparation
    # -----------------------------
    print(f"[INFO] Preparing inputs for method={method}")
    prepared_inputs, preprocess_info = method_module.prepare_inputs(
        adata_rna=adata_mod1,
        adata_mod2=adata_mod2,
        correspondence_path=correspondence_path,
        dataset_name=args.dataset,
        modality_a_name=modality_a,
        modality_b_name=modality_b,
        args=args,
    )

    # -----------------------------
    # Run integration
    # -----------------------------
    print(f"[INFO] Running integration for method={method}")
    wall_t0 = time.time()

    model_obj, adata_integrated, embedding_df, run_stats = method_module.run_method(
        prepared_inputs=prepared_inputs,
        adata_rna_raw=adata_mod1,
        adata_mod2_raw=adata_mod2,
        output_paths=output_paths,
        modality_a_name=modality_a,
        modality_b_name=modality_b,
        args=args,
    )
    total_wall_time_sec = time.time() - wall_t0

    # -----------------------------
    # Save outputs
    # -----------------------------
    print("[INFO] Saving outputs...")
    embedding_df.to_csv(output_paths["embedding"])
    adata_integrated.write(output_paths["integrated_h5ad"])

    runtime_info = {
        "method": method,
        "dataset": args.dataset,
        "modalities": args.modalities,
        "version": args.version,
        "wall_time_sec": total_wall_time_sec,
        **run_stats,
        "model_path": output_paths["model"],
        "embedding_path": output_paths["embedding"],
        "integrated_h5ad_path": output_paths["integrated_h5ad"],
    }

    with open(output_paths["runtime_json"], "w") as f:
        json.dump(runtime_info, f, indent=2)

    with open(output_paths["preprocess_json"], "w") as f:
        json.dump(preprocess_info, f, indent=2)

    print("[DONE]")
    print(f"Model saved to: {output_paths['model']}")
    print(f"Embedding saved to: {output_paths['embedding']}")
    print(f"Integrated h5ad saved to: {output_paths['integrated_h5ad']}")
    print(f"Runtime JSON saved to: {output_paths['runtime_json']}")
    print(f"Preprocess JSON saved to: {output_paths['preprocess_json']}")


if __name__ == "__main__":
    main()