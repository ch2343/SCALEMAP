import os
import argparse
from typing import List

import scanpy as sc

from configs.datasets import get_dataset_config, list_datasets, parse_modalities, get_modality_path

from utils.plotting import (
    generate_raw_modality_umaps,
    generate_integrated_umaps,
    extract_celltype_colors,
)


def build_umap_paths(output_root, method, dataset, modalities, version):
    base_dir = os.path.join(output_root, dataset, method, version)
    umap_dir = os.path.join(base_dir, "umap")
    os.makedirs(umap_dir, exist_ok=True)

    return {
        "base_dir": base_dir,
        "umap_dir": umap_dir,
        "integrated_h5ad": os.path.join(
            base_dir,
            f"integrated_{method}_{dataset}_{modalities}_{version}.h5ad"
        ),
    }


def resolve_celltype_key(adata, requested_key: str) -> str:
    """
    Resolve a usable celltype column from raw/integrated AnnData.
    """
    candidates = [
        requested_key,
        "celltype",
        "celltype.l2",
        "celltype_l2",
        "celltype.l1",
        "celltype_l1",
    ]
    for key in candidates:
        if key in adata.obs.columns:
            return key
    raise ValueError(
        f"No celltype column found. Tried {candidates}. "
        f"Available columns: {list(adata.obs.columns)}"
    )


def discover_available_methods(output_root: str, dataset: str, version: str) -> List[str]:
    """
    Discover methods by checking results/<dataset>/<method>/<version>.
    """
    dataset_dir = os.path.join(output_root, dataset)
    if not os.path.isdir(dataset_dir):
        return []

    methods = []
    for method in sorted(os.listdir(dataset_dir)):
        method_dir = os.path.join(dataset_dir, method, version)
        if os.path.isdir(method_dir):
            methods.append(method)
    return methods


def run_umap_for_one_method(
    method,
    dataset,
    modalities,
    version,
    output_root,
    celltype_key,
    modality_col,
    embed_key,
    skip_raw,
    skip_integrated
):
    dataset_cfg = get_dataset_config(dataset)
    modality_a, modality_b = parse_modalities(modalities)

    print(f"[INFO] Loading raw dataset for {dataset}")
    adata_mod1 = sc.read_h5ad(get_modality_path(dataset_cfg, modality_a))
    adata_mod2 = sc.read_h5ad(get_modality_path(dataset_cfg, modality_b))

    # resolve raw celltype key from modality 1 first (usually RNA)
    raw_celltype_key = resolve_celltype_key(adata_mod1, celltype_key)

    paths = build_umap_paths(output_root, method, dataset, modalities, version)

    if not os.path.exists(paths["integrated_h5ad"]):
        raise FileNotFoundError(f"Integrated h5ad not found: {paths['integrated_h5ad']}")

    print(f"[INFO] Loading integrated embedding for method={method}")
    adata_integrated = sc.read_h5ad(paths["integrated_h5ad"])

    integrated_celltype_key = resolve_celltype_key(adata_integrated, celltype_key)

    # raw UMAPs
    if not skip_raw:
        raw_dir = os.path.join(paths["umap_dir"], "raw")
        print("[INFO] Generating raw UMAPs...")
        generate_raw_modality_umaps(
            adata_dict={
                modality_a: adata_mod1,
                modality_b: adata_mod2,
            },
            outdir=raw_dir,
            celltype_key=raw_celltype_key,
        )

    # stable celltype colors from modality 1 raw if available
    try:
        color_mapping = extract_celltype_colors(adata_mod1, celltype_key=raw_celltype_key)
    except Exception as e:
        print(f"[WARNING] Failed to extract raw celltype colors: {e}")
        color_mapping = None

    # integrated UMAPs
    if not skip_integrated:
        integrated_dir = os.path.join(paths["umap_dir"], "integrated")
        print("[INFO] Generating integrated UMAPs...")
        generate_integrated_umaps(
            adata_integrated=adata_integrated,
            outdir=integrated_dir,
            embed_key=embed_key,
            batch_key=modality_col,
            celltype_key=integrated_celltype_key,
            color_mapping=color_mapping,
            prefix=f"{method}_{dataset}_{modalities}_{version}",
        )

    print(f"[DONE] UMAPs saved under: {paths['umap_dir']}")
    

def main():
    parser = argparse.ArgumentParser(description="Script III: UMAP script")
    parser.add_argument("--dataset", type=str, required=True, choices=list_datasets())
    parser.add_argument("--modalities", type=str, default="RNA_ADT")
    parser.add_argument("--method", type=str, required=True, help="Method name or 'All'")
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--output_root", type=str, default="./results")

    parser.add_argument("--celltype_key", type=str, default="celltype")
    parser.add_argument("--modality_col", type=str, default="modality")
    parser.add_argument("--embed_key", type=str, default="X_multi")
    parser.add_argument("--skip_raw", action="store_true")
    parser.add_argument("--skip_integrated", action="store_true")

    args = parser.parse_args()

    if args.method.lower() == "all":
        methods = discover_available_methods(
            output_root=args.output_root,
            dataset=args.dataset,
            version=args.version,
        )
        if len(methods) == 0:
            raise ValueError(
                f"No methods found under {os.path.join(args.output_root, args.dataset)} "
                f"for version={args.version}"
            )

        print(f"[INFO] Discovered methods: {methods}")
        for method in methods:
            print(f"[INFO] Running UMAP for method: {method}")
            run_umap_for_one_method(
                method=method,
                dataset=args.dataset,
                modalities=args.modalities,
                version=args.version,
                output_root=args.output_root,
                celltype_key=args.celltype_key,
                modality_col=args.modality_col,
                embed_key=args.embed_key,
                skip_raw=args.skip_raw,
                skip_integrated=args.skip_integrated
                
            )
    else:
        method = args.method.lower()
        run_umap_for_one_method(
            method=method,
            dataset=args.dataset,
            modalities=args.modalities,
            version=args.version,
            output_root=args.output_root,
            celltype_key=args.celltype_key,
            modality_col=args.modality_col,
            embed_key=args.embed_key,
            skip_raw=args.skip_raw,
            skip_integrated=args.skip_integrated
        )


if __name__ == "__main__":
    main()