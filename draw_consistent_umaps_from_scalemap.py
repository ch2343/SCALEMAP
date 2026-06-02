import os
import argparse
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sp
import matplotlib.pyplot as plt

from configs.datasets import get_dataset_config
from utils.plotting import generate_integrated_umaps

warnings.filterwarnings("ignore")


METHODS = ["scalemap", "scmodal", "scglue", "bindsc", "maxfuse"]


# -----------------------------
# Helpers
# -----------------------------
def ensure_counts_layer(adata, layer_name="counts", make_raw=True):
    if layer_name in adata.layers:
        return adata

    if adata.raw is not None and adata.raw.X is not None:
        counts = adata.raw.X.copy()
    else:
        counts = adata.X.copy()

    if not sp.issparse(counts):
        counts = sp.csr_matrix(counts)

    adata.layers[layer_name] = counts

    if make_raw and adata.raw is None:
        adata.raw = adata.copy()

    return adata


def resolve_celltype_key(adata, requested_key="celltype"):
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


def parse_modalities_from_dataset_cfg(dataset_cfg):
    mods = dataset_cfg["modalities"]
    if len(mods) != 2:
        raise ValueError(f"Expected exactly 2 modalities, got: {mods}")
    return mods[0], mods[1]


def set_celltype_palette_from_reference(target_adata, reference_adata, key="celltype"):
    """
    Force target_adata to use the same category order and colors as reference_adata.
    """
    if key not in reference_adata.obs.columns:
        raise ValueError(f"{key} not found in reference adata.obs")
    if key not in target_adata.obs.columns:
        raise ValueError(f"{key} not found in target adata.obs")

    ref = reference_adata.copy()
    tgt = target_adata.copy()

    ref.obs[key] = ref.obs[key].astype("category")
    tgt.obs[key] = tgt.obs[key].astype("category")

    cats = list(ref.obs[key].cat.categories)
    tgt.obs[key] = tgt.obs[key].cat.set_categories(cats, ordered=True)

    color_key = f"{key}_colors"
    if color_key not in ref.uns:
        sc.pl._utils._set_default_colors_for_categorical_obs(ref, key)

    tgt.uns[color_key] = list(ref.uns[color_key])
    return tgt

# -----------------------------
# Raw UMAP preprocessing
# -----------------------------
def preprocess_raw_rna_or_atac(adata, n_top_genes=2000, max_value=10.0):
    adata = adata.copy()
    adata = ensure_counts_layer(adata, "counts", make_raw=True)

    sc.pp.highly_variable_genes(
        adata,
        flavor="seurat_v3",
        n_top_genes=n_top_genes,
        layer="counts",
    )
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    adata = adata[:, adata.var["highly_variable"]].copy()

    # zero_center=False avoids densifying large sparse matrices
    sc.pp.scale(adata, zero_center=False, max_value=max_value)

    sc.tl.pca(adata, svd_solver="arpack")
    sc.pp.neighbors(adata, use_rep="X_pca", n_neighbors=30)
    sc.tl.umap(adata, random_state=1234, min_dist=0.3)
    return adata


def preprocess_raw_adt(adata, max_value=10.0):
    adata = adata.copy()
    adata = ensure_counts_layer(adata, "counts", make_raw=True)

    X_counts = adata.layers["counts"]
    if sp.issparse(X_counts):
        counts_per_cell = np.asarray(X_counts.sum(axis=1)).ravel()
    else:
        counts_per_cell = np.asarray(X_counts).sum(axis=1)

    target_sum = float(max(np.median(counts_per_cell), 20.0))

    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=max_value)

    sc.tl.pca(adata, svd_solver="arpack")
    sc.pp.neighbors(adata, use_rep="X_pca", n_neighbors=30)
    sc.tl.umap(adata, random_state=1234, min_dist=0.3)
    return adata


def save_single_umap(adata, color, title, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig = sc.pl.umap(
        adata,
        color=color,
        title=title,
        show=False,
        return_fig=True,
    )
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def draw_raw_umaps(dataset_name, dataset_cfg, result_root, reference_integrated):
    modality1, modality2 = parse_modalities_from_dataset_cfg(dataset_cfg)

    adata1 = sc.read_h5ad(dataset_cfg[modality1])
    adata2 = sc.read_h5ad(dataset_cfg[modality2])

    celltype_key1 = resolve_celltype_key(adata1, "celltype")
    celltype_key2 = resolve_celltype_key(adata2, "celltype")

    if modality1 in ["RNA", "ATAC"]:
        adata1p = preprocess_raw_rna_or_atac(adata1)
    else:
        adata1p = preprocess_raw_adt(adata1)

    if modality2 in ["RNA", "ATAC"]:
        adata2p = preprocess_raw_rna_or_atac(adata2)
    else:
        adata2p = preprocess_raw_adt(adata2)

    # normalize celltype key name to "celltype" for plotting consistency
    adata1p.obs["celltype"] = adata1.obs[celltype_key1].astype(str).values
    adata2p.obs["celltype"] = adata2.obs[celltype_key2].astype(str).values
    adata1p.obs["celltype"] = adata1p.obs["celltype"].astype("category")
    adata2p.obs["celltype"] = adata2p.obs["celltype"].astype("category")

    adata1p = set_celltype_palette_from_reference(adata1p, reference_integrated, key="celltype")
    adata2p = set_celltype_palette_from_reference(adata2p, reference_integrated, key="celltype")

    outdir = os.path.join(result_root, dataset_name, "raw_umap")

    save_single_umap(
        adata1p,
        color="celltype",
        title=f"{dataset_name} {modality1} raw UMAP",
        save_path=os.path.join(outdir, f"UMAP_raw_{modality1}_celltype.png"),
    )
    save_single_umap(
        adata2p,
        color="celltype",
        title=f"{dataset_name} {modality2} raw UMAP",
        save_path=os.path.join(outdir, f"UMAP_raw_{modality2}_celltype.png"),
    )


# -----------------------------
# Integrated UMAP
# -----------------------------
def draw_integrated_umaps_for_methods(dataset_name, methods, version_map, result_root, reference_integrated):
    for method in methods:
        version = version_map[method]
        integrated_h5ad = os.path.join(
            result_root,
            dataset_name,
            method,
            version,
            f"integrated_{method}_{dataset_name}_RNA_{'ATAC' if dataset_name.endswith('_ATAC') else 'ADT'}_{version}.h5ad"
        )

        if not os.path.exists(integrated_h5ad):
            print(f"[WARNING] Missing integrated file, skip: {integrated_h5ad}")
            continue

        print(f"[INFO] Drawing integrated UMAP for {dataset_name} / {method}")
        adata_int = sc.read_h5ad(integrated_h5ad)

        celltype_key = resolve_celltype_key(adata_int, "celltype")
        if celltype_key != "celltype":
            adata_int.obs["celltype"] = adata_int.obs[celltype_key].astype(str).values
            adata_int.obs["celltype"] = adata_int.obs["celltype"].astype("category")

        adata_int = set_celltype_palette_from_reference(adata_int, reference_integrated, key="celltype")

        outdir = os.path.join(result_root, dataset_name, method, version, "umap")
        generate_integrated_umaps(
            adata_integrated=adata_int,
            outdir=outdir,
            embed_key="X_multi",
            batch_key="modality",
            celltype_key="celltype",
            color_mapping=dict(
                zip(
                    list(reference_integrated.obs["celltype"].cat.categories),
                    list(reference_integrated.uns["celltype_colors"])
                )
            ),
            prefix=f"{method}_{dataset_name}_{'RNA_ATAC' if dataset_name.endswith('_ATAC') else 'RNA_ADT'}_{version}",
        )


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--result_root", type=str, default="/nfs/roberts/project/pi_lg689/ch2343/multi-omics/benchmark_script/results")
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--scalemap_version", type=str, default=None)
    parser.add_argument("--scmodal_version", type=str, default=None)
    parser.add_argument("--scglue_version", type=str, default=None)
    parser.add_argument("--bindsc_version", type=str, default=None)
    parser.add_argument("--maxfuse_version", type=str, default=None)
    args = parser.parse_args()

    dataset_cfg = get_dataset_config(args.dataset)

    version_map = {
        "scalemap": args.scalemap_version or ("quickcheck_scalemap_atac_v1" if args.dataset.endswith("_ATAC") else "benchmark_v1"),
        "scmodal": args.scmodal_version or ("quickcheck_scmodal_atac_v1" if args.dataset.endswith("_ATAC") else "benchmark_v1"),
        "scglue":  args.scglue_version  or ("quickcheck_scglue_atac_v1"  if args.dataset.endswith("_ATAC") else "benchmark_v1"),
        "bindsc":  args.bindsc_version  or ("quickcheck_bindsc_atac_v1"  if args.dataset.endswith("_ATAC") else "benchmark_v1"),
        "maxfuse": args.maxfuse_version or ("quickcheck_maxfuse_atac_v1" if args.dataset.endswith("_ATAC") else "benchmark_v1"),
    }

    scalemap_version = version_map["scalemap"]
    modality_string = "RNA_ATAC" if args.dataset.endswith("_ATAC") else "RNA_ADT"
    reference_h5ad = os.path.join(
        args.result_root,
        args.dataset,
        "scalemap",
        scalemap_version,
        f"integrated_scalemap_{args.dataset}_{modality_string}_{scalemap_version}.h5ad",
    )
    if not os.path.exists(reference_h5ad):
        raise FileNotFoundError(f"Scalemap reference integrated file not found: {reference_h5ad}")

    print(f"[INFO] Using scalemap reference: {reference_h5ad}")
    reference_integrated = sc.read_h5ad(reference_h5ad)
    ref_key = resolve_celltype_key(reference_integrated, "celltype")
    if ref_key != "celltype":
        reference_integrated.obs["celltype"] = reference_integrated.obs[ref_key].astype(str).values
    reference_integrated.obs["celltype"] = reference_integrated.obs["celltype"].astype("category")

    if "celltype_colors" not in reference_integrated.uns:
        sc.pl._utils._set_default_colors_for_categorical_obs(reference_integrated, "celltype")

    draw_raw_umaps(args.dataset, dataset_cfg, args.result_root, reference_integrated)
    draw_integrated_umaps_for_methods(args.dataset, args.methods, version_map, args.result_root, reference_integrated)

    print(f"[DONE] Finished consistent raw + integrated UMAP drawing for {args.dataset}")


if __name__ == "__main__":
    main()