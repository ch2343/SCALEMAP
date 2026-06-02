import os
import argparse
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt

RESULTS_ROOT = "/nfs/roberts/project/pi_lg689/ch2343/multi-omics/benchmark_script/results"

DATASET_CONFIGS = {
    "bmcite": {
        "RNA": "/nfs/roberts/project/pi_lg689/ch2343/multi-omics/datasets/bmcite/RNA.h5ad",
        "ADT": "/nfs/roberts/project/pi_lg689/ch2343/multi-omics/datasets/bmcite/Prot.h5ad",
        "modalities": ["RNA", "ADT"],
    },
    "D22": {
        "RNA": "/nfs/roberts/project/pi_lg689/ch2343/multi-omics/datasets/D22/adata_rna.h5ad",
        "ADT": "/nfs/roberts/project/pi_lg689/ch2343/multi-omics/datasets/D22/adata_adt.h5ad",
        "modalities": ["RNA", "ADT"],
    },
    "D23": {
        "RNA": "/nfs/roberts/project/pi_lg689/ch2343/multi-omics/datasets/D23/adata_rna.h5ad",
        "ADT": "/nfs/roberts/project/pi_lg689/ch2343/multi-omics/datasets/D23/adata_adt.h5ad",
        "modalities": ["RNA", "ADT"],
    },
    "GSE164378": {
        "RNA": "/nfs/roberts/project/pi_lg689/ch2343/multi-omics/datasets/GSE164378/RNA.h5ad",
        "ADT": "/nfs/roberts/project/pi_lg689/ch2343/multi-omics/datasets/GSE164378/Prot.h5ad",
        "modalities": ["RNA", "ADT"],
    },
    "COMBAT_subset_005": {
        "RNA": "/nfs/roberts/project/pi_lg689/ch2343/multi-omics/datasets/GSE35216673/COMBAT_subset_005/COMBAT_RNA_005.h5ad",
        "ADT": "/nfs/roberts/project/pi_lg689/ch2343/multi-omics/datasets/GSE35216673/COMBAT_subset_005/COMBAT_ADT_005.h5ad",
        "modalities": ["RNA", "ADT"],
    },
    "tea_seq": {
        "RNA": "/nfs/roberts/project/pi_lg689/ch2343/multi-omics/datasets/tea_seq/RNA.h5ad",
        "ADT": "/nfs/roberts/project/pi_lg689/ch2343/multi-omics/datasets/tea_seq/ADT.h5ad",
        "modalities": ["RNA", "ADT"],
    },
    "COMBAT_full": {
        "RNA": "/nfs/roberts/project/pi_lg689/ch2343/multi-omics/datasets/GSE35216673/COMBAT_RNA.h5ad",
        "ADT": "/nfs/roberts/project/pi_lg689/ch2343/multi-omics/datasets/GSE35216673/COMBAT_ADT.h5ad",
        "modalities": ["RNA", "ADT"],
    },
    "D22_ATAC": {
        "RNA": "/nfs/roberts/project/pi_lg689/ch2343/multi-omics/datasets/D22/adata_rna.h5ad",
        "ATAC": "/nfs/roberts/project/pi_lg689/ch2343/multi-omics/datasets/D22/adata_atac.h5ad",
        "modalities": ["RNA", "ATAC"],
    },
    "D23_ATAC": {
        "RNA": "/nfs/roberts/project/pi_lg689/ch2343/multi-omics/datasets/D23/adata_rna.h5ad",
        "ATAC": "/nfs/roberts/project/pi_lg689/ch2343/multi-omics/datasets/D23/adata_atac.h5ad",
        "modalities": ["RNA", "ATAC"],
    },
    "tea_seq_ATAC": {
        "RNA": "/nfs/roberts/project/pi_lg689/ch2343/multi-omics/datasets/tea_seq/RNA.h5ad",
        "ATAC": "/nfs/roberts/project/pi_lg689/ch2343/multi-omics/datasets/tea_seq/ATAC.h5ad",
        "modalities": ["RNA", "ATAC"],
    },
}


def ensure_counts_layer(adata: ad.AnnData, layer_name: str = "counts") -> ad.AnnData:
    adata = adata.copy()
    if layer_name in adata.layers:
        return adata
    if adata.raw is not None and getattr(adata.raw, "X", None) is not None:
        counts = adata.raw.X.copy()
    else:
        counts = adata.X.copy()
    if not sp.issparse(counts):
        counts = sp.csr_matrix(counts)
    adata.layers[layer_name] = counts
    return adata


def resolve_celltype_key(adata: ad.AnnData) -> str:
    for key in ["celltype", "celltype.l2", "celltype_l2", "celltype.l1", "celltype_l1"]:
        if key in adata.obs.columns:
            return key
    raise ValueError(f"No usable celltype key found in obs columns: {list(adata.obs.columns)}")


def preprocess_rna_or_atac(adata: ad.AnnData, n_top_genes: int = 2000, max_value: float = 10.0) -> ad.AnnData:
    adata = ensure_counts_layer(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top_genes, layer="counts")
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    if "highly_variable" in adata.var.columns:
        adata = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata, zero_center=False, max_value= max_value)
    return adata


def preprocess_adt(adata: ad.AnnData, max_value: float = 10.0) -> ad.AnnData:
    adata = ensure_counts_layer(adata)
    X = adata.layers["counts"]
    if sp.issparse(X):
        counts_per_cell = np.asarray(X.sum(axis=1)).ravel()
    else:
        counts_per_cell = np.asarray(X).sum(axis=1)
    target_sum = float(np.maximum(np.median(counts_per_cell), 20.0))
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value= max_value)
    return adata


def compute_umap(adata: ad.AnnData, n_pcs: int = 50, n_neighbors: int = 30) -> ad.AnnData:
    adata = adata.copy()
    max_pcs = min(n_pcs, max(2, adata.n_vars - 1), max(2, adata.n_obs - 1))
    sc.tl.pca(adata, n_comps=max_pcs, svd_solver="arpack")
    sc.pp.neighbors(adata, use_rep="X_pca", n_neighbors=min(n_neighbors, max(2, adata.n_obs - 1)))
    sc.tl.umap(adata, random_state=1234, min_dist=0.3)
    return adata


def save_umap(adata: ad.AnnData, color_key: str, title: str, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig = sc.pl.umap(adata, color=color_key, title=title, show=False, return_fig=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def process_one_dataset(dataset_name: str) -> None:
    cfg = DATASET_CONFIGS[dataset_name]
    celltype_key = None
    dataset_outdir = os.path.join(RESULTS_ROOT, dataset_name, "raw_umap_corrected")
    os.makedirs(dataset_outdir, exist_ok=True)

    for modality in cfg["modalities"]:
        print(f"[INFO] {dataset_name} | loading {modality}")
        adata = sc.read_h5ad(cfg[modality])
        if celltype_key is None:
            celltype_key = resolve_celltype_key(adata)

        if modality in {"RNA", "ATAC"}:
            adata_p = preprocess_rna_or_atac(adata)
        elif modality == "ADT":
            adata_p = preprocess_adt(adata)
        else:
            raise ValueError(f"Unsupported modality: {modality}")

        adata_u = compute_umap(adata_p)

        # propagate celltype from original obs just in case processing changed dtypes
        adata_u.obs[celltype_key] = adata.obs.loc[adata_u.obs_names, celltype_key].astype(str).values

        png_path = os.path.join(dataset_outdir, f"UMAP_raw_{modality}_celltype.png")
        h5ad_path = os.path.join(dataset_outdir, f"raw_{modality}_umap.h5ad")

        save_umap(
            adata_u,
            color_key=celltype_key,
            title=f"{dataset_name} {modality} raw UMAP",
            save_path=png_path,
        )
        adata_u.write(h5ad_path)
        print(f"[DONE] saved {png_path}")
        print(f"[DONE] saved {h5ad_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate corrected raw-data UMAPs for all listed datasets.")
    parser.add_argument("--dataset", type=str, default="All", choices=["All"] + sorted(DATASET_CONFIGS.keys()))
    args = parser.parse_args()

    datasets: List[str]
    if args.dataset == "All":
        datasets = list(DATASET_CONFIGS.keys())
    else:
        datasets = [args.dataset]

    for ds in datasets:
        process_one_dataset(ds)


if __name__ == "__main__":
    main()
