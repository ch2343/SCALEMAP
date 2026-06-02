import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import anndata as ad

from methods.scalemap_core import multi_resolution_cluster


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


def build_scalemap_inputs_atac(
    adata_rna: ad.AnnData,
    adata_atac: ad.AnnData,
    correspondence_path: str = None,
    dataset_name: str = None,
    hvg_top_genes: int = 2000,
    cluster_resolution: float = 0.5,
    cluster_method: str = "leiden",
    final_scale_max_value: float = 10.0,
):
    adata_rna = adata_rna.copy()
    adata_atac = adata_atac.copy()

    adata_rna = ensure_counts_layer(adata_rna, layer_name="counts", make_raw=True)
    adata_atac = ensure_counts_layer(adata_atac, layer_name="counts", make_raw=True)

    sc.pp.highly_variable_genes(
        adata_rna,
        flavor="seurat_v3",
        n_top_genes=hvg_top_genes,
        layer="counts",
    )
    sc.pp.normalize_total(adata_rna)
    sc.pp.log1p(adata_rna)

    sc.pp.highly_variable_genes(
        adata_atac,
        flavor="seurat_v3",
        n_top_genes=hvg_top_genes,
        layer="counts",
    )
    sc.pp.normalize_total(adata_atac)
    sc.pp.log1p(adata_atac)

    adata_rna = adata_rna[:, adata_rna.var["highly_variable"]].copy()
    adata_atac = adata_atac[:, adata_atac.var["highly_variable"]].copy()

    shared_features = sorted(set(adata_rna.var_names).intersection(set(adata_atac.var_names)))
    if len(shared_features) == 0:
        raise ValueError("No shared RNA-ATAC features found after HVG filtering.")

    rna_unshared = sorted(set(adata_rna.var_names) - set(shared_features))
    atac_unshared = sorted(set(adata_atac.var_names) - set(shared_features))

    adata1 = adata_rna[:, shared_features + rna_unshared].copy()
    adata2 = adata_atac[:, shared_features + atac_unshared].copy()

    adata1.var["feature_name"] = adata1.var_names.values
    adata2.var["feature_name"] = adata2.var_names.values

    sc.pp.scale(adata1, max_value=final_scale_max_value)
    sc.pp.scale(adata2, max_value=final_scale_max_value)

    adata1 = multi_resolution_cluster(
        adata1,
        resolution1=cluster_resolution,
        method=cluster_method,
    )
    adata2 = multi_resolution_cluster(
        adata2,
        resolution1=cluster_resolution,
        method=cluster_method,
    )

    preprocess_info = {
        "dataset": dataset_name,
        "preprocess_mode": "rna_atac",
        "shared_feature_num": int(len(shared_features)),
        "rna_shared_features": int(len(shared_features)),
        "atac_shared_features": int(len(shared_features)),
        "rna_unshared_features": int(len(rna_unshared)),
        "atac_unshared_features": int(len(atac_unshared)),
        "adata1_n_cells": int(adata1.shape[0]),
        "adata2_n_cells": int(adata2.shape[0]),
        "adata1_n_features": int(adata1.shape[1]),
        "adata2_n_features": int(adata2.shape[1]),
        "hvg_top_genes": int(hvg_top_genes),
        "cluster_resolution": float(cluster_resolution),
        "cluster_method": cluster_method,
    }

    return adata1, adata2, preprocess_info