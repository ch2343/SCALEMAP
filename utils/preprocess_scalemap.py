import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

from methods.scalemap_core import multi_resolution_cluster


def load_correspondence_table(dataset_name: str, correspondence_path: str) -> pd.DataFrame:
    """
    Load RNA-protein correspondence table and apply dataset-agnostic cleanup.
    """
    correspondence = pd.read_csv(correspondence_path)

    # common cleanup used in your code
    if "Protein name" not in correspondence.columns or "RNA name" not in correspondence.columns:
        raise ValueError(
            f"Correspondence table must contain columns ['Protein name', 'RNA name'], "
            f"but got {list(correspondence.columns)}"
        )

    # replacements you previously used
    correspondence["Protein name"] = correspondence["Protein name"].replace(
        {
            "TCRab": "TCR-a/b",
            "TCRgd": "TCR-g/d",
            "CD11a-CD18": "CD11a/CD18",
            "CD66a-c-e": "CD66a/c/e",
        }
    )
    return correspondence


def build_rna_protein_correspondence(
    adata_rna: ad.AnnData,
    adata_adt: ad.AnnData,
    correspondence: pd.DataFrame,
) -> np.ndarray:
    """
    Build matched RNA/protein feature pairs from correspondence table.
    Returns array of shape (n_pairs, 2) with columns:
        [RNA name, Protein name]
    """
    pairs = []

    for _, row in correspondence.iterrows():
        curr_protein_name = row["Protein name"]
        curr_rna_names = row["RNA name"]

        if curr_protein_name not in adata_adt.var_names:
            continue
        if "Ignore" in str(curr_rna_names):
            continue

        for rna_name in str(curr_rna_names).split("/"):
            if rna_name in adata_rna.var_names:
                pairs.append([rna_name, curr_protein_name])

    pairs = np.asarray(pairs, dtype=object)

    if len(pairs) == 0:
        raise ValueError("No valid RNA-protein correspondence pairs found.")

    return pairs


def split_shared_unshared(
    adata_rna: ad.AnnData,
    adata_adt: ad.AnnData,
    rna_protein_correspondence: np.ndarray,
):
    """
    Split RNA / ADT into shared and unshared feature blocks.
    """
    rna_shared_names = rna_protein_correspondence[:, 0]
    adt_shared_names = rna_protein_correspondence[:, 1]

    RNA_shared = adata_rna[:, rna_shared_names].copy()
    ADT_shared = adata_adt[:, adt_shared_names].copy()

    RNA_shared.var["feature_name"] = RNA_shared.var_names.values
    ADT_shared.var["feature_name"] = ADT_shared.var_names.values

    RNA_unshared = adata_rna[
        :,
        sorted(set(adata_rna.var_names) - set(rna_shared_names))
    ].copy()

    ADT_unshared = adata_adt[
        :,
        sorted(set(adata_adt.var_names) - set(adt_shared_names))
    ].copy()

    return RNA_shared, ADT_shared, RNA_unshared, ADT_unshared


def preprocess_feature_blocks(
    RNA_shared,
    ADT_shared,
    RNA_unshared,
    ADT_unshared,
    hvg_top_genes: int = 2000,
):
    """
    Normalize / log1p / scale each block.
    """
    # HVG for RNA unshared
    if RNA_unshared.n_vars > 0:
        if RNA_unshared.n_vars > hvg_top_genes:
            sc.pp.highly_variable_genes(
                RNA_unshared,
                flavor="seurat_v3",
                n_top_genes=hvg_top_genes,
            )
            RNA_unshared = RNA_unshared[:, RNA_unshared.var["highly_variable"]].copy()
        RNA_unshared.var["feature_name"] = RNA_unshared.var_names.values

    if ADT_unshared.n_vars > 0:
        ADT_unshared.var["feature_name"] = ADT_unshared.var_names.values

    # compute target_sum from shared counts
    RNA_shared_X = RNA_shared.X.toarray() if hasattr(RNA_shared.X, "toarray") else np.asarray(RNA_shared.X)
    ADT_shared_X = ADT_shared.X.toarray() if hasattr(ADT_shared.X, "toarray") else np.asarray(ADT_shared.X)

    RNA_counts = RNA_shared_X.sum(axis=1)
    target_sum = float(np.maximum(np.median(RNA_counts.copy()), 20.0))

    # shared
    sc.pp.normalize_total(RNA_shared, target_sum=target_sum)
    sc.pp.log1p(RNA_shared)
    sc.pp.scale(RNA_shared)

    sc.pp.normalize_total(ADT_shared, target_sum=target_sum)
    sc.pp.log1p(ADT_shared)
    sc.pp.scale(ADT_shared)

    # RNA unshared
    if RNA_unshared.n_vars > 0:
        sc.pp.normalize_total(RNA_unshared)
        sc.pp.log1p(RNA_unshared)
        sc.pp.scale(RNA_unshared)

    # ADT unshared
    if ADT_unshared.n_vars > 0:
        sc.pp.normalize_total(ADT_unshared, target_sum=target_sum)
        sc.pp.log1p(ADT_unshared)
        sc.pp.scale(ADT_unshared)

    return RNA_shared, ADT_shared, RNA_unshared, ADT_unshared, target_sum

def build_scalemap_inputs(
    adata_rna: ad.AnnData,
    adata_adt: ad.AnnData,
    correspondence_path: str,
    dataset_name: str,
    hvg_top_genes: int = 2000,
    cluster_resolution: float = 0.5,
    cluster_method: str = "leiden",
    final_scale_max_value: float = 10.0,
):
    """
    Full preprocessing wrapper for SCALEMAP.

    Returns
    -------
    adata1
        processed RNA-side AnnData
    adata2
        processed ADT-side AnnData
    preprocess_info
        summary dict with shared feature count and other metadata
    """
    correspondence = load_correspondence_table(dataset_name, correspondence_path)
    rna_protein_correspondence = build_rna_protein_correspondence(
        adata_rna=adata_rna,
        adata_adt=adata_adt,
        correspondence=correspondence,
    )

    RNA_shared, ADT_shared, RNA_unshared, ADT_unshared = split_shared_unshared(
        adata_rna=adata_rna,
        adata_adt=adata_adt,
        rna_protein_correspondence=rna_protein_correspondence,
    )

    RNA_shared, ADT_shared, RNA_unshared, ADT_unshared, target_sum = preprocess_feature_blocks(
        RNA_shared=RNA_shared,
        ADT_shared=ADT_shared,
        RNA_unshared=RNA_unshared,
        ADT_unshared=ADT_unshared,
        hvg_top_genes=hvg_top_genes,
    )

    rna_blocks = [RNA_shared]
    if RNA_unshared.n_vars > 0:
        rna_blocks.append(RNA_unshared)

    adt_blocks = [ADT_shared]
    if ADT_unshared.n_vars > 0:
        adt_blocks.append(ADT_unshared)

    adata1 = ad.concat(rna_blocks, axis=1, merge="same")
    adata2 = ad.concat(adt_blocks, axis=1, merge="same")

    adata1.obs = adata_rna.obs.copy()
    adata2.obs = adata_adt.obs.copy()

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
        "shared_feature_num": int(RNA_shared.shape[1]),
        "rna_shared_features": int(RNA_shared.shape[1]),
        "adt_shared_features": int(ADT_shared.shape[1]),
        "rna_unshared_features": int(RNA_unshared.shape[1]),
        "adt_unshared_features": int(ADT_unshared.shape[1]),
        "adata1_n_cells": int(adata1.shape[0]),
        "adata2_n_cells": int(adata2.shape[0]),
        "adata1_n_features": int(adata1.shape[1]),
        "adata2_n_features": int(adata2.shape[1]),
        "target_sum": float(target_sum),
        "hvg_top_genes": int(hvg_top_genes),
        "cluster_resolution": float(cluster_resolution),
        "cluster_method": cluster_method,
    }

    return adata1, adata2, preprocess_info