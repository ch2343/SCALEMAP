import os
import time
import tracemalloc
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import scanpy as sc
import networkx as nx
import scglue


def ensure_counts_layer(adata, layer_name: str = "counts", make_raw: bool = True):
    """
    Put raw counts into adata.layers[layer_name].
    Priority: adata.raw.X (if present) -> adata.X (as-is).
    Optionally snapshot current state into .raw for safekeeping.
    """
    if layer_name in adata.layers:
        return adata  # already has counts

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


def run_scglue_rna_adt(
    adata_RNA,
    adata_ADT,
    rna_protein_correspondence,
    *,
    adt_counts_layer: str = "counts",
    celltype_source: str = "celltype",
    celltype_target: str = "celltype",
    embed_key: str = "X_multi",
    n_pca_rna: int = 100,
    n_pca_adt: int = 20,
    rna_prob_model: str = "NB",
    adt_prob_model: str = "Normal",
    fit_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[ad.AnnData, Dict[str, Any]]:
    """
    Run scGLUE for RNA+ADT, write integrated h5ad + metrics txt, and return outputs.

    Parameters
    ----------
    adata_RNA, adata_ADT:
        AnnData objects for RNA and ADT.
    rna_protein_correspondence:
        Either:
          - np.ndarray / list of pairs [[rna_gene, protein_name], ...]
          - pd.DataFrame with 2 columns [rna_gene, protein_name]
        Must be aligned to current adata_RNA.var_names and adata_ADT.var_names.
    output_dir:
        Directory to write:
          - integrated AnnData: {integrated_h5ad_name}
          - metrics txt: {metrics_txt_name}
    fit_kwargs:
        Extra kwargs forwarded to scglue.models.fit_SCGLUE (e.g., max_epochs, batch_size, etc.).
    calculate_metrics_fn / compute_two_modality_metrics_fn:
        If provided, used to compute and write metrics. Otherwise metrics are skipped.

    Returns
    -------
    adata_integrated:
        Integrated AnnData with .obsm[embed_key] and .X = combined embeddings.
    run_info:
        Dict with training_time_sec and peak_memory_bytes (tracemalloc).
    out1_df:
        DataFrame returned by calculate_metrics_fn (or None if not provided).
    out2_dict:
        Dict returned by compute_two_modality_metrics_fn (or {} if not provided).
    """
    fit_kwargs = fit_kwargs or {}

    # ----------------------------
    # 0) Defensive copies
    # ----------------------------
    adata_RNA = adata_RNA.copy()
    adata_ADT = adata_ADT.copy()

    # ----------------------------
    # 1) Ensure counts layer for RNA/ADT (RNA needs counts for NB)
    # ----------------------------
    adata_RNA = ensure_counts_layer(adata_RNA, layer_name=adt_counts_layer, make_raw=True)
    adata_ADT = ensure_counts_layer(adata_ADT, layer_name=adt_counts_layer, make_raw=True)

    # ----------------------------
    # 2) Normalize correspondence into an array of pairs
    # ----------------------------
    if isinstance(rna_protein_correspondence, pd.DataFrame):
        pairs = rna_protein_correspondence.iloc[:, :2].values
    else:
        pairs = np.asarray(rna_protein_correspondence)

    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("rna_protein_correspondence must be a 2-col table/array of [rna_gene, protein_name].")

    # Keep only valid features
    pairs_filtered = []
    rna_vars_set = set(adata_RNA.var_names)
    adt_vars_set = set(adata_ADT.var_names)
    for rna_gene, protein_name in pairs:
        if (rna_gene in rna_vars_set) and (protein_name in adt_vars_set):
            pairs_filtered.append([rna_gene, protein_name])
    pairs_filtered = np.asarray(pairs_filtered, dtype=object)

    if pairs_filtered.shape[0] == 0:
        raise ValueError("No valid RNA-ADT pairs after filtering by adata var_names.")

    # ----------------------------
    # 3) Configure datasets (omics labels)
    # ----------------------------
    scglue.models.configure_dataset(adata_RNA, "RNA", use_highly_variable=False)
    scglue.models.configure_dataset(adata_ADT, "ADT", use_highly_variable=False)

    # ----------------------------
    # 4) PCA reps (for configure_dataset below)
    # ----------------------------
    sc.tl.pca(adata_RNA, n_comps=n_pca_rna, svd_solver="auto")
    sc.tl.pca(adata_ADT, n_comps=n_pca_adt, svd_solver="auto")

    # ----------------------------
    # 5) Rename ADT proteins -> corresponding RNA gene names (only those in mapping)
    #    and then suffix _rna/_prot to avoid collisions
    # ----------------------------
    protein_to_rna = {protein: rna for rna, protein in pairs_filtered}

    new_adt_names = [
        protein_to_rna[name] if name in protein_to_rna else name
        for name in adata_ADT.var_names
    ]
    adata_ADT.var_names = new_adt_names

    # Recompute mask based on (possibly) renamed ADT var_names
    p = np.asarray(adata_ADT.var_names, dtype=object)
    r = np.asarray(adata_RNA.var_names, dtype=object)
    mask = (p.reshape(-1, 1) == r.reshape(1, -1))  # prot x rna

    rna_vars = [v + "_rna" for v in adata_RNA.var_names]
    prot_vars = [v + "_prot" for v in adata_ADT.var_names]
    adata_RNA.var_names = rna_vars
    adata_ADT.var_names = prot_vars

    # ----------------------------
    # 6) Build feature guidance graph
    # ----------------------------
    adj = pd.DataFrame(mask, index=prot_vars, columns=rna_vars)
    diag_edges = adj[adj > 0].stack().index.tolist()
    diag_edges = [(n1, n2, {"weight": 1.0, "sign": 1}) for n1, n2 in diag_edges]

    self_loop_rna = [(g, g, {"weight": 1.0, "sign": 1}) for g in rna_vars]
    self_loop_prot = [(g, g, {"weight": 1.0, "sign": 1}) for g in prot_vars]

    graph = nx.Graph()
    graph.add_nodes_from(rna_vars)
    graph.add_nodes_from(prot_vars)
    graph.add_edges_from(diag_edges)
    graph.add_edges_from(self_loop_rna)
    graph.add_edges_from(self_loop_prot)

    # Optional sanity check
    scglue.graph.check_graph(graph, [adata_RNA, adata_ADT])

    # ----------------------------
    # 7) Configure datasets for training
    # ----------------------------
    scglue.models.configure_dataset(
        adata_RNA,
        rna_prob_model,
        use_highly_variable=False,
        use_layer=adt_counts_layer,
        use_rep="X_pca",
    )
    scglue.models.configure_dataset(
        adata_ADT,
        adt_prob_model,
        use_highly_variable=False,
        use_rep="X_pca",
    )

    # ----------------------------
    # 8) Fit SCGLUE + profile time/memory
    # ----------------------------
    tracemalloc.start()
    t0 = time.time()
    model = scglue.models.fit_SCGLUE(
        {"RNA": adata_RNA, "ADT": adata_ADT},
        graph,
        **fit_kwargs
    )
    t1 = time.time()
    peak_memory = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    run_info = {
        "training_time_sec": float(t1 - t0),
        "peak_memory_bytes": int(peak_memory),
        "n_pairs": int(pairs_filtered.shape[0]),
        "n_rna_cells": int(adata_RNA.n_obs),
        "n_adt_cells": int(adata_ADT.n_obs),
        "n_rna_features": int(adata_RNA.n_vars),
        "n_adt_features": int(adata_ADT.n_vars),
        "n_pca_rna": int(n_pca_rna),
        "n_pca_adt": int(n_pca_adt),
        "rna_prob_model": str(rna_prob_model),
        "adt_prob_model": str(adt_prob_model),
    }

    # ----------------------------
    # 9) Encode embeddings
    # ----------------------------
    adata_RNA.obsm[embed_key] = model.encode_data("RNA", adata_RNA)
    adata_ADT.obsm[embed_key] = model.encode_data("ADT", adata_ADT)

    rna_embeddings = adata_RNA.obsm[embed_key]
    adt_embeddings = adata_ADT.obsm[embed_key]
    combined_embeddings = np.vstack([rna_embeddings, adt_embeddings])

    # ----------------------------
    # 10) Build integrated AnnData
    # ----------------------------
    obs_combined = pd.concat([adata_RNA.obs, adata_ADT.obs], axis=0)
    obs_combined = obs_combined.copy()
    obs_combined["modality"] = (["RNA"] * adata_RNA.n_obs) + (["ADT"] * adata_ADT.n_obs)

    adata_integrated = ad.AnnData(
        X=combined_embeddings,
        obs=obs_combined
    )
    adata_integrated.obsm[embed_key] = combined_embeddings
    adata_integrated.obsm["X_multi"] = adata_integrated.X

    return adata_integrated, run_info

def run_scglue_rna_atac(
    adata_RNA,
    adata_ATAC,
    *,
    counts_layer: str = "counts",
    embed_key: str = "X_multi",
    n_pca_rna: int = 100,
    n_pca_atac: int = 100,
    rna_prob_model: str = "NB",
    atac_prob_model: str = "Normal",
    fit_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[ad.AnnData, Dict[str, Any]]:
    """
    Run scGLUE for RNA+ATAC (gene-aligned / gene-activity-like ATAC matrix),
    using direct shared feature names as guidance edges.
    """
    fit_kwargs = fit_kwargs or {}

    # ----------------------------
    # 0) Defensive copies
    # ----------------------------
    adata_RNA = adata_RNA.copy()
    adata_ATAC = adata_ATAC.copy()

    # ----------------------------
    # 1) Ensure counts layer
    # ----------------------------
    adata_RNA = ensure_counts_layer(adata_RNA, layer_name=counts_layer, make_raw=True)
    adata_ATAC = ensure_counts_layer(adata_ATAC, layer_name=counts_layer, make_raw=True)

    # ----------------------------
    # 2) Find shared features directly by name
    # ----------------------------
    shared = sorted(set(adata_RNA.var_names).intersection(set(adata_ATAC.var_names)))
    if len(shared) == 0:
        raise ValueError("No shared RNA-ATAC features found by direct name alignment.")

    # ----------------------------
    # 3) PCA before renaming
    # ----------------------------
    sc.tl.pca(adata_RNA, n_comps=n_pca_rna, svd_solver="auto")
    sc.tl.pca(adata_ATAC, n_comps=n_pca_atac, svd_solver="auto")

    # ----------------------------
    # 4) Suffix feature names to avoid collisions
    # ----------------------------
    original_rna_vars = list(adata_RNA.var_names)
    original_atac_vars = list(adata_ATAC.var_names)

    adata_RNA.var_names = [f"{g}_rna" for g in original_rna_vars]
    adata_ATAC.var_names = [f"{g}_atac" for g in original_atac_vars]

    shared_rna = [f"{g}_rna" for g in shared]
    shared_atac = [f"{g}_atac" for g in shared]

    # ----------------------------
    # 5) Build feature guidance graph
    # ----------------------------
    graph = nx.Graph()

    graph.add_nodes_from(adata_RNA.var_names, omics="RNA")
    graph.add_nodes_from(adata_ATAC.var_names, omics="ATAC")

    # self loops
    for n in graph.nodes:
        graph.add_edge(n, n, weight=1.0, sign=1)

    # diagonal matched-feature edges
    for r, a in zip(shared_rna, shared_atac):
        graph.add_edge(r, a, weight=1.0, sign=1)

    scglue.graph.check_graph(graph, [adata_RNA, adata_ATAC])

    # ----------------------------
    # 6) Configure datasets
    # ----------------------------
    scglue.models.configure_dataset(
        adata_RNA,
        rna_prob_model,
        use_highly_variable=False,
        use_layer=counts_layer,
        use_rep="X_pca",
    )
    scglue.models.configure_dataset(
        adata_ATAC,
        atac_prob_model,
        use_highly_variable=False,
        use_layer=counts_layer if atac_prob_model in {"NB", "ZINB"} else None,
        use_rep="X_pca",
    )

    # ----------------------------
    # 7) Fit SCGLUE
    # ----------------------------
    tracemalloc.start()
    t0 = time.time()

    model = scglue.models.fit_SCGLUE(
        {"RNA": adata_RNA, "ATAC": adata_ATAC},
        graph,
        **fit_kwargs
    )

    t1 = time.time()
    peak_memory = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    run_info = {
        "training_time_sec": float(t1 - t0),
        "peak_memory_bytes": int(peak_memory),
        "n_shared_features": int(len(shared)),
        "n_rna_cells": int(adata_RNA.n_obs),
        "n_atac_cells": int(adata_ATAC.n_obs),
        "n_rna_features": int(adata_RNA.n_vars),
        "n_atac_features": int(adata_ATAC.n_vars),
        "n_pca_rna": int(n_pca_rna),
        "n_pca_atac": int(n_pca_atac),
        "rna_prob_model": str(rna_prob_model),
        "atac_prob_model": str(atac_prob_model),
    }

    # ----------------------------
    # 8) Encode embeddings
    # ----------------------------
    adata_RNA.obsm[embed_key] = model.encode_data("RNA", adata_RNA)
    adata_ATAC.obsm[embed_key] = model.encode_data("ATAC", adata_ATAC)

    rna_embeddings = adata_RNA.obsm[embed_key]
    atac_embeddings = adata_ATAC.obsm[embed_key]
    combined_embeddings = np.vstack([rna_embeddings, atac_embeddings])

    # ----------------------------
    # 9) Build integrated AnnData
    # ----------------------------
    obs_combined = pd.concat([adata_RNA.obs, adata_ATAC.obs], axis=0)
    obs_combined = obs_combined.copy()
    obs_combined["modality"] = (["RNA"] * adata_RNA.n_obs) + (["ATAC"] * adata_ATAC.n_obs)

    adata_integrated = ad.AnnData(
        X=combined_embeddings,
        obs=obs_combined
    )
    adata_integrated.obsm[embed_key] = combined_embeddings
    adata_integrated.obsm["X_multi"] = combined_embeddings

    return adata_integrated, run_info

MODEL_FILE_EXT = "dill"


def add_method_args(parser):
    parser.add_argument("--n_pca_rna", type=int, default=100)
    parser.add_argument("--n_pca_mod2", type=int, default=20)
    parser.add_argument("--rna_prob_model", type=str, default="NB")
    parser.add_argument("--mod2_prob_model", type=str, default="Normal")


def _load_pairs_from_csv(
    adata_rna: ad.AnnData,
    adata_adt: ad.AnnData,
    correspondence_path: str,
) -> np.ndarray:
    correspondence = pd.read_csv(correspondence_path)
    pairs = []

    for _, row in correspondence.iterrows():
        protein = row.iloc[0]
        genes = row.iloc[1]

        if protein not in adata_adt.var_names:
            continue
        if "Ignore" in str(genes):
            continue

        for g in str(genes).split("/"):
            if g in adata_rna.var_names:
                pairs.append([g, protein])

    return np.asarray(pairs, dtype=object)


def prepare_inputs(
    adata_rna: ad.AnnData,
    adata_mod2: ad.AnnData,
    correspondence_path: str,
    dataset_name: str,
    modality_a_name: str,
    modality_b_name: str,
    args,
):
    if args.preprocess_mode == "auto":
        if modality_b_name == "ADT":
            preprocess_mode = "rna_adt"
        elif modality_b_name == "ATAC":
            preprocess_mode = "rna_atac"
        else:
            raise ValueError(
                f"Unsupported modality pair: {modality_a_name}_{modality_b_name}"
            )
    else:
        preprocess_mode = args.preprocess_mode

    if preprocess_mode == "rna_adt":
        if correspondence_path is None:
            raise ValueError("RNA-ADT preprocessing requires a correspondence table.")

        pairs = _load_pairs_from_csv(adata_rna, adata_mod2, correspondence_path)

        preprocess_info = {
            "dataset": dataset_name,
            "n_pairs": int(pairs.shape[0]),
            "method_preprocessing": "internal_scglue_rna_adt",
            "n_rna_cells": int(adata_rna.n_obs),
            "n_mod2_cells": int(adata_mod2.n_obs),
            "n_rna_features": int(adata_rna.n_vars),
            "n_mod2_features": int(adata_mod2.n_vars),
        }

        prepared_inputs = {
            "mode": "rna_adt",
            "pairs": pairs,
        }

    elif preprocess_mode == "rna_atac":
        shared = sorted(set(adata_rna.var_names).intersection(set(adata_mod2.var_names)))
        if len(shared) == 0:
            raise ValueError("No shared RNA-ATAC features found by direct name alignment.")

        preprocess_info = {
            "dataset": dataset_name,
            "n_shared_features": int(len(shared)),
            "method_preprocessing": "internal_scglue_rna_atac",
            "n_rna_cells": int(adata_rna.n_obs),
            "n_mod2_cells": int(adata_mod2.n_obs),
            "n_rna_features": int(adata_rna.n_vars),
            "n_mod2_features": int(adata_mod2.n_vars),
        }

        prepared_inputs = {
            "mode": "rna_atac",
            "shared_features": shared,
        }

    else:
        raise ValueError(f"Unsupported preprocess_mode: {preprocess_mode}")

    return prepared_inputs, preprocess_info

def run_method(
    prepared_inputs,
    adata_rna_raw: ad.AnnData,
    adata_mod2_raw: ad.AnnData,
    output_paths: Dict[str, str],
    modality_a_name: str,
    modality_b_name: str,
    args,
):
    if prepared_inputs["mode"] == "rna_adt":
        adata_integrated, run_info = run_scglue_rna_adt(
            adata_RNA=adata_rna_raw,
            adata_ADT=adata_mod2_raw,
            rna_protein_correspondence=prepared_inputs["pairs"],
            embed_key="X_multi",
            n_pca_rna=args.n_pca_rna,
            n_pca_adt=args.n_pca_mod2,
            rna_prob_model=args.rna_prob_model,
            adt_prob_model=args.mod2_prob_model,
        )

    elif prepared_inputs["mode"] == "rna_atac":
        adata_integrated, run_info = run_scglue_rna_atac(
            adata_RNA=adata_rna_raw,
            adata_ATAC=adata_mod2_raw,
            embed_key="X_multi",
            n_pca_rna=args.n_pca_rna,
            n_pca_atac=args.n_pca_mod2,
            rna_prob_model=args.rna_prob_model,
            atac_prob_model=args.mod2_prob_model,
        )

    else:
        raise ValueError(f"Unsupported prepared_inputs mode: {prepared_inputs['mode']}")

    embedding_df = pd.DataFrame(
        adata_integrated.obsm["X_multi"],
        index=adata_integrated.obs_names,
        columns=[f"latent_{i+1}" for i in range(adata_integrated.obsm["X_multi"].shape[1])],
    )

    if "modality" in adata_integrated.obs.columns:
        embedding_df.insert(
            0,
            "modality",
            adata_integrated.obs["modality"].astype(str).values,
        )

    run_stats = {
        "total_runtime_min": float(run_info.get("training_time_sec", np.nan)) / 60,
        "peak_memory_use": (
            run_info.get("peak_memory_bytes", None) / 1024**3
            if run_info.get("peak_memory_bytes", None) is not None else None
        ),
        **run_info,
    }

    return None, adata_integrated, embedding_df, run_stats