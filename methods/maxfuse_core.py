# ============================================================
# MaxFuse (RNA + ADT) — MOST memory-efficient practical version
# - NO metrics, NO FOSCTTM (avoids NxN distance explosion)
# - Memory-efficient preprocessing: normalize/log1p + scale(zero_center=False)
# - Uses PCA embeddings as "active arrays" to drastically reduce dense size
# - Writes integrated AnnData to disk and prints the saved path ("link")
# - Reports wall-clock time + peak RSS (true memory) if psutil is available
# ============================================================

import os
import time
from typing import Optional, Dict, Any, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import scanpy as sc
import maxfuse as mf

try:
    import psutil
except ImportError:
    psutil = None


# ---------------------------
# Memory helpers
# ---------------------------

def _rss_mib() -> float:
    """Resident Set Size (true process memory) in MiB."""
    if psutil is None:
        return float("nan")
    return psutil.Process(os.getpid()).memory_info().rss / (1024**2)


def _ensure_float32(X):
    if sp.issparse(X):
        return X.astype(np.float32) if X.dtype != np.float32 else X
    X = np.asarray(X)
    return X.astype(np.float32, copy=False) if X.dtype != np.float32 else X


def _to_dense_float32(X) -> np.ndarray:
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X)
    return X.astype(np.float32, copy=False) if X.dtype != np.float32 else X


def _col_std_sparse_safe(X, eps=1e-12) -> np.ndarray:
    """
    Column-wise std for sparse or dense without densifying the full matrix.
    Uses sqrt(E[x^2] - (E[x])^2).
    """
    if sp.issparse(X):
        mean = np.asarray(X.mean(axis=0)).ravel()
        sq = X.copy()
        sq.data **= 2
        mean2 = np.asarray(sq.mean(axis=0)).ravel()
        var = np.maximum(mean2 - mean**2, 0.0)
        return np.sqrt(var + eps)
    else:
        return np.asarray(X).std(axis=0)


def _row_sums(X) -> np.ndarray:
    if sp.issparse(X):
        return np.asarray(X.sum(axis=1)).ravel()
    return np.asarray(X).sum(axis=1).ravel()


def _preprocess_for_pca(
    adata: ad.AnnData,
    *,
    target_sum: Optional[float] = None,
    do_hvg: bool = False,
    n_top_genes: int = 2000,
    scale_max_value: Optional[float] = 10.0,
    verbose: bool = True,
) -> ad.AnnData:
    """
    Memory-friendly preprocessing:
      normalize_total -> log1p -> (optional HVG) -> scale(zero_center=False)
    Keeping zero_center=False avoids densifying sparse matrices internally.
    """
    if verbose:
        print(f"[pre] start shape={adata.shape}, RSS={_rss_mib():.2f} MiB")

    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)

    if do_hvg:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
        adata = adata[:, adata.var.highly_variable].copy()
        if verbose:
            print(f"[pre] after HVG shape={adata.shape}, RSS={_rss_mib():.2f} MiB")

    sc.pp.scale(adata, zero_center=False, max_value=scale_max_value)
    adata.X = _ensure_float32(adata.X)

    if verbose:
        print(f"[pre] done  shape={adata.shape}, X={type(adata.X)}, RSS={_rss_mib():.2f} MiB")

    return adata


def _compute_pca_dense(
    adata: ad.AnnData,
    *,
    n_pcs: int = 20,
    use_highly_variable: bool = False,  # already HVG-filtered if desired
    verbose: bool = True,
) -> np.ndarray:
    """
    Compute PCA and return dense float32 (n_cells x n_pcs).
    scanpy PCA output is already dense and relatively small.
    """
    if verbose:
        print(f"[pca] computing PCA n_pcs={n_pcs} on shape={adata.shape}, RSS={_rss_mib():.2f} MiB")

    sc.tl.pca(adata, n_comps=n_pcs, use_highly_variable=use_highly_variable, svd_solver="randomized")

    Xpca = adata.obsm["X_pca"]
    Xpca = np.asarray(Xpca, dtype=np.float32)

    if verbose:
        print(f"[pca] done PCA -> {Xpca.shape}, RSS={_rss_mib():.2f} MiB")

    return Xpca


# ---------------------------
# Main function
# ---------------------------

def run_maxfuse_rna_adt_most_memory_efficient(
    adata_RNA: ad.AnnData,
    adata_ADT: ad.AnnData,
    rna_protein_correspondence: Union[np.ndarray, pd.DataFrame, list],
    output_dir: str,
    integrated_h5ad_name: str = "maxfuse_integrated.h5ad",
    *,
    celltype_key_source: str = "celltype",
    celltype_key_target: str = "celltype",
    modality_key: str = "modality",
    embed_key: str = "X_multi",
    modality_a_name: str = "RNA",
    modality_b_name: str = "ADT",
    std_eps: float = 1e-5,
    dim_use: int = 20,
    hvg_n_top: int = 2000,
    scale_max_value: Optional[float] = 10.0,
    active_use_pca: bool = True,
    active_pcs_rna: int = 30,
    active_pcs_adt: int = 20,
    split_params: Optional[Dict[str, Any]] = None,
    graph_params: Optional[Dict[str, Any]] = None,
    init_pivot_params: Optional[Dict[str, Any]] = None,
    refine_pivot_params: Optional[Dict[str, Any]] = None,
    filter_bad_match_params: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Tuple[ad.AnnData, Dict[str, Any], str]:
    """
    Returns:
      - cca_adata: integrated embedding AnnData (cells stacked RNA then ADT)
      - run_info: timing + peak RSS
      - integrated_path: file path (printable "link") where .h5ad is stored
    """

    # Light defaults to reduce memory
    split_params = split_params or dict(
        max_outward_size=3000,
        matching_ratio=2,
        metacell_size=3,
        verbose=verbose,
    )
    graph_params = graph_params or dict(
        n_neighbors1=10,
        n_neighbors2=10,
        svd_components1=15,
        svd_components2=15,
        resolution1=2,
        resolution2=2,
        resolution_tol=0.1,
        verbose=verbose,
    )
    init_pivot_params = init_pivot_params or dict(
        wt1=0.7, wt2=0.7,
        svd_components1=15, svd_components2=15,
    )
    refine_pivot_params = refine_pivot_params or dict(
        wt1=0.7, wt2=0.7,
        svd_components1=15, svd_components2=15,
        cca_components=10,
        n_iters=2,
        randomized_svd=True,
        svd_runs=1,
        verbose=verbose,
    )
    filter_bad_match_params = filter_bad_match_params or dict(target="pivot", filter_prop=0.3)


    # Defensive copies
    adata_RNA = adata_RNA.copy()
    adata_ADT = adata_ADT.copy()

    if verbose:
        print(f"[start] RNA={adata_RNA.shape}, ADT={adata_ADT.shape}, RSS={_rss_mib():.2f} MiB")

    # ----------------------------
    # 1) Parse and filter correspondence
    # ----------------------------
    if isinstance(rna_protein_correspondence, pd.DataFrame):
        pairs = rna_protein_correspondence.iloc[:, :2].values
    else:
        pairs = np.asarray(rna_protein_correspondence, dtype=object)

    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("rna_protein_correspondence must be 2 columns: [rna_gene, protein_name].")

    rna_set = set(adata_RNA.var_names)
    adt_set = set(adata_ADT.var_names)
    pairs_filtered = np.asarray([[g, p] for g, p in pairs if (g in rna_set) and (p in adt_set)], dtype=object)
    if pairs_filtered.shape[0] == 0:
        raise ValueError("No valid RNA-ADT pairs after filtering by var_names.")

    # ----------------------------
    # 2) Shared arrays (small) + target_sum
    # ----------------------------
    rna_shared = adata_RNA[:, pairs_filtered[:, 0]].copy()
    adt_shared = adata_ADT[:, pairs_filtered[:, 1]].copy()

    # remove static shared features without densifying
    rna_std = _col_std_sparse_safe(rna_shared.X)
    adt_std = _col_std_sparse_safe(adt_shared.X)
    mask = (rna_std > std_eps) & (adt_std > std_eps)
    rna_shared = rna_shared[:, mask].copy()
    adt_shared = adt_shared[:, mask].copy()

    if verbose:
        print(f"[shared] kept shared features={rna_shared.shape[1]}, RSS={_rss_mib():.2f} MiB")

    # target_sum based on medians of row sums
    target_sum = float((np.median(_row_sums(rna_shared.X)) + np.median(_row_sums(adt_shared.X))) / 2.0)

    # preprocess shared (no HVG), then densify (shared features usually ~tens)
    rna_shared = _preprocess_for_pca(rna_shared, target_sum=target_sum, do_hvg=False,
                                    scale_max_value=scale_max_value, verbose=verbose)
    adt_shared = _preprocess_for_pca(adt_shared, target_sum=target_sum, do_hvg=False,
                                    scale_max_value=scale_max_value, verbose=verbose)

    shared_arr1 = _to_dense_float32(rna_shared.X)
    shared_arr2 = _to_dense_float32(adt_shared.X)

    # ----------------------------
    # 3) Active arrays (big) — use PCA to keep dense arrays small
    # ----------------------------
    # RNA active: HVG then PCA
    adata_RNA_p = _preprocess_for_pca(adata_RNA, target_sum=None, do_hvg=True, n_top_genes=hvg_n_top,
                                     scale_max_value=scale_max_value, verbose=verbose)
    # ADT active: usually small, no HVG, PCA optional (still fine)
    adata_ADT_p = _preprocess_for_pca(adata_ADT, target_sum=None, do_hvg=False,
                                     scale_max_value=scale_max_value, verbose=verbose)
    peak_rss = _rss_mib()
    def _update_peak():
        nonlocal peak_rss
        cur = _rss_mib()
        if not np.isnan(cur):
            peak_rss = max(peak_rss, cur)

    t0 = time.time()

    # remove static active features (still sparse-safe)
    rna_std_active = _col_std_sparse_safe(adata_RNA_p.X)
    adt_std_active = _col_std_sparse_safe(adata_ADT_p.X)
    adata_RNA_p = adata_RNA_p[:, rna_std_active > std_eps].copy()
    adata_ADT_p = adata_ADT_p[:, adt_std_active > std_eps].copy()

    if active_use_pca:
        active_arr1 = _compute_pca_dense(adata_RNA_p, n_pcs=active_pcs_rna, verbose=verbose)
        active_arr2 = _compute_pca_dense(adata_ADT_p, n_pcs=active_pcs_adt, verbose=verbose)
    else:
        # Less memory-friendly: dense scaled matrices (not recommended for big N)
        active_arr1 = _to_dense_float32(adata_RNA_p.X)
        active_arr2 = _to_dense_float32(adata_ADT_p.X)

    if verbose:
        print(f"[active] active_arr1={active_arr1.shape}, active_arr2={active_arr2.shape}, RSS={_rss_mib():.2f} MiB")

    # ----------------------------
    # 4) Run MaxFuse + measure time + peak RSS
    # ----------------------------
    

    fusor = mf.model.Fusor(
        shared_arr1=shared_arr1,
        shared_arr2=shared_arr2,
        active_arr1=active_arr1,
        active_arr2=active_arr2,
        labels1=None,
        labels2=None,
    )

    fusor.split_into_batches(**split_params); _update_peak()
    fusor.construct_graphs(**graph_params); _update_peak()
    fusor.find_initial_pivots(**init_pivot_params); _update_peak()
    fusor.refine_pivots(**refine_pivot_params); _update_peak()
    fusor.filter_bad_matches(**filter_bad_match_params); _update_peak()

    rna_cca, adt_cca = fusor.get_embedding(active_arr1=fusor.active_arr1, active_arr2=fusor.active_arr2)
    _update_peak()

    # Integrated embedding AnnData (store only the first dim_use)
    X_int = np.concatenate((rna_cca[:, :dim_use], adt_cca[:, :dim_use]), axis=0).astype(np.float32, copy=False)
    cca_adata = ad.AnnData(X_int)
    cca_adata.obsm[embed_key] = cca_adata.X

    cca_adata.obs[modality_key] = (
        [modality_a_name] * rna_cca.shape[0]
        + [modality_b_name] * adt_cca.shape[0]
    )

    # Keep your previous behavior (only valid if RNA/ADT cells are matched 1:1)
    if celltype_key_source in adata_RNA.obs.columns:
        cca_adata.obs[celltype_key_target] = list(adata_RNA.obs[celltype_key_source]) * 2
    else:
        cca_adata.obs[celltype_key_target] = pd.NA

    t1 = time.time()

    run_info = {
        "total_runtime_min": (t1 - t0) / 60,
        "peak_memory_use": float(peak_rss)/1024,
        "rss_end_mib": float(_rss_mib()),
        "target_sum": float(target_sum),
        "dim_use": int(dim_use),
        "shapes": {
            "shared_arr1": tuple(shared_arr1.shape),
            "shared_arr2": tuple(shared_arr2.shape),
            "active_arr1": tuple(active_arr1.shape),
            "active_arr2": tuple(active_arr2.shape),
            "embedding_rna": tuple(rna_cca.shape),
            "embedding_adt": tuple(adt_cca.shape),
        },
        "params": {
            "split_params": split_params,
            "graph_params": graph_params,
            "init_pivot_params": init_pivot_params,
            "refine_pivot_params": refine_pivot_params,
            "filter_bad_match_params": filter_bad_match_params,
        }
    }

    # ----------------------------
    # 5) Write integrated h5ad + print "link"
    # ----------------------------
    os.makedirs(output_dir, exist_ok=True)
    integrated_path = os.path.join(output_dir, integrated_h5ad_name)
    cca_adata.write(integrated_path)


    return cca_adata, run_info, integrated_path
    
def run_maxfuse_rna_atac_most_memory_efficient(
    adata_RNA: ad.AnnData,
    adata_ATAC: ad.AnnData,
    output_dir: str,
    integrated_h5ad_name: str = "maxfuse_integrated.h5ad",
    *,
    celltype_key_source: str = "celltype",
    celltype_key_target: str = "celltype",
    modality_key: str = "modality",
    embed_key: str = "X_multi",
    modality_a_name: str = "RNA",
    modality_b_name: str = "ATAC",
    std_eps: float = 1e-5,
    dim_use: int = 20,
    hvg_n_top: int = 2000,
    scale_max_value: Optional[float] = 10.0,
    active_use_pca: bool = True,
    active_pcs_rna: int = 30,
    active_pcs_atac: int = 30,
    split_params: Optional[Dict[str, Any]] = None,
    graph_params: Optional[Dict[str, Any]] = None,
    init_pivot_params: Optional[Dict[str, Any]] = None,
    refine_pivot_params: Optional[Dict[str, Any]] = None,
    filter_bad_match_params: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Tuple[ad.AnnData, Dict[str, Any], str]:
    """
    Memory-efficient MaxFuse for RNA + ATAC gene-activity matrices.
    Shared features are built by direct gene-name intersection.
    """

    split_params = split_params or dict(
        max_outward_size=3000,
        matching_ratio=2,
        metacell_size=3,
        verbose=verbose,
    )
    graph_params = graph_params or dict(
        n_neighbors1=10,
        n_neighbors2=10,
        svd_components1=15,
        svd_components2=15,
        resolution1=2,
        resolution2=2,
        resolution_tol=0.1,
        verbose=verbose,
    )
    init_pivot_params = init_pivot_params or dict(
        wt1=0.7, wt2=0.7,
        svd_components1=15, svd_components2=15,
    )
    refine_pivot_params = refine_pivot_params or dict(
        wt1=0.7, wt2=0.7,
        svd_components1=15, svd_components2=15,
        cca_components=10,
        n_iters=2,
        randomized_svd=True,
        svd_runs=1,
        verbose=verbose,
    )
    filter_bad_match_params = filter_bad_match_params or dict(target="pivot", filter_prop=0.3)

    adata_RNA = adata_RNA.copy()
    adata_ATAC = adata_ATAC.copy()

    if verbose:
        print(f"[start] RNA={adata_RNA.shape}, ATAC={adata_ATAC.shape}, RSS={_rss_mib():.2f} MiB")

    # ----------------------------
    # 1) Shared genes by direct name intersection
    # ----------------------------
    shared_genes = sorted(set(adata_RNA.var_names).intersection(set(adata_ATAC.var_names)))
    if len(shared_genes) == 0:
        raise ValueError(
            "No shared genes between RNA and ATAC. "
            "ATAC should be a gene-activity-like matrix with gene names."
        )

    rna_shared = adata_RNA[:, shared_genes].copy()
    atac_shared = adata_ATAC[:, shared_genes].copy()

    # ----------------------------
    # 2) Filter static shared features
    # ----------------------------
    rna_std = _col_std_sparse_safe(rna_shared.X)
    atac_std = _col_std_sparse_safe(atac_shared.X)
    mask = (rna_std > std_eps) & (atac_std > std_eps)

    rna_shared = rna_shared[:, mask].copy()
    atac_shared = atac_shared[:, mask].copy()

    if verbose:
        print(f"[shared] kept shared genes={rna_shared.shape[1]}, RSS={_rss_mib():.2f} MiB")

    # ----------------------------
    # 3) Common target_sum
    # ----------------------------
    rna_counts = _row_sums(rna_shared.X)
    atac_counts = _row_sums(atac_shared.X)
    target_sum = float((np.median(rna_counts) + np.median(atac_counts)) / 2.0)

    # ----------------------------
    # 4) Preprocess shared arrays
    # ----------------------------
    rna_shared = _preprocess_for_pca(
        rna_shared,
        target_sum=target_sum,
        do_hvg=False,
        scale_max_value=scale_max_value,
        verbose=verbose,
    )
    atac_shared = _preprocess_for_pca(
        atac_shared,
        target_sum=target_sum,
        do_hvg=False,
        scale_max_value=scale_max_value,
        verbose=verbose,
    )

    shared_arr1 = _to_dense_float32(rna_shared.X)
    shared_arr2 = _to_dense_float32(atac_shared.X)

    # ----------------------------
    # 5) Active arrays
    # ----------------------------
    adata_RNA_p = _preprocess_for_pca(
        adata_RNA,
        target_sum=None,
        do_hvg=True,
        n_top_genes=hvg_n_top,
        scale_max_value=scale_max_value,
        verbose=verbose,
    )

    adata_ATAC_p = _preprocess_for_pca(
        adata_ATAC,
        target_sum=None,
        do_hvg=True,
        n_top_genes=hvg_n_top,
        scale_max_value=scale_max_value,
        verbose=verbose,
    )

    peak_rss = _rss_mib()

    def _update_peak():
        nonlocal peak_rss
        cur = _rss_mib()
        if not np.isnan(cur):
            peak_rss = max(peak_rss, cur)

    t0 = time.time()

    rna_std_active = _col_std_sparse_safe(adata_RNA_p.X)
    atac_std_active = _col_std_sparse_safe(adata_ATAC_p.X)

    adata_RNA_p = adata_RNA_p[:, rna_std_active > std_eps].copy()
    adata_ATAC_p = adata_ATAC_p[:, atac_std_active > std_eps].copy()

    if active_use_pca:
        active_arr1 = _compute_pca_dense(adata_RNA_p, n_pcs=active_pcs_rna, verbose=verbose)
        active_arr2 = _compute_pca_dense(adata_ATAC_p, n_pcs=active_pcs_atac, verbose=verbose)
    else:
        active_arr1 = _to_dense_float32(adata_RNA_p.X)
        active_arr2 = _to_dense_float32(adata_ATAC_p.X)

    if verbose:
        print(f"[active] active_arr1={active_arr1.shape}, active_arr2={active_arr2.shape}, RSS={_rss_mib():.2f} MiB")

    # ----------------------------
    # 6) Run MaxFuse
    # ----------------------------
    fusor = mf.model.Fusor(
        shared_arr1=shared_arr1,
        shared_arr2=shared_arr2,
        active_arr1=active_arr1,
        active_arr2=active_arr2,
        labels1=None,
        labels2=None,
    )

    fusor.split_into_batches(**split_params); _update_peak()
    fusor.construct_graphs(**graph_params); _update_peak()
    fusor.find_initial_pivots(**init_pivot_params); _update_peak()
    fusor.refine_pivots(**refine_pivot_params); _update_peak()
    fusor.filter_bad_matches(**filter_bad_match_params); _update_peak()

    rna_cca, atac_cca = fusor.get_embedding(
        active_arr1=fusor.active_arr1,
        active_arr2=fusor.active_arr2,
    )
    _update_peak()

    # ----------------------------
    # 7) Integrated AnnData
    # ----------------------------
    X_int = np.concatenate(
        (rna_cca[:, :dim_use], atac_cca[:, :dim_use]),
        axis=0
    ).astype(np.float32, copy=False)

    cca_adata = ad.AnnData(X_int)
    cca_adata.obsm[embed_key] = cca_adata.X

    cca_adata.obs[modality_key] = (
        [modality_a_name] * rna_cca.shape[0]
        + [modality_b_name] * atac_cca.shape[0]
    )

    if celltype_key_source in adata_RNA.obs.columns:
        # paired multiome assumption, same as your old behavior
        cca_adata.obs[celltype_key_target] = list(adata_RNA.obs[celltype_key_source]) * 2
    else:
        cca_adata.obs[celltype_key_target] = pd.NA

    t1 = time.time()

    run_info = {
        "total_runtime_min": (t1 - t0) / 60,
        "peak_memory_use": float(peak_rss) / 1024,
        "rss_end_mib": float(_rss_mib()),
        "target_sum": float(target_sum),
        "dim_use": int(dim_use),
        "n_shared_features": int(len(shared_genes)),
        "shapes": {
            "shared_arr1": tuple(shared_arr1.shape),
            "shared_arr2": tuple(shared_arr2.shape),
            "active_arr1": tuple(active_arr1.shape),
            "active_arr2": tuple(active_arr2.shape),
            "embedding_rna": tuple(rna_cca.shape),
            "embedding_atac": tuple(atac_cca.shape),
        },
        "params": {
            "split_params": split_params,
            "graph_params": graph_params,
            "init_pivot_params": init_pivot_params,
            "refine_pivot_params": refine_pivot_params,
            "filter_bad_match_params": filter_bad_match_params,
        }
    }

    os.makedirs(output_dir, exist_ok=True)
    integrated_path = os.path.join(output_dir, integrated_h5ad_name)
    cca_adata.write(integrated_path)

    return cca_adata, run_info, integrated_path

MODEL_FILE_EXT = "txt"


def add_method_args(parser):
    parser.add_argument("--dim_use", type=int, default=20)
    parser.add_argument("--std_eps", type=float, default=1e-5)
    parser.add_argument("--hvg_n_top", type=int, default=2000)
    parser.add_argument("--scale_max_value", type=float, default=10.0)

    parser.add_argument("--active_use_pca", action="store_true", default=True)
    parser.add_argument("--active_pcs_rna", type=int, default=20)
    parser.add_argument("--active_pcs_mod2", type=int, default=20)
    

def _load_pairs_from_csv(adata_rna, adata_adt, correspondence_path):
    import pandas as pd
    import numpy as np

    correspondence = pd.read_csv(correspondence_path)
    pairs = []

    # same logic as your other methods
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
    adata_rna,
    adata_mod2,
    correspondence_path,
    dataset_name,
    modality_a_name,
    modality_b_name,
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
            "method_preprocessing": "internal_maxfuse_rna_adt",
        }

        prepared_inputs = {
            "mode": "rna_adt",
            "pairs": pairs
        }

    elif preprocess_mode == "rna_atac":
        shared_genes = sorted(set(adata_rna.var_names).intersection(set(adata_mod2.var_names)))
        if len(shared_genes) == 0:
            raise ValueError(
                "No shared genes between RNA and ATAC for MaxFuse. "
                "ATAC should be gene-activity-like with gene names."
            )

        preprocess_info = {
            "dataset": dataset_name,
            "n_shared_features": int(len(shared_genes)),
            "method_preprocessing": "internal_maxfuse_rna_atac",
        }

        prepared_inputs = {
            "mode": "rna_atac",
            "shared_genes": shared_genes
        }

    else:
        raise ValueError(f"Unsupported preprocess_mode: {preprocess_mode}")

    return prepared_inputs, preprocess_info


def run_method(
    prepared_inputs,
    adata_rna_raw,
    adata_mod2_raw,
    output_paths,
    modality_a_name,
    modality_b_name,
    args,
):
    if prepared_inputs["mode"] == "rna_adt":
        adata_integrated, run_info, integrated_path = run_maxfuse_rna_adt_most_memory_efficient(
            adata_RNA=adata_rna_raw,
            adata_ADT=adata_mod2_raw,
            rna_protein_correspondence=prepared_inputs["pairs"],
            output_dir=output_paths["base_dir"],
            integrated_h5ad_name=os.path.basename(output_paths["integrated_h5ad"]),
            celltype_key_source="celltype",
            celltype_key_target="celltype",
            modality_key="modality",
            embed_key="X_multi",
            modality_a_name=modality_a_name,
            modality_b_name=modality_b_name,
            std_eps=args.std_eps,
            dim_use=args.dim_use,
            hvg_n_top=args.hvg_n_top,
            scale_max_value=args.scale_max_value,
            active_use_pca=args.active_use_pca,
            active_pcs_rna=args.active_pcs_rna,
            active_pcs_adt=args.active_pcs_mod2,
            verbose=True,
        )

    elif prepared_inputs["mode"] == "rna_atac":
        adata_integrated, run_info, integrated_path = run_maxfuse_rna_atac_most_memory_efficient(
            adata_RNA=adata_rna_raw,
            adata_ATAC=adata_mod2_raw,
            output_dir=output_paths["base_dir"],
            integrated_h5ad_name=os.path.basename(output_paths["integrated_h5ad"]),
            celltype_key_source="celltype",
            celltype_key_target="celltype",
            modality_key="modality",
            embed_key="X_multi",
            modality_a_name=modality_a_name,
            modality_b_name=modality_b_name,
            std_eps=args.std_eps,
            dim_use=args.dim_use,
            hvg_n_top=args.hvg_n_top,
            scale_max_value=args.scale_max_value,
            active_use_pca=args.active_use_pca,
            active_pcs_rna=args.active_pcs_rna,
            active_pcs_atac=args.active_pcs_mod2,
            verbose=True,
        )

    else:
        raise ValueError(f"Unsupported prepared_inputs mode: {prepared_inputs['mode']}")

    embedding_df = pd.DataFrame(
        adata_integrated.obsm["X_multi"],
        index=adata_integrated.obs_names,
        columns=[f"latent_{i+1}" for i in range(adata_integrated.obsm["X_multi"].shape[1])]
    )
    embedding_df.insert(0, "modality", adata_integrated.obs["modality"].astype(str).values)

    run_info["model_ckpt_path"] = output_paths["model"]
    run_info["integrated_h5ad_path"] = integrated_path

    return None, adata_integrated, embedding_df, run_info