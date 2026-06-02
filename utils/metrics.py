import warnings
warnings.filterwarnings("ignore")

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import scipy
from scipy.spatial.distance import cdist
from sklearn import preprocessing
from sklearn.neighbors import KDTree, NearestNeighbors

import scib

# Optional R-backed packages for kBET/lisi through scib
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.conversion import localconverter

rscript = """
library(kBET)
library(lisi)
"""
ro.r(rscript)
compute_lisi = ro.r["compute_lisi"]


# =========================================================
# Basic helpers
# =========================================================

def to_dense_array(X):
    if scipy.sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)


def ensure_label_column(adata, preferred: str = "celltype_l2", fallback: str = "celltype"):
    """
    Make sure a label column exists and return its name.
    """
    if preferred in adata.obs.columns:
        return preferred
    if fallback in adata.obs.columns:
        return fallback
    raise ValueError(
        f"Neither '{preferred}' nor '{fallback}' exists in adata.obs. "
        f"Available columns include: {list(adata.obs.columns)}"
    )


def ensure_batch_column(adata, batch_key: str = "modality", modality_key: str = "modality"):
    """
    If batch column is missing, fall back to modality.
    """
    if batch_key in adata.obs.columns:
        return batch_key
    if modality_key in adata.obs.columns:
        adata.obs[batch_key] = adata.obs[modality_key].astype(str)
        return batch_key
    raise ValueError(
        f"Neither batch key '{batch_key}' nor modality key '{modality_key}' exists in adata.obs."
    )


# =========================================================
# Positive / true-positive metrics
# =========================================================

def positive_true_positive(
    adata,
    batch_key: str = "modality",
    celltype_key: str = "celltype",
    use_raw: bool = False,
    k1: int = 20,
    k2: int = 100,
    tp_thr: float = 3.0,
    distance: str = "cosine",
    embed: str = "X_pca",
):
    """
    Positive-cell rate and true-positive-cell rate, following your iMAP-style definition.
    """
    celltype_list = adata.obs[celltype_key]
    batch_list = adata.obs[batch_key]

    temp_c = adata.obs[celltype_key].value_counts()
    temp_b = pd.crosstab(adata.obs[celltype_key], adata.obs[batch_key])
    temp_b_prob = temp_b.divide(temp_b.sum(1), axis=0)

    if use_raw:
        X = to_dense_array(adata.X)
    else:
        X = adata.obsm[embed]

    if distance == "cosine":
        X = preprocessing.normalize(X, axis=1)

    tree = KDTree(X)

    p_list = []
    tp_list = []

    for cell in range(len(X)):
        neig1 = min(k1, temp_c[celltype_list.iloc[cell]])
        NNs = tree.query(X[cell].reshape(1, -1), neig1 + 1, return_distance=False)[0, 1:]
        c_NN = celltype_list.iloc[NNs]
        true_rate = np.sum(c_NN == celltype_list.iloc[cell]) / neig1

        if true_rate > 0.5:
            p_list.append(True)
        else:
            p_list.append(False)

        if p_list[cell]:
            neig2 = min(k2, temp_c[celltype_list.iloc[cell]])
            NNs = tree.query(X[cell].reshape(1, -1), neig2, return_distance=False)[0]
            NNs_c = celltype_list.iloc[NNs]
            NNs_i = (NNs_c == celltype_list.iloc[cell]).values
            NNs = NNs[NNs_i]
            neig2 = len(NNs)
            NNs_b = batch_list.iloc[NNs]

            max_b = 0.0
            b_prob = temp_b_prob.loc[celltype_list.iloc[cell]]
            for b in set(batch_list):
                if b_prob[b] > 0 and b_prob[b] < 1:
                    p_b = np.sum(NNs_b == b)
                    stat_b = abs(p_b - neig2 * b_prob[b]) / np.sqrt(neig2 * b_prob[b] * (1 - b_prob[b]))
                    max_b = max(max_b, stat_b)

            tp_list.append(max_b <= tp_thr)
        else:
            tp_list.append(False)

    pos_rate = float(np.sum(p_list) / len(p_list))
    truepos_rate = float(np.sum(tp_list) / len(tp_list))
    return pos_rate, truepos_rate


# =========================================================
# LISI metric
# =========================================================

def calculate_lisi_batch_only(
    adata,
    labels=None,
    total_cells=None,
    batch_key: str = "modality",
    celltype_key: Optional[str] = "celltype",
    embed: str = "X_multi",
):
    """
    Calculate only batch LISI (lisi_b), following your original logic.

    When celltype_key is provided:
      - compute batch LISI within each cell type
      - average weighted by cell count

    When celltype_key is None:
      - compute global batch LISI
    """
    adata = adata.copy()
    batch_key = ensure_batch_column(adata, batch_key=batch_key)
    if celltype_key is not None:
        celltype_key = ensure_label_column(adata, preferred=celltype_key, fallback="celltype")

    if embed not in adata.obsm:
        raise ValueError(f"Embedding '{embed}' not found in adata.obsm.")

    if celltype_key is None:
        if adata.shape[0] < 90:
            perplexity = max(2, int(adata.shape[0] / 6))
        else:
            perplexity = 30

        X = np.asarray(adata.obsm[embed], dtype=float)
        meta = adata.obs[[batch_key]].copy()
        meta[batch_key] = meta[batch_key].astype(str)

        with localconverter(ro.default_converter + numpy2ri.converter + pandas2ri.converter):
            lisi_res = compute_lisi(X, meta, batch_key, perplexity=perplexity)

        lisi_res = np.array(lisi_res)
        denom = len(set(meta[batch_key])) - 1
        if denom <= 0:
            return np.nan

        lisi_b = (np.mean(lisi_res) - 1.0) / denom
        return float(lisi_b)

    if labels is None:
        labels = list(adata.obs[celltype_key].astype(str).unique())
    if total_cells is None:
        total_cells = adata.shape[0]

    lisi_b = 0.0

    for label in labels:
        adata_sub = adata[adata.obs[celltype_key].astype(str) == str(label)].copy()

        if adata_sub.shape[0] < 2:
            continue

        n_batches = len(set(adata_sub.obs[batch_key]))
        if n_batches <= 1:
            continue

        if adata_sub.shape[0] < 90:
            perplexity = max(2, int(adata_sub.shape[0] / 6))
        else:
            perplexity = 30

        X_sub = np.asarray(adata_sub.obsm[embed], dtype=float)
        meta_sub = adata_sub.obs[[batch_key]].copy()
        meta_sub[batch_key] = meta_sub[batch_key].astype(str)

        with localconverter(ro.default_converter + numpy2ri.converter + pandas2ri.converter):
            lisi_res = compute_lisi(X_sub, meta_sub, batch_key, perplexity=perplexity)

        lisi_res = np.array(lisi_res)
        lisi_batch = (np.mean(lisi_res) - 1.0) / (n_batches - 1.0)
        lisi_b += float(lisi_batch) * adata_sub.shape[0]

    lisi_b /= float(total_cells)
    return float(lisi_b)

# =========================================================
# scIB + clustering metrics
# =========================================================

def compute_scib_style_metrics(
    adata,
    batch_key: str = "modality",
    celltype_key: str = "celltype",
    embed: str = "X_multi",
    n_neighbors: int = 15,
    tp_thr: float = 3.0,
) -> Dict[str, float]:
    """
    Compute:
      - ASW_label
      - ARI
      - NMI
      - ASW_batch
      - kBET Accept Rate
      - pos rate
      - true pos rate
      - LISI_batch
    """
    adata = adata.copy()

    batch_key = ensure_batch_column(adata, batch_key=batch_key)
    celltype_key = ensure_label_column(adata, preferred=celltype_key, fallback="celltype")

    adata.obs[celltype_key] = adata.obs[celltype_key].astype("category")
    adata.obs[batch_key] = adata.obs[batch_key].astype("category")

    # Skip tiny / single-batch cell types for robustness
    labels = set(adata.obs[celltype_key])
    labels_keep = set(labels)
    total_cells = adata.shape[0]

    for label in list(labels):
        adata_sub = adata[adata.obs[celltype_key] == label]
        if len(set(adata_sub.obs[batch_key])) == 1 or adata_sub.shape[0] < 10:
            print(f"Cell cluster {label} contains only one batch or has less than 10 cells. Skip.")
            total_cells -= adata_sub.shape[0]
            labels_keep.remove(label)

    if len(labels_keep) < len(labels):
        adata = adata[adata.obs[celltype_key].isin(labels_keep)].copy()

    print("ASW / kBET / clustering metrics...")
    asw_label = float(
        scib.metrics.silhouette(
            adata,
            label_key=celltype_key,
            embed=embed,
            metric="euclidean",
        )
    )

    asw_batch = float(
        scib.metrics.silhouette_batch(
            adata,
            batch_key=batch_key,
            label_key=celltype_key,
            embed=embed,
            metric="euclidean",
            return_all=False,
            verbose=False,
        )
    )

    kbet_score = float(
        scib.metrics.kBET(
            adata,
            batch_key=batch_key,
            label_key=celltype_key,
            type_=None,
            embed=embed,
            scaled=True,
            verbose=False,
        )
    )

    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=embed)

    scib.cl.opt_louvain(
        adata,
        label_key=celltype_key,
        cluster_key="cluster",
        plot=False,
        inplace=True,
        force=True,
        verbose=False,
    )

    nmi = float(scib.me.nmi(adata, group1="cluster", group2=celltype_key))
    ari = float(scib.me.ari(adata, group1="cluster", group2=celltype_key))

    pos_rate, truepos_rate = positive_true_positive(
        adata,
        batch_key=batch_key,
        celltype_key=celltype_key,
        k1=20,
        k2=100,
        tp_thr=tp_thr,
        embed=embed,
    )

    lisi_b = calculate_lisi_batch_only(
        adata,
        labels=list(adata.obs[celltype_key].astype(str).unique()),
        total_cells=adata.shape[0],
        batch_key=batch_key,
        celltype_key=celltype_key,
        embed=embed,
    )

    return {
        "ASW_label": asw_label,
        "ARI": ari,
        "NMI": nmi,
        "ASW_batch": asw_batch,
        "kBET Accept Rate": kbet_score,
        "pos rate": pos_rate,
        "true pos rate": truepos_rate,
        "LISI_batch": lisi_b,
    }


# =========================================================
# Cross-modality pairing metrics
# =========================================================

def align_obs_names_for_two_modalities(
    adata,
    modality_col: str = "modality",
    modalities: Tuple[str, str] = ("RNA", "ADT"),
):
    """
    Force the two modalities to share the same obs_names in the same order.
    This is useful when downstream pair metrics assume matched cells share obs_names.

    Strategy:
      - subset each modality
      - require same number of cells
      - use obs_names from modality 1 for both
    """
    adata = adata.copy()
    m1, m2 = modalities

    adata1 = adata[adata.obs[modality_col].astype(str) == m1].copy()
    adata2 = adata[adata.obs[modality_col].astype(str) == m2].copy()

    if adata1.n_obs != adata2.n_obs:
        raise ValueError(
            f"Cannot force paired obs_names because {m1} has {adata1.n_obs} cells "
            f"but {m2} has {adata2.n_obs} cells."
        )

    new_names = adata1.obs_names.astype(str).tolist() + adata1.obs_names.astype(str).tolist()
    adata.obs_names = new_names
    return adata


def compute_integration_metrics_two_modalities(
    adata_integrated,
    modality_col: str = "modality",
    label_col: str = "celltype_l2",
    modalities: Tuple[str, str] = ("RNA", "ADT"),
    embed: str = "X_multi",
) -> Dict[str, float]:
    """
    Compute:
      - Label Transfer Accuracy (both directions, then average)
      - Average Pair Distance
      - Average FOSCTTM
    Assumes matched cells share obs_names across the two modalities.
    """
    if label_col not in adata_integrated.obs.columns:
        if "celltype" in adata_integrated.obs.columns:
            label_col = "celltype"
        else:
            raise ValueError(
                f"Label column '{label_col}' not found and no fallback 'celltype'."
            )

    m1, m2 = modalities
    all_modalities = set(adata_integrated.obs[modality_col].astype(str).unique())
    if m1 not in all_modalities or m2 not in all_modalities:
        raise ValueError(f"Modalities {modalities} not both present; found {sorted(all_modalities)}.")

    X = adata_integrated.obsm[embed] if embed in adata_integrated.obsm else adata_integrated.X
    X = to_dense_array(X)

    obs_df = pd.DataFrame({
        "obs_name": adata_integrated.obs_names.astype(str),
        "modality": adata_integrated.obs[modality_col].astype(str).values,
        "label": adata_integrated.obs[label_col].astype(str).values,
        "original_index": np.arange(adata_integrated.n_obs, dtype=int),
    })

    df1 = obs_df[obs_df["modality"] == m1].copy()
    df2 = obs_df[obs_df["modality"] == m2].copy()

    common_names = set(df1["obs_name"]).intersection(set(df2["obs_name"]))
    if len(common_names) == 0:
        return {
            "Label Transfer Accuracy": np.nan,
            "Average Pair Distance": np.nan,
            "Average FOSCTTM": np.nan,
            f"acc_{m1}_to_{m2}": np.nan,
            f"acc_{m2}_to_{m1}": np.nan,
            "n_pairs": 0,
        }

    df1a = df1.set_index("obs_name").loc[sorted(common_names)]
    df2a = df2.set_index("obs_name").loc[sorted(common_names)]

    idx1 = df1a["original_index"].to_numpy()
    idx2 = df2a["original_index"].to_numpy()
    n_pairs = idx1.shape[0]

    X1 = X[idx1, :]
    X2 = X[idx2, :]
    labels1 = df1a["label"].to_numpy()
    labels2 = df2a["label"].to_numpy()

    # 1-NN label transfer
    nbrs_1 = NearestNeighbors(n_neighbors=1).fit(X1)
    _, to_1 = nbrs_1.kneighbors(X2)
    acc_m2_to_m1 = float(np.mean(labels1[to_1.ravel()] == labels2))

    nbrs_2 = NearestNeighbors(n_neighbors=1).fit(X2)
    _, to_2 = nbrs_2.kneighbors(X1)
    acc_m1_to_m2 = float(np.mean(labels2[to_2.ravel()] == labels1))

    avg_acc = float((acc_m1_to_m2 + acc_m2_to_m1) / 2.0)

    # Pair distance
    D12 = cdist(X1, X2)
    true_d = np.diag(D12)
    sum_row = D12.sum(axis=1)
    sum_col = D12.sum(axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        term_row = np.where(sum_row > 0, true_d / sum_row, 0.0)
        term_col = np.where(sum_col > 0, true_d / sum_col, 0.0)

    pair_dist_vec = (n_pairs / 2.0) * (term_row + term_col)
    pair_distance = float(np.mean(pair_dist_vec))

    # FOSCTTM
    closer_2 = (D12 < true_d[:, None]).sum(axis=1) - 1
    closer_1 = (D12 < true_d[None, :]).sum(axis=0) - 1
    closer_2 = np.clip(closer_2, 0, None)
    closer_1 = np.clip(closer_1, 0, None)
    fos_i = (closer_1 + closer_2) / (2.0 * n_pairs)
    foscttm = float(np.mean(fos_i))

    return {
        "Label Transfer Accuracy": avg_acc,
        "Average Pair Distance": pair_distance,
        "Average FOSCTTM": foscttm,
        f"acc_{m1}_to_{m2}": acc_m1_to_m2,
        f"acc_{m2}_to_{m1}": acc_m2_to_m1,
        "n_pairs": int(n_pairs),
    }


# =========================================================
# Runtime / memory helpers
# =========================================================

def format_runtime_metrics(
    peak_memory_bytes: Optional[float] = None,
    training_time_sec: Optional[float] = None,
) -> Dict[str, float]:
    """
    Convert raw runtime stats into the required display metrics.
    """
    out = {}

    if peak_memory_bytes is None or pd.isna(peak_memory_bytes):
        out["Peak Memory (GiB)"] = np.nan
    else:
        out["Peak Memory (GiB)"] = float(peak_memory_bytes) / (1024 ** 3)

    if training_time_sec is None or pd.isna(training_time_sec):
        out["Training Time (min)"] = np.nan
    else:
        out["Training Time (min)"] = float(training_time_sec) / 60.0

    return out


# =========================================================
# One-stop combined evaluation
# =========================================================

def summarize_all_metrics(
    adata_integrated,
    *,
    method: str,
    dataset: str,
    modalities: str,
    version: str,
    embed: str = "X_multi",
    batch_key: str = "modality",
    celltype_key: str = "celltype",
    modality_col: str = "modality",
    pair_modalities: Tuple[str, str] = ("RNA", "ADT"),
    peak_memory_bytes: Optional[float] = None,
    training_time_sec: Optional[float] = None,
    force_shared_obs_names: bool = False,
) -> pd.DataFrame:
    """
    Final combined one-row summary table.
    """
    adata_eval = adata_integrated.copy()

    if embed not in adata_eval.obsm:
        adata_eval.obsm[embed] = to_dense_array(adata_eval.X)

    # scIB-style metrics
    scib_metrics = compute_scib_style_metrics(
        adata_eval,
        batch_key=batch_key,
        celltype_key=celltype_key,
        embed=embed,
    )

    # Paired modality metrics
    if force_shared_obs_names:
        adata_pair = align_obs_names_for_two_modalities(
            adata_eval,
            modality_col=modality_col,
            modalities=pair_modalities,
        )
    else:
        adata_pair = adata_eval

    pair_metrics = compute_integration_metrics_two_modalities(
        adata_pair,
        modality_col=modality_col,
        label_col=celltype_key,
        modalities=pair_modalities,
        embed=embed,
    )

    runtime_metrics = format_runtime_metrics(
        peak_memory_bytes=peak_memory_bytes,
        training_time_sec=training_time_sec,
    )

    m1, m2 = pair_modalities

    result = {
        "method": method,
        "dataset": dataset,
        "modalities": modalities,
        "version": version,

        "ASW_label": scib_metrics["ASW_label"],
        "ARI": scib_metrics["ARI"],
        "NMI": scib_metrics["NMI"],
        "ASW_batch": scib_metrics["ASW_batch"],
        "kBET Accept Rate": scib_metrics["kBET Accept Rate"],
        "pos rate": scib_metrics["pos rate"],
        "true pos rate": scib_metrics["true pos rate"],
        "LISI_batch": scib_metrics["LISI_batch"],

        f"Label Transfer Accuracy ({m1}→{m2})": pair_metrics.get(f"acc_{m1}_to_{m2}", pd.NA),
        f"Label Transfer Accuracy ({m2}→{m1})": pair_metrics.get(f"acc_{m2}_to_{m1}", pd.NA),

        "Average Pair Distance": pair_metrics["Average Pair Distance"],
        "Average FOSCTTM": pair_metrics["Average FOSCTTM"],

        "Peak Memory (GiB)": runtime_metrics["Peak Memory (GiB)"],
        "Training Time (min)": runtime_metrics["Training Time (min)"],
    }

    return pd.DataFrame([result])