import os
import json
import textwrap
import subprocess
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc

from utils.preprocess_scalemap import build_scalemap_inputs
from utils.preprocess_scalemap_atac import build_scalemap_inputs_atac


MODEL_FILE_EXT = "txt"


def add_method_args(parser):
    # same preprocessing args as scalemap / scmodal
    parser.add_argument("--hvg_top_genes", type=int, default=2000)
    parser.add_argument("--cluster_resolution", type=float, default=0.5)
    parser.add_argument("--cluster_method", type=str, default="leiden")
    parser.add_argument("--final_scale_max_value", type=float, default=10.0)

    # bindSC args
    parser.add_argument("--bindsc_alpha", type=float, default=0.1)
    parser.add_argument("--bindsc_lambda", type=float, default=0.7)
    parser.add_argument("--bindsc_K", type=int, default=15)
    parser.add_argument("--bindsc_num_iteration", type=int, default=50)
    parser.add_argument("--bindsc_tolerance", type=float, default=0.01)
    parser.add_argument("--bindsc_block_size", type=int, default=0)

    parser.add_argument("--bindsc_temp_dirname", type=str, default="bindsc_tmp")
    parser.add_argument("--rscript_executable", type=str, default="Rscript")


def prepare_inputs(
    adata_rna: ad.AnnData,
    adata_mod2: ad.AnnData,
    correspondence_path: str,
    dataset_name: str,
    modality_a_name: str,
    modality_b_name: str,
    args,
):
    """
    bindSC uses the same preprocessing pipeline as scalemap/scmodal.

    RNA-ADT:
        build_scalemap_inputs(...)
    RNA-ATAC:
        build_scalemap_inputs_atac(...)
    """
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

        adata1, adata2, preprocess_info = build_scalemap_inputs(
            adata_rna=adata_rna,
            adata_adt=adata_mod2,
            correspondence_path=correspondence_path,
            dataset_name=dataset_name,
            hvg_top_genes=args.hvg_top_genes,
            cluster_resolution=args.cluster_resolution,
            cluster_method=args.cluster_method,
            final_scale_max_value=args.final_scale_max_value,
        )

    elif preprocess_mode == "rna_atac":
        adata1, adata2, preprocess_info = build_scalemap_inputs_atac(
            adata_rna=adata_rna,
            adata_atac=adata_mod2,
            dataset_name=dataset_name,
            hvg_top_genes=args.hvg_top_genes,
            cluster_resolution=args.cluster_resolution,
            cluster_method=args.cluster_method,
            final_scale_max_value=args.final_scale_max_value,
        )

    else:
        raise ValueError(f"Unsupported preprocess_mode: {preprocess_mode}")

    prepared_inputs = {
        "adata1": adata1,  # RNA processed
        "adata2": adata2,  # modality 2 processed
        "shared_feature_num": int(preprocess_info["shared_feature_num"]),
    }
    return prepared_inputs, preprocess_info


def _write_bindsc_r_script(script_path: str):
    r_code = r'''
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 14) {
  stop("Expected 14 positional arguments:
       rna_h5ad mod2_h5ad shared_feature_num integrated_h5ad runtime_json model_txt temp_dir
       alpha lambda K num_iteration tolerance block_size modality2_name")
}

rna_h5ad           <- args[[1]]
mod2_h5ad          <- args[[2]]
shared_feature_num <- as.integer(args[[3]])
integrated_h5ad    <- args[[4]]
runtime_json       <- args[[5]]
model_txt          <- args[[6]]
temp_dir           <- args[[7]]
alpha              <- as.numeric(args[[8]])
lambda_val         <- as.numeric(args[[9]])
K_val              <- as.integer(args[[10]])
num_iteration      <- as.integer(args[[11]])
tolerance          <- as.numeric(args[[12]])
block_size         <- as.integer(args[[13]])
modality2_name     <- args[[14]]

suppressPackageStartupMessages({
  library(zellkonverter)
  library(bindSC)
  library(Matrix)
  library(peakRAM)
  library(jsonlite)
  library(SingleCellExperiment)
  library(SummarizedExperiment)
})

dir.create(temp_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(dirname(integrated_h5ad), recursive = TRUE, showWarnings = FALSE)
dir.create(dirname(runtime_json), recursive = TRUE, showWarnings = FALSE)
dir.create(dirname(model_txt), recursive = TRUE, showWarnings = FALSE)

sce_rna  <- zellkonverter::readH5AD(rna_h5ad,  verbose = FALSE)
sce_mod2 <- zellkonverter::readH5AD(mod2_h5ad, verbose = FALSE)

rownames(sce_rna)  <- make.unique(rownames(sce_rna))
rownames(sce_mod2) <- make.unique(rownames(sce_mod2))

x_all <- SummarizedExperiment::assay(sce_mod2, "X")  # modality 2 processed, full matrix
y_all <- SummarizedExperiment::assay(sce_rna,  "X")  # RNA processed, full matrix

if (!inherits(x_all, "dgCMatrix")) {
  x_all <- Matrix::Matrix(x_all, sparse = TRUE)
}
if (!inherits(y_all, "dgCMatrix")) {
  y_all <- Matrix::Matrix(y_all, sparse = TRUE)
}

if (!identical(colnames(x_all), colnames(y_all))) {
  stop("RNA and modality-2 cells are not identical and in the same order.")
}

cell_ids <- colnames(x_all)

# Keep full modality 2 / RNA for bindSC
x <- x_all[seq_len(shared_feature_num), , drop = FALSE]
y <- y_all[, , drop = FALSE]

# shared block comes from the FIRST shared_feature_num RNA features
shared_feature_num <- min(shared_feature_num, nrow(y))
if (shared_feature_num <= 0) {
  stop("shared_feature_num must be positive.")
}
z0 <- y[seq_len(shared_feature_num), , drop = FALSE]

# Reuse Python-side preprocessing clusters
rna_cd  <- as.data.frame(colData(sce_rna))
mod2_cd <- as.data.frame(colData(sce_mod2))

rownames(rna_cd)  <- colnames(sce_rna)
rownames(mod2_cd) <- colnames(sce_mod2)

pick_cluster_col <- function(df) {
  for (nm in c("leiden1", "louvain1", "leiden2", "louvain2")) {
    if (nm %in% colnames(df)) {
      return(nm)
    }
  }
  return(NULL)
}

rna_cluster_col  <- pick_cluster_col(rna_cd)
mod2_cluster_col <- pick_cluster_col(mod2_cd)

if (is.null(rna_cluster_col) || is.null(mod2_cluster_col)) {
  stop(sprintf(
    "Could not find preprocessing cluster columns. RNA cols: %s | MOD2 cols: %s",
    paste(colnames(rna_cd), collapse = ", "),
    paste(colnames(mod2_cd), collapse = ", ")
  ))
}

y.clst <- factor(rna_cd[cell_ids,  rna_cluster_col,  drop = TRUE])
x.clst <- factor(mod2_cd[cell_ids, mod2_cluster_col, drop = TRUE])

pk <- peakRAM::peakRAM({
  res <- BiCCA(
    X = x,
    Y = y,
    Z0 = z0,
    X.clst = x.clst,
    Y.clst = y.clst,
    alpha = alpha,
    lambda = lambda_val,
    K = K_val,
    temp.path = temp_dir,
    num.iteration = num_iteration,
    tolerance = tolerance,
    save = TRUE,
    parameter.optimize = FALSE,
    block.size = block_size
  )
})

# keep order: modality 2 first, RNA second
bindsc_embedding <- rbind(res$u, res$r)

cell_ids2  <- c(cell_ids, cell_ids)
modalities <- c(rep(modality2_name, length(cell_ids)), rep("RNA", length(cell_ids)))

counts_matrix <- Matrix::Matrix(t(bindsc_embedding), sparse = TRUE)
rownames(counts_matrix) <- paste0("bindsc_latent_", seq_len(nrow(counts_matrix)))
colnames(counts_matrix) <- cell_ids2

sce_int <- SingleCellExperiment::SingleCellExperiment(
  assays = list(X = counts_matrix)
)
colnames(sce_int) <- cell_ids2
colData(sce_int)$modality <- modalities
reducedDim(sce_int, "X_multi") <- bindsc_embedding

if (ncol(rowData(sce_int)) == 0) {
  rowData(sce_int)$gene_id <- rownames(sce_int)
}

zellkonverter::writeH5AD(sce_int, file = integrated_h5ad)

runtime_info <- list(
  total_runtime_min = as.numeric(pk$Elapsed_Time_sec[1]) / 60.0,
  peak_memory_use   = as.numeric(pk$Peak_RAM_Used_MiB[1]) / 1024.0
)

jsonlite::write_json(runtime_info, runtime_json, auto_unbox = TRUE, pretty = TRUE)

writeLines(
  c(
    "bindSC model artifact placeholder",
    paste("integrated_h5ad:", integrated_h5ad),
    paste("shared_feature_num:", shared_feature_num),
    paste("alpha:", alpha),
    paste("lambda:", lambda_val),
    paste("K:", K_val),
    paste("num_iteration:", num_iteration),
    paste("tolerance:", tolerance),
    paste("block_size:", block_size),
    paste("modality2_name:", modality2_name)
  ),
  con = model_txt
)
'''
    with open(script_path, "w") as f:
        f.write(textwrap.dedent(r_code))

def _clean_for_zellkonverter(adata: ad.AnnData) -> ad.AnnData:
    """
    Strip AnnData to essentials so zellkonverter can read it robustly.
    Keep:
      - X
      - layers['counts'] if available
      - obs / var
    Remove problematic uns/obsm/etc.
    """
    adata = adata.copy()

    X = adata.X
    counts = adata.layers.get("counts", None)

    obs = adata.obs.copy()
    var = adata.var.copy()

    new = ad.AnnData(X=X, obs=obs, var=var)

    if counts is not None:
        new.layers["counts"] = counts

    new.uns = {}
    new.obsm = {}
    new.varm = {}
    new.obsp = {}
    new.varp = {}

    new.obs_names_make_unique()
    new.var_names_make_unique()

    for col in new.obs.columns:
        if isinstance(new.obs[col].dtype, pd.CategoricalDtype):
            new.obs[col] = new.obs[col].astype(str)

    for col in new.var.columns:
        if isinstance(new.var[col].dtype, pd.CategoricalDtype):
            new.var[col] = new.var[col].astype(str)

    return new


def _resolve_celltype_col(adata: ad.AnnData):
    for key in ["celltype", "celltype.l2", "celltype_l2", "celltype.l1", "celltype_l1"]:
        if key in adata.obs.columns:
            return key
    return None


def _restore_final_obs_from_python(
    adata_integrated: ad.AnnData,
    adata_rna_raw: ad.AnnData,
    adata_mod2_raw: ad.AnnData,
    modality_a_name: str = "RNA",
    modality_b_name: str = "ADT",
) -> ad.AnnData:
    """
    Replace R-generated obs with original Python metadata.
    Assumes integrated order is modality 2 first, RNA second, with identical cell IDs repeated.
    """
    adata_integrated = adata_integrated.copy()

    n_total = adata_integrated.n_obs
    if n_total % 2 != 0:
        raise ValueError(f"Expected even number of integrated cells for two modalities, got {n_total}.")

    n = n_total // 2
    cell_ids = list(adata_integrated.obs_names[:n])

    mod2_obs = adata_mod2_raw.obs.loc[cell_ids].copy()
    rna_obs = adata_rna_raw.obs.loc[cell_ids].copy()

    mod2_celltype_col = _resolve_celltype_col(adata_mod2_raw)
    rna_celltype_col = _resolve_celltype_col(adata_rna_raw)

    if mod2_celltype_col is not None:
        mod2_obs["celltype"] = mod2_obs[mod2_celltype_col].astype(str)
    if rna_celltype_col is not None:
        rna_obs["celltype"] = rna_obs[rna_celltype_col].astype(str)

    mod2_obs["modality"] = modality_b_name
    rna_obs["modality"] = modality_a_name

    final_obs = pd.concat([mod2_obs, rna_obs], axis=0)
    final_obs.index = cell_ids + cell_ids

    adata_integrated.obs = final_obs
    adata_integrated.obs_names = final_obs.index.astype(str)

    return adata_integrated
    

def run_method(
    prepared_inputs,
    adata_rna_raw: ad.AnnData,
    adata_mod2_raw: ad.AnnData,
    output_paths: Dict[str, str],
    modality_a_name: str,
    modality_b_name: str,
    args,
):
    """
    Python wrapper that:
      - writes processed inputs to disk
      - calls R bindSC
      - loads integrated h5ad
      - restores original Python metadata
      - returns outputs like other methods
    """
    base_dir = output_paths["base_dir"]
    os.makedirs(base_dir, exist_ok=True)

    bindsc_input_dir = os.path.join(base_dir, "bindsc_inputs")
    os.makedirs(bindsc_input_dir, exist_ok=True)

    rna_processed_h5ad = os.path.join(bindsc_input_dir, "RNA_processed.h5ad")
    mod2_processed_h5ad = os.path.join(bindsc_input_dir, f"{modality_b_name}_processed.h5ad")
    shared_json = os.path.join(bindsc_input_dir, "shared_feature_num.json")
    runtime_json = output_paths["runtime_json"]
    model_txt = output_paths["model"]
    integrated_h5ad = output_paths["integrated_h5ad"]
    r_script_path = os.path.join(bindsc_input_dir, "run_bindsc.R")
    temp_dir = os.path.join(base_dir, args.bindsc_temp_dirname)

    # save preprocessed inputs for R
    rna_clean = _clean_for_zellkonverter(prepared_inputs["adata1"])
    mod2_clean = _clean_for_zellkonverter(prepared_inputs["adata2"])

    rna_clean.write(rna_processed_h5ad)
    mod2_clean.write(mod2_processed_h5ad)

    with open(shared_json, "w") as f:
        json.dump({"shared_feature_num": prepared_inputs["shared_feature_num"]}, f, indent=2)

    _write_bindsc_r_script(r_script_path)

    cmd = [
        args.rscript_executable,
        r_script_path,
        rna_processed_h5ad,
        mod2_processed_h5ad,
        str(prepared_inputs["shared_feature_num"]),
        integrated_h5ad,
        runtime_json,
        model_txt,
        temp_dir,
        str(args.bindsc_alpha),
        str(args.bindsc_lambda),
        str(args.bindsc_K),
        str(args.bindsc_num_iteration),
        str(args.bindsc_tolerance),
        str(args.bindsc_block_size),
        modality_b_name,
    ]

    print("[INFO] Running bindSC via R...")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    if not os.path.exists(integrated_h5ad):
        raise FileNotFoundError(f"bindSC integrated output not found: {integrated_h5ad}")

    adata_integrated = sc.read_h5ad(integrated_h5ad)

    if "X_multi" not in adata_integrated.obsm:
        adata_integrated.obsm["X_multi"] = np.asarray(adata_integrated.X)

    # overwrite obs with original Python metadata
    adata_integrated = _restore_final_obs_from_python(
        adata_integrated=adata_integrated,
        adata_rna_raw=adata_rna_raw,
        adata_mod2_raw=adata_mod2_raw,
        modality_a_name=modality_a_name,
        modality_b_name=modality_b_name,
    )

    embedding_df = pd.DataFrame(
        adata_integrated.obsm["X_multi"],
        index=adata_integrated.obs_names,
        columns=[f"latent_{i+1}" for i in range(adata_integrated.obsm["X_multi"].shape[1])],
    )

    if "modality" in adata_integrated.obs.columns:
        embedding_df.insert(
            0,
            "modality",
            adata_integrated.obs["modality"].astype(str).values
        )

    if os.path.exists(runtime_json):
        with open(runtime_json, "r") as f:
            runtime_info = json.load(f)
    else:
        runtime_info = {}

    run_stats = {
        "total_runtime_min": runtime_info.get("total_runtime_min", np.nan),
        "peak_memory_use": runtime_info.get("peak_memory_use", np.nan),
        "shared_feature_num": int(prepared_inputs["shared_feature_num"]),
        "model_ckpt_path": model_txt,
        "rna_processed_h5ad": rna_processed_h5ad,
        "mod2_processed_h5ad": mod2_processed_h5ad,
        "integrated_h5ad_path": integrated_h5ad,
    }

    return None, adata_integrated, embedding_df, run_stats