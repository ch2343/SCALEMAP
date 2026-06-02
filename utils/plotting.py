import os
import numpy as np
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt


def ensure_embedding_in_obsm(adata: ad.AnnData, embed_key: str = "X_multi"):
    """
    Make sure embedding exists in adata.obsm.
    """
    if embed_key not in adata.obsm:
        adata.obsm[embed_key] = np.asarray(adata.X)


def compute_umap_scanpy(
    adata: ad.AnnData,
    use_rep: str,
    n_neighbors: int = 30,
    random_state: int = 1234,
):
    """
    Compute neighbors + UMAP using scanpy.
    Only change: min_dist = 0.3
    """
    adata = adata.copy()

    sc.pp.neighbors(
        adata,
        use_rep=use_rep,
        n_neighbors=n_neighbors
    )

    sc.tl.umap(
        adata,
        random_state=random_state,
        min_dist=0.3
    )

    return adata


def extract_celltype_colors(raw_adata: ad.AnnData, celltype_key: str = "celltype"):
    """
    Extract stable celltype color mapping from a raw AnnData object.
    """
    color_key = f"{celltype_key}_colors"
    mapping = None

    if color_key in raw_adata.uns and str(raw_adata.obs[celltype_key].dtype) == "category":
        cats = list(raw_adata.obs[celltype_key].cat.categories)
        cols = list(raw_adata.uns[color_key])
        mapping = dict(zip(cats, cols))

    return mapping


def apply_celltype_colors(
    adata: ad.AnnData,
    color_mapping: dict,
    celltype_key: str = "celltype",
):
    """
    Apply an existing color mapping to another AnnData object.
    """
    if color_mapping is None:
        return

    if celltype_key not in adata.obs.columns:
        return

    adata.obs[celltype_key] = adata.obs[celltype_key].astype("category")
    cats = list(adata.obs[celltype_key].cat.categories)

    adata.uns[f"{celltype_key}_colors"] = [
        color_mapping.get(cat, "#808080") for cat in cats
    ]


def save_umap_plot(
    adata: ad.AnnData,
    color: str,
    title: str,
    save_path: str,
):
    """
    Save one UMAP figure.
    """
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


def generate_raw_modality_umaps(
    adata_dict: dict,
    outdir: str,
    celltype_key: str = "celltype",
):
    """
    Generate raw UMAPs for all modalities.
    adata_dict example:
        {"RNA": adata_rna, "ADT": adata_adt}
    """
    os.makedirs(outdir, exist_ok=True)

    for modality, adata in adata_dict.items():

        adata_plot = adata.copy()

        sc.pp.pca(adata_plot)

        sc.pp.neighbors(
            adata_plot,
            use_rep="X_pca",
            n_neighbors=30
        )

        sc.tl.umap(
            adata_plot,
            random_state=1234,
            min_dist=0.3
        )

        save_umap_plot(
            adata_plot,
            color=celltype_key,
            title=f"{modality} raw UMAP",
            save_path=os.path.join(outdir, f"UMAP_raw_{modality}.png"),
        )


def generate_integrated_umaps(
    adata_integrated: ad.AnnData,
    outdir: str,
    embed_key: str = "X_multi",
    batch_key: str = "modality",
    celltype_key: str = "celltype",
    color_mapping: dict = None,
    prefix: str = "integrated",
):
    """
    Generate batch-view and celltype-view UMAPs from integrated embedding.
    """
    os.makedirs(outdir, exist_ok=True)

    ensure_embedding_in_obsm(adata_integrated, embed_key=embed_key)

    adata_plot = compute_umap_scanpy(
        adata_integrated,
        use_rep=embed_key
    )

    if color_mapping is not None:
        apply_celltype_colors(
            adata_plot,
            color_mapping,
            celltype_key=celltype_key
        )

    save_umap_plot(
        adata_plot,
        color=batch_key,
        title=f"{prefix} batch view",
        save_path=os.path.join(outdir, f"UMAP_{prefix}_batch.png"),
    )

    save_umap_plot(
        adata_plot,
        color=celltype_key,
        title=f"{prefix} celltype view",
        save_path=os.path.join(outdir, f"UMAP_{prefix}_celltype.png"),
    )