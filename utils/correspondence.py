import os
import numpy as np
import pandas as pd
import anndata as ad


def load_valid_pairs_for_dataset(
    adata_rna: ad.AnnData,
    adata_mod2: ad.AnnData,
    correspondence_path: str,
) -> pd.DataFrame:
    corr = pd.read_csv(correspondence_path)
    corr = corr.iloc[:, :2].copy()
    corr.columns = ["Protein name", "RNA name"]

    valid_rows = []
    rna_set = set(adata_rna.var_names)
    mod2_set = set(adata_mod2.var_names)

    for _, row in corr.iterrows():
        mod2_feat = row["Protein name"]
        genes = row["RNA name"]

        if mod2_feat not in mod2_set:
            continue
        if "Ignore" in str(genes):
            continue

        for g in str(genes).split("/"):
            if g in rna_set:
                valid_rows.append({"RNA name": g, "Protein name": mod2_feat})

    return pd.DataFrame(valid_rows).drop_duplicates().reset_index(drop=True)


def write_nested_subset_tables(
    valid_pairs_df: pd.DataFrame,
    outdir: str,
    dataset_name: str,
    percentages=(1.0, 0.8, 0.6, 0.4, 0.2),
    seed: int = 42,
):
    """
    Create nested subsets:
      1.0 -> 0.8 -> 0.6 -> 0.4 -> 0.2
    where each smaller set is a subset of the previous one.
    """
    os.makedirs(outdir, exist_ok=True)

    percentages = list(percentages)
    rng = np.random.default_rng(seed)

    current_df = valid_pairs_df.copy().reset_index(drop=True)
    current_pct = 1.0

    for pct in percentages:
        if pct == 1.0:
            subset_df = valid_pairs_df.copy().reset_index(drop=True)
        else:
            frac = pct / current_pct
            n_keep = int(round(len(current_df) * frac))
            n_keep = max(1, min(n_keep, len(current_df)))
            keep_idx = rng.choice(current_df.index.to_numpy(), size=n_keep, replace=False)
            subset_df = current_df.loc[np.sort(keep_idx)].copy().reset_index(drop=True)

        pct_label = f"{pct:.1f}"
        out_csv = os.path.join(outdir, f"{dataset_name}_corr_subset_{pct_label}.csv")
        subset_df[["Protein name", "RNA name"]].to_csv(out_csv, index=False)

        current_df = subset_df
        current_pct = pct


def subset_mod2_by_correspondence(
    adata_rna: ad.AnnData,
    adata_mod2: ad.AnnData,
    correspondence_path: str,
    modality2_name: str = "ADT",
) -> ad.AnnData:
    valid_pairs = load_valid_pairs_for_dataset(adata_rna, adata_mod2, correspondence_path)
    features = pd.unique(valid_pairs["Protein name"]).tolist()
    features = [p for p in features if p in adata_mod2.var_names]

    if len(features) == 0:
        raise ValueError(
            f"No valid {modality2_name} features found for correspondence table: {correspondence_path}"
        )

    return adata_mod2[:, features].copy()