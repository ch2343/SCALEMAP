# SCALEMAP

SCALEMAP is a multi-omics integration framework for paired single-cell datasets, including RNA–protein (RNA–ADT) and RNA–ATAC integration. SCALEMAP learns a shared latent representation by combining cross-modality reconstruction, shared-feature alignment, adversarial modality alignment, and structure-preserving regularization.

## Installation

Clone the repository:

```bash
git clone https://github.com/ch2343/SCALEMAP.git
cd SCALEMAP
```

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate scalemap
```

Alternatively, install dependencies with:

```bash
pip install -r requirements.txt
```

Some benchmark methods may require additional method-specific dependencies. For example, `bindSC` requires R packages, and `scGLUE` or `MaxFuse` may require separate installation steps.

## Input Data

SCALEMAP expects each modality to be stored as an AnnData `.h5ad` object.

For RNA–ADT integration:

```text
RNA: cells × genes
ADT: cells × proteins
```

For RNA–ATAC integration:

```text
RNA: cells × genes
ATAC: cells × gene activity features
```

The two modalities should contain matched cells. Ideally, the two AnnData objects should have the same cell order or matching cell IDs in `obs_names`.

## Dataset Configuration

Before running SCALEMAP, create your local dataset configuration file:

```bash
cp configs/datasets.example.py configs/datasets.py
```

Then edit `configs/datasets.py` to specify your own dataset paths.

Example RNA–ADT dataset:

```python
DATASET_CONFIGS = {
    "D22": {
        "RNA": "/path/to/D22/adata_rna.h5ad",
        "ADT": "/path/to/D22/adata_adt.h5ad",
        "correspondence": "resources/protein_gene_conversion.csv",
        "modalities": ["RNA", "ADT"],
    },
}
```

Example RNA–ATAC dataset:

```python
DATASET_CONFIGS = {
    "D22_ATAC": {
        "RNA": "/path/to/D22/adata_rna.h5ad",
        "ATAC": "/path/to/D22/adata_atac.h5ad",
        "modalities": ["RNA", "ATAC"],
    },
}
```

Large datasets are not included in this repository. Users should prepare their own `.h5ad` files and update `configs/datasets.py`.

## Feature Correspondence

For RNA–ADT integration, SCALEMAP uses a correspondence table between proteins and RNA genes.

Included resource files:

```text
resources/protein_gene_conversion.csv
resources/rna_protein_COMBAT.csv
```

For RNA–ATAC integration, ATAC is assumed to be represented as a gene activity matrix. Shared features are identified by matching RNA gene names with ATAC gene activity feature names.

## Data Processing

For RNA–ADT integration, SCALEMAP performs the following preprocessing:

```text
1. Load RNA and ADT AnnData objects.
2. Load the RNA–protein correspondence table.
3. Identify valid RNA–protein shared feature pairs.
4. Split features into shared and unshared blocks.
5. Normalize, log-transform, and scale the data.
6. Select highly variable RNA features when needed.
7. Concatenate shared features first, followed by unshared features.
8. Cluster each modality to provide structure information for training.
```

For RNA–ATAC integration, SCALEMAP uses a gene-activity-based pipeline:

```text
1. Load RNA and ATAC gene activity AnnData objects.
2. Select highly variable features for RNA and ATAC.
3. Normalize, log-transform, and scale each modality.
4. Identify shared features by gene-name intersection.
5. Reorder features so shared features appear first.
6. Cluster each modality before training.
```

The number of shared features is stored during preprocessing and used by SCALEMAP to define feature-level alignment losses.

## Quick Start: RNA–ADT Integration

Run SCALEMAP on an RNA–ADT dataset:

```bash
python script_1_integration.py \
  --dataset D22 \
  --modalities RNA_ADT \
  --method scalemap \
  --version test_v1 \
  --output_root ./results
```

## Quick Start: RNA–ATAC Integration

Run SCALEMAP on an RNA–ATAC dataset:

```bash
python script_1_integration.py \
  --dataset D22_ATAC \
  --modalities RNA_ATAC \
  --method scalemap \
  --version test_atac_v1 \
  --output_root ./results \
  --preprocess_mode rna_atac
```

## Important SCALEMAP Parameters

The most commonly used parameters are:

```text
--dataset
```

Dataset name defined in `configs/datasets.py`.

```text
--modalities
```

Modality pair. Use `RNA_ADT` for RNA–protein integration and `RNA_ATAC` for RNA–ATAC integration.

```text
--method
```

Integration method. Use `--method scalemap` to run SCALEMAP.

```text
--version
```

A user-defined run name. Output files are saved under:

```text
results/<dataset>/<method>/<version>/
```

```text
--output_root
```

Root directory for output files. Default is `./results`.

```text
--seed
```

Random seed for reproducibility.

```text
--hvg_top_genes
```

Number of highly variable genes or features selected during preprocessing. Default is commonly `2000`.

```text
--preprocess_mode
```

Preprocessing mode. For RNA–ATAC integration, use:

```bash
--preprocess_mode rna_atac
```

```text
--training_steps
```

Number of training iterations. Larger values may improve convergence but increase runtime.

```text
--batch_size
```

Mini-batch size for training.

```text
--n_latent
```

Dimension of the shared latent embedding.

```text
--lambdaAE
```

Weight for reconstruction losses. This controls how strongly SCALEMAP preserves modality-specific feature information.

```text
--lambdaMNN
```

Weight for shared-feature alignment. This controls how strongly SCALEMAP aligns the shared feature block across modalities.

```text
--lambdaGAN
```

Weight for adversarial modality alignment in the latent space.

```text
--lambdaNoise
```

Weight for noise-based consistency regularization.

```text
--cluster_method
```

Clustering method used during preprocessing. Common choices are `leiden` and `louvain`.

```text
--cluster_resolution
```

Resolution parameter for preprocessing clustering.

## Example With Custom Parameters

```bash
python script_1_integration.py \
  --dataset D22 \
  --modalities RNA_ADT \
  --method scalemap \
  --version scalemap_custom_v1 \
  --output_root ./results \
  --seed 42 \
  --training_steps 4000 \
  --batch_size 250 \
  --lambdaAE 10 \
  --lambdaNoise 0.1 \
  --lambdaGAN 2 \
  --n_latent 50
```

## Output Files

Each integration run creates:

```text
results/<dataset>/<method>/<version>/
├── model_<method>_<dataset>_<modalities>_<version>.pt
├── embedding_<method>_<dataset>_<modalities>_<version>.csv
├── integrated_<method>_<dataset>_<modalities>_<version>.h5ad
├── runtime_<method>_<dataset>_<modalities>_<version>.json
└── preprocess_<method>_<dataset>_<modalities>_<version>.json
```

The integrated embedding is stored in:

```python
adata_integrated.obsm["X_multi"]
```

## Evaluation

Evaluate an RNA–ADT integration result:

```bash
python script_2_evaluation.py \
  --dataset D22 \
  --modalities RNA_ADT \
  --method scalemap \
  --version test_v1 \
  --output_root ./results \
  --batch_key modality \
  --celltype_key celltype \
  --pair_modalities RNA ADT \
  --force_shared_obs_names
```

Evaluate an RNA–ATAC integration result:

```bash
python script_2_evaluation.py \
  --dataset D22_ATAC \
  --modalities RNA_ATAC \
  --method scalemap \
  --version test_atac_v1 \
  --output_root ./results \
  --batch_key modality \
  --celltype_key celltype \
  --pair_modalities RNA ATAC \
  --force_shared_obs_names
```

Evaluation metrics include biological conservation, modality mixing, label transfer accuracy, pair distance, FOSCTTM, runtime, and memory usage.

## UMAP Visualization

Draw integrated UMAPs:

```bash
python script_3_umap.py \
  --dataset D22 \
  --modalities RNA_ADT \
  --method scalemap \
  --version test_v1 \
  --output_root ./results \
  --skip_raw
```

UMAP figures are saved under:

```text
results/<dataset>/<method>/<version>/umap/
```

## Benchmarking Other Methods

Although this repository focuses on SCALEMAP, it also includes a benchmarking interface for several existing methods.

Supported method names:

```text
scalemap
scmodal
scglue
maxfuse
bindsc
```

Example:

```bash
python script_1_integration.py \
  --dataset D22 \
  --modalities RNA_ADT \
  --method scmodal \
  --version scmodal_test_v1 \
  --output_root ./results
```

For scGLUE on RNA–ATAC data:

```bash
python script_1_integration.py \
  --dataset D22_ATAC \
  --modalities RNA_ATAC \
  --method scglue \
  --version scglue_atac_test_v1 \
  --output_root ./results \
  --preprocess_mode rna_atac \
  --n_pca_rna 100 \
  --n_pca_mod2 100 \
  --rna_prob_model NB \
  --mod2_prob_model Normal
```

## Combining Benchmark Metrics

Combine metrics across datasets for one method:

```bash
python combine_method_metrics.py \
  scalemap \
  test_v1 \
  ./results
```

For multi-seed experiments:

```bash
python combine_method_metrics_across_seeds.py \
  scalemap \
  benchmark_v1 \
  ./results \
  --seeds 10 20 42
```

This produces mean, standard deviation, and long-format metric summaries.

## Custom Datasets

To add a new RNA–ADT dataset:

```python
DATASET_CONFIGS["my_dataset"] = {
    "RNA": "/path/to/RNA.h5ad",
    "ADT": "/path/to/ADT.h5ad",
    "correspondence": "resources/protein_gene_conversion.csv",
    "modalities": ["RNA", "ADT"],
}
```

To add a new RNA–ATAC dataset:

```python
DATASET_CONFIGS["my_dataset_ATAC"] = {
    "RNA": "/path/to/RNA.h5ad",
    "ATAC": "/path/to/ATAC.h5ad",
    "modalities": ["RNA", "ATAC"],
}
```

## Notes

- `configs/datasets.py` is user-specific and should not contain private paths in public repositories.
- Large files such as `.h5ad`, model checkpoints, logs, and result files are excluded from Git tracking.
- RNA–ATAC integration assumes that ATAC has already been converted to a gene activity matrix.
- For new RNA–ADT datasets, users should provide a valid RNA–protein correspondence table.

## Citation

If you use SCALEMAP or this repository, please cite the corresponding method paper or repository.

## License

Please add a license file before public release if needed.
