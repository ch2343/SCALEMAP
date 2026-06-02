# SCALEMAP

SCALEMAP is a deep learning framework for paired single-cell multi-omics integration, including both **RNA–ADT** and **RNA–ATAC** datasets. Rather than relying on explicit cell-cell matching, SCALEMAP learns a shared latent representation through cross-modality translation and shared-feature alignment, enabling robust integration across modalities while preserving biologically meaningful structure.

---

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

Alternatively:

```bash
pip install -r requirements.txt
```

---

## Input Data

SCALEMAP expects each modality to be stored as an AnnData (`.h5ad`) object.

### RNA–ADT

```text
RNA: cells × genes
ADT: cells × proteins
```

### RNA–ATAC

```text
RNA: cells × genes
ATAC: cells × gene activity features
```

The two modalities should correspond to the same cells and ideally share the same cell IDs.

---

## Dataset Configuration

Create your local dataset configuration file:

```bash
cp configs/datasets.example.py configs/datasets.py
```

Then specify your dataset paths.

Example RNA–ADT dataset:

```python
DATASET_CONFIGS = {
    "my_dataset": {
        "RNA": "/path/to/RNA.h5ad",
        "ADT": "/path/to/ADT.h5ad",
        "correspondence": "resources/protein_gene_conversion.csv",
        "modalities": ["RNA", "ADT"],
    }
}
```

Example RNA–ATAC dataset:

```python
DATASET_CONFIGS = {
    "my_dataset_ATAC": {
        "RNA": "/path/to/RNA.h5ad",
        "ATAC": "/path/to/ATAC.h5ad",
        "modalities": ["RNA", "ATAC"],
    }
}
```

---

## Data Processing

### RNA–ADT

For RNA–ADT integration, SCALEMAP:

1. Loads RNA and ADT AnnData objects.
2. Uses a protein–gene correspondence table.
3. Identifies valid RNA–protein shared feature pairs.
4. Normalizes and log-transforms both modalities.
5. Places shared features at the beginning of the feature matrices.
6. Performs clustering to provide structural information during training.

### RNA–ATAC

For RNA–ATAC integration, SCALEMAP:

1. Selects highly variable features.
2. Normalizes and log-transforms RNA and ATAC gene activity matrices.
3. Identifies shared features using gene-name intersection.
4. Reorders shared features to the front.
5. Performs clustering before training.

---

## Quick Start (Command Line)

### RNA–ADT Integration

```bash
python script_1_integration.py \
    --dataset D22 \
    --modalities RNA_ADT \
    --method scalemap \
    --version test_v1 \
    --output_root ./results
```

### RNA–ATAC Integration

```bash
python script_1_integration.py \
    --dataset D22_ATAC \
    --modalities RNA_ATAC \
    --method scalemap \
    --version test_atac_v1 \
    --output_root ./results \
    --preprocess_mode rna_atac
```

---

## Python API Example

SCALEMAP can also be run directly inside a Python session.

### RNA–ADT Example

```python
import scanpy as sc

from utils.preprocess_scalemap import build_scalemap_inputs
from methods.scalemap_core import run_scalemap

adata_rna = sc.read_h5ad("RNA.h5ad")
adata_adt = sc.read_h5ad("ADT.h5ad")

adata1, adata2, preprocess_info = build_scalemap_inputs(
    adata_rna=adata_rna,
    adata_adt=adata_adt,
    correspondence_path="resources/protein_gene_conversion.csv",
    dataset_name="example_dataset"
)

model, adata_integrated, embedding_df, run_stats = run_scalemap(
    adata1=adata1,
    adata2=adata2,
    adata_rna_raw=adata_rna,
    adata_adt_raw=adata_adt,
    shared_feature_num=preprocess_info["shared_feature_num"],
    seed=42
)

adata_integrated.write("integrated_scalemap.h5ad")
embedding_df.to_csv("embedding_scalemap.csv")
```

### RNA–ATAC Example

```python
import scanpy as sc

from utils.preprocess_scalemap_atac import build_scalemap_rna_atac_inputs
from methods.scalemap_core import run_scalemap

adata_rna = sc.read_h5ad("RNA.h5ad")
adata_atac = sc.read_h5ad("ATAC.h5ad")

adata1, adata2, preprocess_info = build_scalemap_rna_atac_inputs(
    adata_rna=adata_rna,
    adata_atac=adata_atac,
    dataset_name="example_rna_atac"
)

model, adata_integrated, embedding_df, run_stats = run_scalemap(
    adata1=adata1,
    adata2=adata2,
    adata_rna_raw=adata_rna,
    adata_adt_raw=adata_atac,
    shared_feature_num=preprocess_info["shared_feature_num"],
    seed=42
)

adata_integrated.write("integrated_scalemap_rna_atac.h5ad")
embedding_df.to_csv("embedding_scalemap_rna_atac.csv")
```

---

## Important Parameters

Most users only need to modify the following parameters.

| Parameter | Description |
|------------|------------|
| `--dataset` | Dataset name defined in `configs/datasets.py` |
| `--modalities` | `RNA_ADT` or `RNA_ATAC` |
| `--method` | Integration method (`scalemap`) |
| `--version` | Name of the current run |
| `--output_root` | Output directory |
| `--seed` | Random seed |
| `--preprocess_mode` | Use `rna_atac` for RNA–ATAC integration |

Example:

```bash
python script_1_integration.py \
    --dataset D22 \
    --modalities RNA_ADT \
    --method scalemap \
    --version benchmark_v1 \
    --output_root ./results \
    --seed 42
```

---

## Evaluation

Evaluate integrated embeddings:

### RNA–ADT

```bash
python script_2_evaluation.py \
    --dataset D22 \
    --modalities RNA_ADT \
    --method scalemap \
    --version benchmark_v1 \
    --output_root ./results \
    --batch_key modality \
    --celltype_key celltype \
    --pair_modalities RNA ADT
```

### RNA–ATAC

```bash
python script_2_evaluation.py \
    --dataset D22_ATAC \
    --modalities RNA_ATAC \
    --method scalemap \
    --version benchmark_v1 \
    --output_root ./results \
    --batch_key modality \
    --celltype_key celltype \
    --pair_modalities RNA ATAC
```

Evaluation includes:

- ASW Label
- ASW Batch
- ARI
- NMI
- kBET
- Label Transfer Accuracy
- FOSCTTM
- Runtime
- Memory Usage

---

## UMAP Visualization

Generate integrated UMAPs:

```bash
python script_3_umap.py \
    --dataset D22 \
    --modalities RNA_ADT \
    --method scalemap \
    --version benchmark_v1 \
    --output_root ./results \
    --skip_raw
```

Figures will be stored under:

```text
results/<dataset>/<method>/<version>/umap/
```

---

## Benchmarking Additional Methods

The repository also supports benchmarking several existing integration methods:

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
    --version scmodal_test \
    --output_root ./results
```

---

## Repository Structure

```text
SCALEMAP/
├── configs/
├── methods/
├── utils/
├── resources/
│   ├── protein_gene_conversion.csv
│   └── rna_protein_COMBAT.csv
├── script_1_integration.py
├── script_2_evaluation.py
├── script_3_umap.py
├── environment.yml
├── requirements.txt
└── README.md
```

---

## Notes

- Dataset files are not included in this repository.
- Users should provide their own `.h5ad` datasets.
- RNA–ATAC integration assumes ATAC has been converted into a gene activity matrix.
- Public repositories should not contain institution-specific file paths.

---

## Citation

If you use SCALEMAP in your work, please cite the corresponding manuscript.
