DATA_ROOT = "/path/to/your/downloaded/datasets"

DATASET_CONFIGS = {
    "bmcite": {
        "RNA": f"{DATA_ROOT}/bmcite/RNA.h5ad",
        "ADT": f"{DATA_ROOT}/bmcite/Prot.h5ad",
        "correspondence": "resources/protein_gene_conversion.csv",
        "modalities": ["RNA", "ADT"],
    },
}