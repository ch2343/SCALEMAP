import os
import time
import tracemalloc
import random
from itertools import combinations

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import umap
from annoy import AnnoyIndex
from scipy import sparse as sp
from scipy.spatial.distance import cdist

from typing import Any, Dict, Optional, Tuple


from annoy import AnnoyIndex
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from itertools import combinations



#from metrices_quick import *

def get_correspondence_indices(correspondence_file, adata_ADT, adata_RNA):
    # Load correspondence file and preprocess
    correspondence = correspondence_file
    correspondence['Protein name'] = correspondence['Protein name'].replace(
        to_replace={'CD11a-CD18': 'CD11a/CD18', 'CD66a-c-e': 'CD66a/c/e'}
    )

    # Find valid RNA-Protein pairs
    rna_protein_correspondence = []
    for i in range(correspondence.shape[0]):
        curr_protein_name, curr_rna_names = correspondence.iloc[i]
        if curr_protein_name not in adata_ADT.var_names:
            continue
        if 'Ignore' in curr_rna_names:  # Skip ignored mappings
            continue
        curr_rna_names = curr_rna_names.split('/')  # Split RNA names
        for r in curr_rna_names:
            if r in adata_RNA.var_names:
                rna_protein_correspondence.append([r, curr_protein_name])

    # Convert correspondence to array
    rna_protein_correspondence = np.array(rna_protein_correspondence)

    # Extract indices for RNA and Protein
    rna_indices = [adata_RNA.var_names.get_loc(rna) for rna, _ in rna_protein_correspondence]
    protein_indices = [adata_ADT.var_names.get_loc(protein) for _, protein in rna_protein_correspondence]

    return rna_indices, protein_indices

def multi_resolution_cluster(adata, resolution1=0.5, resolution2=7, method="Leiden"):
    import scanpy as sc

    # Ensure adata is a full object
    adata = adata.copy()
    # Perform PCA
    sc.tl.pca(adata)

    # Compute neighbors using the PCA representation
    sc.pp.neighbors(adata, use_rep="X_pca", metric="euclidean")

    # Perform clustering
    if method.lower() == "leiden":
        sc.tl.leiden(adata, resolution=resolution1, key_added="leiden1")
        sc.tl.leiden(adata, resolution=resolution2, key_added="leiden2")
    elif method.lower() == "louvain":
        sc.tl.louvain(adata, resolution=resolution1, key_added="louvain1")
        sc.tl.louvain(adata, resolution=resolution2, key_added="louvain2")
    else:
        raise ValueError("Method should be 'Louvain' or 'Leiden'")

    # Validate clustering results
    if method.lower() == "leiden":
        if 'leiden1' not in adata.obs.columns or 'leiden2' not in adata.obs.columns:
            raise KeyError("Clustering results 'leiden1' or 'leiden2' are missing in adata.obs.")
    elif method.lower() == "louvain":
        if 'louvain1' not in adata.obs.columns or 'louvain2' not in adata.obs.columns:
            raise KeyError("Clustering results 'louvain1' or 'louvain2' are missing in adata.obs.")

    return adata
    
def count_labels_per_batch(labels, batch_ids):
    unique_batches = batch_ids.unique()
    label_counts_per_batch = {batch: (labels[batch_ids == batch].unique(), 
                                      torch.stack([(labels[batch_ids == batch] == l).sum() for l in labels[batch_ids == batch].unique()]))
                              for batch in unique_batches}
    return label_counts_per_batch
    


def separate_rare_common_cells(labels, threshold=0.05):
    """
    Separate cells into rare and common based on the frequency of their labels.
    
    Args:
        labels: Cluster labels from unsupervised clustering.
        threshold: Percentage threshold to define rare cell types.
    
    Returns:
        rare_mask: Boolean mask indicating rare cells.
        common_mask: Boolean mask indicating common cells.
    """
    unique_labels, counts = torch.unique(labels, return_counts=True)
    total_cells = len(labels)
    rare_labels = unique_labels[counts / total_cells < threshold]
    
    rare_mask = torch.isin(labels, rare_labels)
    common_mask = ~rare_mask
    
    return rare_mask, common_mask

     
def annotate_by_nn(vec_tar, vec_ref, label_ref, k=20, metric='cosine'):
    dist_mtx = cdist(vec_tar, vec_ref, metric=metric)
    idx = dist_mtx.argsort()[:, :k]
    labels = [max(list(label_ref[i]), key=list(label_ref[i]).count) for i in idx]
    return labels

def compute_umap(adata, rep=None):
    import umap

    reducer = umap.UMAP(n_neighbors=30,
                        n_components=2,
                        metric="correlation",
                        n_epochs=None,
                        learning_rate=1.0,
                        min_dist=0.3,
                        spread=1.0,
                        set_op_mix_ratio=1.0,
                        local_connectivity=1,
                        repulsion_strength=1,
                        negative_sample_rate=5,
                        a=None,
                        b=None,
                        random_state=1234,
                        metric_kwds=None,
                        angular_rp_forest=False,
                        verbose=True)
    if rep is None:
        X_umap = reducer.fit_transform(adata.X)
    else:
        X_umap = reducer.fit_transform(adata.obsm[rep])

    adata.obsm['X_umap'] = X_umap

def to_numpy_dense(X, dtype=np.float32):
    """Convert sparse / matrix-like / tensor input to a dense numpy array."""
    if sp.issparse(X):
        X = X.toarray()
    elif isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    return np.asarray(X, dtype=dtype)



import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

        
import torch.nn as nn

class SharedEncoder(nn.Module):
    def __init__(self, rna_dim, protein_dim, latent_dim):
        super(SharedEncoder, self).__init__()
        
        # Modality-specific input layers
        self.rna_input = nn.Sequential(
            nn.Linear(rna_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256)
        )
        self.protein_input = nn.Sequential(
            nn.Linear(protein_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256)
        )
        
        # Shared batch normalization layer
        self.shared_batch_norm = nn.BatchNorm1d(latent_dim)

        # Shared layers for embedding space
        self.shared_layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, latent_dim)
        )
    
    def forward(self, rna=None, protein=None, rna_indices=None, protein_indices=None):
        """
        Process RNA and protein inputs. If both are provided, compute combined embeddings,
        normalize, and then separate them back to their original numbers.
        """
        rna_embed = None
        protein_embed = None

        # Process RNA if provided
        if rna is not None:
            rna_input_embed = self.rna_input(rna)

        # Process protein if provided
        if protein is not None:
            protein_input_embed = self.protein_input(protein)

        # If both modalities are provided
        if rna is not None and protein is not None:
            # Combine embeddings across batches
            combined_input = torch.cat((rna_input_embed, protein_input_embed), dim=0)
            # Pass through shared layers
            #combined_input = self.shared_batch_norm1(combined_input)
            normalized_embed = self.shared_layers(combined_input)
            
            # Apply shared batch normalization
            normalized_embed = self.shared_batch_norm(normalized_embed)

            # Separate embeddings back to RNA and protein
            rna_embed = normalized_embed[:rna_input_embed.size(0)]
            protein_embed = normalized_embed[rna_input_embed.size(0):]

        # If only RNA is provided
        elif rna is not None:
            #normalized_embed = self.shared_batch_norm(rna_input_embed)
            rna_embed = self.shared_layers(rna_input_embed)

        # If only protein is provided
        elif protein is not None:
            #normalized_embed = self.shared_batch_norm(protein_input_embed)
            protein_embed = self.shared_layers(protein_input_embed)

        return rna_embed, protein_embed

        

class Decoder(nn.Module):
    def __init__(self, p_dim, latent_dim=256):
        super(Decoder, self).__init__()
        self.relu = nn.ReLU()
        
        # Main decoder pathway
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, p_dim),
        )

    def forward(self, z):
        # Main decoding
        decoded = self.decoder(z)
        
        return decoded

class Decoder_combined(nn.Module):
    def __init__(self, p_dim_RNA, p_dim_ADT, latent_dim):
        super(Decoder_combined, self).__init__()
        self.decoder_RNA = Decoder(p_dim_RNA, latent_dim)
        self.decoder_ADT = Decoder(p_dim_ADT, latent_dim)

    def forward(self, z1, z2):
        x1 = self.decoder_RNA(z1)
        x2 = self.decoder_ADT(z2)
        return x1, x2

        
class CrossEntropy(nn.Module):
    def __init__(self, reduction='mean'):
        super(CrossEntropy, self).__init__()
        self.reduction = reduction

    def forward(self, output, target):
        # Apply log softmax to the output
        log_preds = F.log_softmax(output, dim=-1)
        
        # Compute the negative log likelihood loss
        loss = F.nll_loss(log_preds, target, reduction=self.reduction)
        
        return loss
        
class discriminator(nn.Module):
    def __init__(self, n_input, domain_number):
        super(discriminator, self).__init__()
        n_hidden = 128

        # Define layers
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, domain_number)
        self.loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, x, batch_ids, generator=False):
        # Forward pass through layers
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        output = self.fc3(h)
        #output = torch.clamp(output, min=-50.0, max=50.0)
        softmax_probs = F.softmax(output, dim=1)
        
        # Loss computation
        D_loss = self.loss(output, batch_ids)
        #if self.loss.reduction == 'mean':
            #D_loss = D_loss.mean()
        #elif self.loss.reduction == 'sum':
            #D_loss = D_loss.sum()

        return D_loss

def combine_latent_vectors(z1, z2):
    """
    Combines latent vectors z1 and z2, and creates an indicator vector.

    Args:
        z1 (torch.Tensor): Latent vector from the first source (e.g., ADT).
        z2 (torch.Tensor): Latent vector from the second source (e.g., RNA).

    Returns:
        combined_z (torch.Tensor): Combined latent vectors from z1 and z2.
        source_indicator (torch.Tensor): A vector indicating the source of each entry (0 for z1, 1 for z2).
    """
    # Combine z1 and z2 along the batch dimension (dim=0)
    combined_z = torch.cat((z1, z2), dim=0)
    
    # Create an indicator vector
    source_indicator = torch.cat((
        torch.zeros(z1.size(0), dtype=torch.int64),  # 0 for z1
        torch.ones(z2.size(0), dtype=torch.int64)    # 1 for z2
    ), dim=0)
    
    return combined_z, source_indicator

def combine_latent_vectors_three(z0, z1, z2):
    """
    Combines three latent vectors (z0, z1, z2) and creates a source indicator vector.
    
    Args:
        z0 (torch.Tensor): Latent vector from modality 0.
        z1 (torch.Tensor): Latent vector from modality 1.
        z2 (torch.Tensor): Latent vector from modality 2.
    
    Returns:
        combined_z (torch.Tensor): Concatenated latent vectors.
        source_indicator (torch.Tensor): Indicator for each sample 
                                         (0 for z0, 1 for z1, 2 for z2).
    """
    # Concatenate the latent vectors along dim=0
    combined_z = torch.cat((z0, z1, z2), dim=0)
    
    # Create the source indicator vector
    source_indicator = torch.cat([
        torch.full((z0.size(0),), 0, dtype=torch.int64),
        torch.full((z1.size(0),), 1, dtype=torch.int64),
        torch.full((z2.size(0),), 2, dtype=torch.int64)
    ], dim=0)

    return combined_z, source_indicator


def calculate_weighted_cosine_loss(x_A, x_AtoB, shared_gene_num, v1_true):
    """
    Calculate a weighted cosine similarity loss where weights are inversely proportional to cluster sizes.

    Args:
        x_A (torch.Tensor): Original data matrix for modality A.
        x_AtoB (torch.Tensor): Reconstructed data matrix for modality A mapped to modality B.
        shared_gene_num (int): Number of shared genes/features to consider.
        v1_true (torch.Tensor): Cluster labels for each sample in x_A.

    Returns:
        torch.Tensor: Weighted cosine similarity loss.
    """
    # Initialize total loss and weight normalization
    loss_cosine = 0
    total_weight = 0

    # Get unique clusters and their sizes
    unique_clusters, cluster_sizes = torch.unique(v1_true, return_counts=True)

    # Loop through each unique cluster
    for cluster, cluster_size in zip(unique_clusters, cluster_sizes):
        # Mask for the current cluster
        mask_A = v1_true == cluster

        # Ensure there are samples in the current cluster
        if mask_A.sum() > 0:
            # Select the aligned features for original and reconstructed data
            x_A_cluster = x_A[mask_A, :shared_gene_num]
            x_AtoB_cluster = x_AtoB[mask_A, :shared_gene_num]

            # Compute cosine similarity for all samples in the cluster
            cluster_cosine_loss = (1 - F.cosine_similarity(x_A_cluster, x_AtoB_cluster)).mean()

            # Apply weight inversely proportional to cluster size
            weight = 1.0
            loss_cosine += weight * cluster_cosine_loss
            total_weight += weight

    # Normalize loss by total weight
    if total_weight > 0:
        loss_cosine /= total_weight

    return loss_cosine

import torch
import torch.nn.functional as F

def contrastive_loss(x_A, x_AtoB, temperature=0.1):
    """
    Compute the contrastive loss for RNA-protein pairs.
    
    Args:
        x_A: RNA expression (batch_size x shared_feature_dim)
        x_AtoB: Protein expression (batch_size x shared_feature_dim)
        temperature: Temperature parameter for scaling the similarity scores.
    
    Returns:
        Contrastive loss.
    """
    # Normalize the inputs
    x_A_normalized = F.normalize(x_A, p=2, dim=1)
    x_AtoB_normalized = F.normalize(x_AtoB, p=2, dim=1)
    
    # Compute cosine similarity between positive pairs
    sim_pos = torch.sum(x_A_normalized * x_AtoB_normalized, dim=1)  # (batch_size,)
    
    # Compute cosine similarity between x_A and all x_AtoB (including negatives)
    sim_matrix = torch.matmul(x_A_normalized, x_AtoB_normalized.T)  # (batch_size, batch_size)
    
    # Scale by temperature
    sim_matrix = sim_matrix / temperature
    
    # Compute the contrastive loss
    numerator = torch.exp(sim_pos / temperature)  # exp(sim_pos / tau)
    denominator = torch.sum(torch.exp(sim_matrix), dim=1)  # sum over all x_AtoB'
    loss = -torch.log(numerator / denominator).mean()  # average over the batch
    
    return loss

def rank_and_weight_loss(loss, top_k_ratio=0.05):
    """
    Rank the discriminator loss and assign weights based on the ranking.
    
    Args:
        loss: Tensor of discriminator losses for each cell in the mini-batch.
        top_k_ratio: Proportion of cells to assign higher weights (e.g., top 20%).
    
    Returns:
        weights: Tensor of weights for each cell.
    """
    # Rank the losses in descending order
    ranked_indices = torch.argsort(loss, descending=True)
    
    # Create a weight tensor
    weights = torch.zeros_like(loss)
    
    # Assign higher weights to the top-k cells
    top_k = int(top_k_ratio * len(loss))
    weights[ranked_indices[:top_k]] = 0.1  # Higher weight for top-k cells
    weights[ranked_indices[top_k:]] = 1  # Lower weight for the rest
    
    return weights
    
import torch

def rank_and_weight_loss_2(loss, top_k_ratio=0.05, mid_k_ratio=0.20):
    """
    Reverse weighting: downweight hard cells, emphasize easy ones.
    """

    ranked_indices = torch.argsort(loss, descending=True)
    n = len(loss)

    top_k = max(1, int(top_k_ratio * n))
    mid_k = max(top_k + 1, int(mid_k_ratio * n))

    weights = torch.full_like(loss, 1.5)  # easy cells (majority)

    # medium-hard cells
    weights[ranked_indices[top_k:mid_k]] = 0.3

    # hardest cells
    weights[ranked_indices[:top_k]] = 0.05

    # normalize
    weights = weights / weights.mean()

    return weights

def _add_noise(x, sigma=0.1, dropout_p=0.0):
    """Gaussian noise + (optional) feature dropout."""
    if sigma > 0:
        x = x + sigma * torch.randn_like(x)
    if dropout_p > 0:
        mask = torch.bernoulli(torch.full_like(x, 1 - dropout_p))
        x = x * mask
    return x

import torch

@torch.no_grad()
def rna_depth_augment(
    x: torch.Tensor,
    depth_range=(0.5, 0.9),       # keep 50–90% reads per cell
    gene_dropout_p: float = 0.0,  # optional, small (e.g., 0.02)
    mode: str = "auto",           # "auto" | "binomial" | "poisson"
):
    """
    Sequencing-depth augmentation for count-like matrices.

    x: [N, G] dense tensor or sparse_coo (values assumed >= 0)
    Returns a tensor with same shape/storage and dtype=float32.
    """
    assert 0.0 < depth_range[0] <= depth_range[1] <= 1.0
    is_sparse = x.is_sparse if isinstance(x, torch.Tensor) else False
    device = x.device

    # per-cell scales s in [low, high], shape [N,1]
    N = x.shape[0]
    s = torch.empty(N, 1, device=device).uniform_(depth_range[0], depth_range[1])

    def looks_integer(t: torch.Tensor) -> bool:
        # Check if values are near integers (only on a small sample for speed)
        if t.numel() == 0: return True
        sample = t.reshape(-1)
        if sample.numel() > 10000:
            idx = torch.randint(sample.numel(), (10000,), device=sample.device)
            sample = sample.index_select(0, idx)
        frac = (sample - sample.round()).abs().max()
        return float(frac) < 1e-3

    if is_sparse:
        x = x.coalesce()
        idx = x.indices()          # [2, nnz]
        val = x.values()           # [nnz]
        rows = idx[0]              # row (cell) ids for each nnz entry
        p = s[rows, 0]             # [nnz]
        if mode == "binomial" or (mode == "auto" and looks_integer(val)):
            # Binomial thinning: X' ~ Binomial(X, p)
            # torch.binomial expects counts>=0 and 0<=p<=1
            out_val = torch.binomial(val.to(torch.float32), p.clamp(0,1))
        else:
            # Poisson thinning: X' ~ Poisson(p * X)
            out_val = torch.poisson((p * val.to(torch.float32)).clamp_min(0))
        if gene_dropout_p > 0:
            keep = torch.bernoulli(torch.full_like(out_val, 1.0 - gene_dropout_p))
            out_val = out_val * keep
        # remove zeros
        nz_mask = out_val > 0
        new_idx = idx[:, nz_mask]
        new_val = out_val[nz_mask].to(torch.float32)
        return torch.sparse_coo_tensor(new_idx, new_val, x.size(), device=device).coalesce()
    else:
        # dense
        if mode == "binomial" or (mode == "auto" and looks_integer(x)):
            out = torch.binomial(x.to(torch.float32), s.clamp(0,1))
        else:
            out = torch.poisson((s * x.to(torch.float32)).clamp_min(0))
        if gene_dropout_p > 0:
            keep = torch.bernoulli(torch.full_like(out, 1.0 - gene_dropout_p))
            out = out * keep
        return out

# Convenience: two independent “views” with independent depth scales
@torch.no_grad()
def make_depth_views(x, depth_range=(0.5, 0.9), gene_dropout_p=0.0, mode="auto"):
    return (
        rna_depth_augment(x, depth_range, gene_dropout_p, mode),
        rna_depth_augment(x, depth_range, gene_dropout_p, mode),
    )


def _info_nce_excluding_same_labels(z_q, z_k, labels, temperature=0.1, eps=1e-8):
    """
    InfoNCE with positives = (i,i) across views and negatives = all j != i
    EXCEPT those with the same label as i (excluded from denominator).
    z_q, z_k: [N, D] (already L2-normalized recommended)
    labels:  [N] int64 (cluster ids aligned with the batch order)
    """
    N, D = z_q.shape
    # cosine similarity matrix (q vs k)
    sim = torch.matmul(F.normalize(z_q, dim=1), F.normalize(z_k, dim=1).T) / temperature  # [N, N]

    # positive logits are the diagonal
    pos = torch.diag(sim)  # [N]

    # Build a mask of allowed negatives: j != i and label[j] != label[i]
    labels = labels.view(-1, 1)  # [N,1]
    same_label = (labels == labels.T)  # [N,N]
    not_self = ~torch.eye(N, dtype=torch.bool, device=z_q.device)
    neg_mask = not_self & (~same_label)

    # For numerical stability: set masked entries to large negative
    neg_logits = sim.masked_fill(~neg_mask, float('-inf'))  # [N,N]

    # log-sum-exp over valid negatives; if a row has no valid negatives, fall back to all j!=i
    # detect rows with no valid negatives
    rows_no_neg = (neg_mask.sum(dim=1) == 0)  # [N]
    if rows_no_neg.any():
        fallback_mask = not_self  # allow all except self
        neg_logits = torch.where(rows_no_neg.view(-1,1), sim.masked_fill(~fallback_mask, float('-inf')), neg_logits)

    # InfoNCE loss: -log( exp(pos) / (exp(pos) + sum exp(negs)) )
    denom = torch.logsumexp(torch.stack([pos, torch.logsumexp(neg_logits, dim=1)], dim=1), dim=1)  # [N]
    loss = -(pos - denom)
    return loss.mean()

import torch
import numpy as np

class FeatureSubsetModel:
    def __init__(self, shared_gene_num):
        """
        Simple model that just returns the first shared_gene_num features
        Args:
            shared_gene_num: Number of features to keep from input
        """
        self.shared_gene_num = shared_gene_num
    
    def predict_model(self, x):
        """
        Returns first shared_gene_num features of input x
        Args:
            x: Input array/tensor of shape (batch_size, features) or (features,)
               Features can be any length >= shared_gene_num
        Returns:
            Subsetted array/tensor with only first shared_gene_num features
        """
        # Convert numpy array to tensor if needed
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
            
        # Handle single sample vs batch
        if x.dim() == 1:
            return x[:self.shared_gene_num]
        else:
            return x[:, :self.shared_gene_num]
    
import os
import time
import numpy as np
import scanpy as sc
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Model(object):
    def __init__(self, batch_size=250, training_steps=2000, seed=10, n_latent=50,
                 lambdaAE = 20.0, lambdaMNN = 1, lambdaGAN = 2, lambdaNoise = 0.2, lr1 = 0.001, lr2 = 0.002,
                 cluster_label_key="leiden1",
                 model_path="models", data_path="data", result_path="results"):

        # add device
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

        self.batch_size = batch_size
        self.training_steps = training_steps
        self.n_latent = n_latent
        self.lambdaAE = lambdaAE
        self.lambdaMNN = lambdaMNN
        self.lambdaGAN = lambdaGAN
        self.lambdaNoise = lambdaNoise
        self.model_path = model_path
        self.data_path = data_path
        self.result_path = result_path
        self.cluster_label_key = cluster_label_key
        self.lr1  = lr1
        self.lr2 = lr2


    def preprocess(self, 
                   adata_A_input, 
                   adata_B_input, 
                   predict_model
                   #gene_list,
                   #protein_list
                   ):
        self.adata_A = adata_A_input.copy()
        self.adata_B = adata_B_input.copy()

        self.predict_model = predict_model.predict_model
        self.emb_A = to_numpy_dense(self.adata_A.X)
        self.emb_B = to_numpy_dense(self.adata_B.X)
        self.obs_A = self.adata_A.obs
        self.obs_B = self.adata_B.obs

    def preprocess_additional_inputs(self, 
                   adata_A_input, 
                   adata_B_input, 
                   shared_gene_num,
                   layer_adata_A_MNN=None, 
                   layer_adata_B_MNN=None, 
                   ):
        # For ATAC-seq data, an option is to let adata_X_input be LSI matrices, 
        # layer_adata_X_MNN be the layer name storing gene activity matrices
        # The first K=shared_gene_num features in self.feat_A_MNN and self.feat_B_MNN should be positively related .

        assert ((layer_adata_A_MNN is not None) or (layer_adata_B_MNN is not None)), "One of the layer names should be feeded; otherwise, use .preprocess() function."
        adata_A = adata_A_input.copy()
        adata_B = adata_B_input.copy()

        self.shared_gene_num = shared_gene_num
        self.emb_A = adata_A.X
        self.emb_B = adata_B.X
        self.obs_A = adata_A.obs
        self.obs_B = adata_B.obs
        if layer_adata_A_MNN is None:
            self.feat_A_MNN = self.emb_A
        else:
            self.feat_A_MNN = adata_A.obsm[layer_adata_A_MNN]
        if layer_adata_B_MNN is None:
            self.feat_B_MNN = self.emb_B
        else:
            self.feat_B_MNN = adata_B.obsm[layer_adata_B_MNN]


    def train(self, save_path: str = None):
        begin_time = time.time()
        print("Begining time: ", time.asctime(time.localtime(begin_time)))
        self.E = SharedEncoder(self.emb_A.shape[1], self.emb_B.shape[1], self.n_latent).to(self.device)
        self.G = Decoder_combined(self.emb_A.shape[1], self.emb_B.shape[1], self.n_latent).to(self.device)
        self.D_Z = discriminator(self.n_latent, 2).to(self.device)
        params_G = list(self.E.parameters()) + list(self.G.parameters())
        optimizer_G = optim.Adam(params_G, lr=self.lr1, weight_decay=0.001)
        optimizer_D = optim.Adam(list(self.D_Z.parameters()), lr=self.lr2, weight_decay=0.001)
        self.E.train()
        self.G.train()
        self.D_Z.train()
        noise_sigma=0.1
        dropout_p=0.1
        temperature=0.1

        N_A = self.emb_A.shape[0]
        N_B = self.emb_B.shape[0]

        for step in range(self.training_steps):
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            index_A = np.random.choice(np.arange(N_A), size=self.batch_size)
            index_B = np.random.choice(np.arange(N_B), size=self.batch_size)
            x_A = torch.from_numpy(self.emb_A[index_A, :]).float().to(self.device)
            x_B = torch.from_numpy(self.emb_B[index_B, :]).float().to(self.device)
            l1_full = self.obs_A[self.cluster_label_key].cat.codes.values
            l2_full = self.obs_B[self.cluster_label_key].cat.codes.values
            l1 = torch.as_tensor(l1_full[index_A], dtype=torch.int64, device=self.device)
            l2 = torch.as_tensor(l2_full[index_B], dtype=torch.int64, device=self.device)

            # ----- make two noisy views for each modality -----
            xA_q, xA_k = make_depth_views(x_A, depth_range=(0.6, 0.95), gene_dropout_p=0.0)  # RNA
            xB_q, xB_k = make_depth_views(x_B, depth_range=(0.6, 0.95), gene_dropout_p=0.0)  # ATAC/ADT counts too


            zA_q, zB_q = self.E(xA_q, xB_q)   # use the A branch output
            zA_k, zB_k = self.E(xA_k, xB_k)

            loss_A = _info_nce_excluding_same_labels(zA_q, zA_k, l1, temperature=temperature)
            loss_B = _info_nce_excluding_same_labels(zB_q, zB_k, l2, temperature=temperature)

            loss_noise = (loss_A + loss_B) * 0.5
            z_A, z_B = self.E(x_A, x_B)
            z_A = z_A.to(self.device)
            z_B = z_B.to(self.device)
            x_BtoA, x_AtoB = self.G(z_B, z_A)
            x_Arecon, x_Brecon = self.G(z_A, z_B)
            z_Arecon, z_Brecon = self.E(x_Arecon, x_Brecon)
            z_BtoA, z_AtoB = self.E(x_BtoA, x_AtoB)
            A_pred_B = self.predict_model(x_A)
            x_BtoA_pred_B = self.predict_model(x_BtoA)
            loss_cos1 = (1 - torch.sum(F.normalize(A_pred_B, p=2) * F.normalize(self.predict_model(x_AtoB), p=2), 1)).mean()
            loss_cos2 = (1 - torch.sum(F.normalize(self.predict_model(x_B), p=2) * F.normalize(x_BtoA_pred_B, p=2), 1)).mean()
            def normalize(tensor):
                return (tensor - tensor.mean()) / (tensor.std() + 1e-8)  # Adding small epsilon to avoid division by zero

            loss_mse1 = torch.mean((normalize(A_pred_B) - normalize(self.predict_model(x_AtoB)))**2)
            loss_mse2 = torch.mean((normalize(self.predict_model(x_B)) - normalize(x_BtoA_pred_B))**2)
            loss_contrastive_A = contrastive_loss(A_pred_B, self.predict_model(x_AtoB))
            loss_contrastive_B = contrastive_loss(self.predict_model(x_B), x_BtoA_pred_B)
            loss_MNN = 0 * (loss_mse1 + loss_mse2) + 0.2 * (loss_contrastive_A + loss_contrastive_B) + 0 * (loss_cos1 + loss_cos2)
            combined_z, source_indicator = combine_latent_vectors(z_A, z_B)

            # discriminator loss:
            for _ in range(7):
                optimizer_D.zero_grad()
                # Compute discriminator loss for all cells
                loss_D_all = self.D_Z(combined_z, source_indicator)
    
                # Rank and weight the losses
                weights_D = rank_and_weight_loss(loss_D_all)
    
                # Apply weights to the discriminator loss
                loss_D = (weights_D * loss_D_all).mean()
                #loss_D = loss_D_all.mean()
                loss_D.backward(retain_graph=True)
                optimizer_D.step()

            # autoencoder loss:
            loss_AE_A = torch.mean((x_Arecon - x_A)**2)
            loss_AE_B = torch.mean((x_Brecon - x_B)**2)
            loss_AE = loss_AE_A + loss_AE_B

            loss_RE_A = torch.mean((z_BtoA - z_B)**2)
            loss_RE_B = torch.mean((z_AtoB - z_A)**2)
            loss_RE = loss_RE_A + loss_RE_B


            # generator loss
            loss_G_GAN_all = -self.D_Z(combined_z, source_indicator, generator=False)
            weights_G = rank_and_weight_loss(loss_G_GAN_all)  # Use the same weights as discriminator
            loss_G_GAN = (weights_G * loss_G_GAN_all).mean()
            #loss_G_GAN = loss_G_GAN_all.mean()
            

            optimizer_G.zero_grad()
            loss_G = self.lambdaGAN * loss_G_GAN + self.lambdaAE * loss_AE + self.lambdaAE * loss_RE + self.lambdaMNN * loss_MNN + self.lambdaNoise * loss_noise
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(params_G, 5.0)
            optimizer_G.step()

            if not step % 50:
                print("step %d, loss_GAN=%f, loss_AE=%f, loss_MNN=%f"
                 % (step, loss_G_GAN, self.lambdaAE*loss_AE, self.lambdaMNN*loss_MNN))

        end_time = time.time()
        print("Ending time: ", time.asctime(time.localtime(end_time)))
        self.train_time = end_time - begin_time
        print("Training takes %.2f seconds" % self.train_time)

        if save_path is None:
            save_path = os.path.join(self.model_path, "ckpt.pth")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        state = {
            "E": self.E.state_dict(),
            "G": self.G.state_dict()
        }
        torch.save(state, save_path)


    def eval(self, load_path: str = None):
        begin_time = time.time()
        print("Begining time: ", time.asctime(time.localtime(begin_time)))

        self.E = SharedEncoder(self.emb_A.shape[1], self.emb_B.shape[1], self.n_latent).to(self.device)
        self.G = Decoder_combined(self.emb_A.shape[1], self.emb_B.shape[1], self.n_latent).to(self.device)
        if load_path is None:
            load_path = os.path.join(self.model_path, "ckpt.pth")

        ckpt = torch.load(load_path, map_location=self.device)
        self.E.load_state_dict(ckpt["E"])
        self.G.load_state_dict(ckpt["G"])

        x_A = torch.from_numpy(self.emb_A).float().to(self.device)
        x_B = torch.from_numpy(self.emb_B).float().to(self.device)

        self.E.eval()
        self.G.eval()

        with torch.no_grad():
            z_A, z_B = self.E(x_A, x_B)
            x_BtoA, x_AtoB = self.G(z_B, z_A)

        end_time = time.time()
        
        print("Ending time: ", time.asctime(time.localtime(end_time)))
        self.eval_time = end_time - begin_time
        print("Evaluating takes %.2f seconds" % self.eval_time)

        self.latent = np.concatenate((z_A.detach().cpu().numpy(), z_B.detach().cpu().numpy()), axis=0)
        self.data_Aspace = np.concatenate((self.emb_A, x_BtoA.detach().cpu().numpy()), axis=0)
        self.data_Bspace = np.concatenate((x_AtoB.detach().cpu().numpy(), self.emb_B), axis=0)

    def get_imputed_df(self, 
                       scale = 'scaled' # if scale=='log', then restore expression after log1p
                       ):

        x_BtoA = self.data_Aspace[self.emb_A.shape[0]:]
        x_AtoB = self.data_Bspace[:self.emb_A.shape[0]]
        if scale == 'log':
            x_BtoA = x_BtoA * self.adata_A.var['std'].values.reshape(1, -1) + self.adata_A.var['mean'].values.reshape(1, -1)
            x_AtoB = x_AtoB * self.adata_B.var['std'].values.reshape(1, -1) + self.adata_B.var['mean'].values.reshape(1, -1)
        imputed_df_BtoA = pd.DataFrame(x_BtoA, index=self.adata_B.obs.index, columns=self.adata_A.var.feature_name)
        imputed_df_BtoA = imputed_df_BtoA.groupby(imputed_df_BtoA.columns, axis=1).mean()
        imputed_df_AtoB = pd.DataFrame(x_AtoB, index=self.adata_A.obs.index, columns=self.adata_B.var.feature_name)
        imputed_df_AtoB = imputed_df_AtoB.groupby(imputed_df_AtoB.columns, axis=1).mean()
        self.imputed_df_BtoA = imputed_df_BtoA
        self.imputed_df_AtoB = imputed_df_AtoB

        

# =========================================================
# High-level wrappers for Script I / II / III
# =========================================================

def build_integrated_adata(
    model: Model,
    adata_rna_raw: ad.AnnData,
    adata_adt_raw: ad.AnnData,
    modality_a_name: str = "RNA",
    modality_b_name: str = "ADT",
    embedding_key: str = "X_multi",
) -> ad.AnnData:
    """
    Build integrated AnnData from trained SCALEMAP model.
    """
    obs_rna = adata_rna_raw.obs.copy()
    obs_adt = adata_adt_raw.obs.copy()

    obs_rna = obs_rna.copy()
    obs_adt = obs_adt.copy()
    obs_rna["modality"] = modality_a_name
    obs_adt["modality"] = modality_b_name

    integrated_obs = pd.concat([obs_rna, obs_adt], axis=0)

    adata_integrated = ad.AnnData(X=model.latent, obs=integrated_obs)
    adata_integrated.obsm[embedding_key] = model.latent.copy()

    return adata_integrated


def build_embedding_df(
    adata_integrated: ad.AnnData,
    embedding_key: str = "X_multi",
) -> pd.DataFrame:
    """
    Convert integrated embedding to csv-friendly DataFrame.
    Row names = cell IDs.
    Includes modality column.
    """
    emb = adata_integrated.obsm[embedding_key]
    df = pd.DataFrame(
        emb,
        index=adata_integrated.obs_names,
        columns=[f"latent_{i+1}" for i in range(emb.shape[1])],
    )

    if "modality" in adata_integrated.obs.columns:
        df.insert(0, "modality", adata_integrated.obs["modality"].astype(str).values)

    return df


def run_scalemap(
    adata1: ad.AnnData,
    adata2: ad.AnnData,
    adata_rna_raw: ad.AnnData,
    adata_adt_raw: ad.AnnData,
    model_save_path: str,
    *,
    shared_feature_num: int,
    training_steps: int = 2000,
    batch_size: int = 250,
    seed: int = 42,
    n_latent: int = 50,
    lambdaAE: float = 20.0,
    lambdaMNN: float = 1.0,
    lambdaGAN: float = 2.0,
    lambdaNoise: float = 0.2,
    lr1: float = 0.001,
    lr2: float = 0.002,
    compute_umap_flag: bool = False,
    embedding_key: str = "X_multi",
    cluster_label_key: str = "leiden1",
    modality_a_name: str = "RNA",
    modality_b_name: str = "ADT",
) -> Tuple[Model, ad.AnnData, pd.DataFrame, Dict[str, Any]]:
    """
    End-to-end wrapper for SCALEMAP integration.

    Returns
    -------
    model
    adata_integrated
    embedding_df
    run_stats
    """
    predictor = FeatureSubsetModel(shared_gene_num=shared_feature_num)

    model = Model(
        batch_size=batch_size,
        training_steps=training_steps,
        seed=seed,
        n_latent=n_latent,
        lambdaAE=lambdaAE,
        lambdaMNN=lambdaMNN,
        lambdaGAN=lambdaGAN,
        lambdaNoise = lambdaNoise,
        lr1 = lr1,
        lr2 = lr2,
        cluster_label_key=cluster_label_key
    )

    model.preprocess(adata1, adata2, predict_model=predictor)

    t0 = time.time()
    tracemalloc.start()
    model.train(save_path=model_save_path)
    model.eval(load_path=model_save_path)
    total_runtime = (time.time() - t0) / 60
    peak_memory = tracemalloc.get_traced_memory()[1]/ 1024**3
    tracemalloc.stop()

    adata_integrated = build_integrated_adata(
        model=model,
        adata_rna_raw=adata_rna_raw,
        adata_adt_raw=adata_adt_raw,
        modality_a_name=modality_a_name,
        modality_b_name=modality_b_name,
        embedding_key=embedding_key,
    )

    if compute_umap_flag:
        compute_umap(adata_integrated, rep=embedding_key)

    embedding_df = build_embedding_df(adata_integrated, embedding_key=embedding_key)

    run_stats = {
        "train_time_sec": model.train_time,
        "eval_time_sec": model.eval_time,
        "total_runtime_min": total_runtime,
        "peak_memory_use": peak_memory,
        "n_cells_modality_a": int(adata1.shape[0]),
        "n_cells_modality_b": int(adata2.shape[0]),
        "n_features_modality_a": int(adata1.shape[1]),
        "n_features_modality_b": int(adata2.shape[1]),
        "latent_dim": int(n_latent),
        "seed": int(seed),
        "shared_feature_num": int(shared_feature_num),
        "model_ckpt_path": model_save_path,
    }

    return model, adata_integrated, embedding_df, run_stats
    
    
MODEL_FILE_EXT = "pt"


def add_method_args(parser):
    parser.add_argument("--hvg_top_genes", type=int, default=2000)
    parser.add_argument("--cluster_resolution", type=float, default=0.5)
    parser.add_argument("--cluster_method", type=str, default="leiden")
    parser.add_argument("--cluster_label_key", type=str, default=None)
    parser.add_argument("--final_scale_max_value", type=float, default=10.0)

    parser.add_argument("--training_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=250)
    parser.add_argument("--n_latent", type=int, default=50)
    parser.add_argument("--lambdaAE", type=float, default=20.0)
    parser.add_argument("--lambdaMNN", type=float, default=1.0)
    parser.add_argument("--lambdaGAN", type=float, default=2.0)
    parser.add_argument("--lambdaNoise", type=float, default=0.2)
    parser.add_argument("--lr1", type=float, default=0.001)
    parser.add_argument("--lr2", type=float, default=0.002)



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
            raise ValueError(f"Unsupported modality pair: {modality_a_name}_{modality_b_name}")
    else:
        preprocess_mode = args.preprocess_mode

    if preprocess_mode == "rna_adt":
        from utils.preprocess_scalemap import build_scalemap_inputs

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
        from utils.preprocess_scalemap_atac import build_scalemap_inputs_atac

        adata1, adata2, preprocess_info = build_scalemap_inputs_atac(
            adata_rna=adata_rna,
            adata_atac=adata_mod2,
            correspondence_path=correspondence_path,
            dataset_name=dataset_name,
            hvg_top_genes=args.hvg_top_genes,
            cluster_resolution=args.cluster_resolution,
            cluster_method=args.cluster_method,
            final_scale_max_value=args.final_scale_max_value,
        )

    else:
        raise ValueError(f"Unsupported preprocess_mode: {preprocess_mode}")

    prepared_inputs = {
        "adata1": adata1,
        "adata2": adata2,
        "shared_feature_num": preprocess_info["shared_feature_num"],
    }
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
    if args.cluster_label_key is not None:
        cluster_label_key = args.cluster_label_key
    else:
        method_lower = args.cluster_method.lower()
        if method_lower == "leiden":
            cluster_label_key = "leiden1"
        elif method_lower == "louvain":
            cluster_label_key = "louvain1"
        else:
            raise ValueError(f"Unsupported cluster_method: {args.cluster_method}")

    model, adata_integrated, embedding_df, run_stats = run_scalemap(
        adata1=prepared_inputs["adata1"],
        adata2=prepared_inputs["adata2"],
        adata_rna_raw=adata_rna_raw,
        adata_adt_raw=adata_mod2_raw,
        model_save_path=output_paths["model"],
        shared_feature_num=prepared_inputs["shared_feature_num"],
        training_steps=args.training_steps,
        batch_size=args.batch_size,
        seed=args.seed,
        n_latent=args.n_latent,
        lambdaAE=args.lambdaAE,
        lambdaMNN=args.lambdaMNN,
        lambdaGAN=args.lambdaGAN,
        lambdaNoise=args.lambdaNoise,
        lr1 = args.lr1,
        lr2 = args.lr2,
        compute_umap_flag=False,
        embedding_key="X_multi",
        modality_a_name=modality_a_name,
        modality_b_name=modality_b_name,
        cluster_label_key=cluster_label_key,
    )
    return model, adata_integrated, embedding_df, run_stats