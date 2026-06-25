# Sparse Autoencoders (SAE) for Interpretable Text Analysis

## Overview

This project implements a two-pipeline architecture for interpreting neural network representations through Sparse Autoencoders. The system decomposes model activations into interpretable features at different levels of granularity:

- Uses the local `SAELens` submodule under `external/sae-lens` for SAE loading and BatchTopK support.
- GemmaScope-inspired SAEs are loaded from pretrained weights, but the upstream `GemmaScope` repository is not included as a git submodule.
- The `interp_embed` reference is documented for design comparison, but no local `interp_embed` repo is checked in.

- **Pipeline 1**: Token-level SAE on Gemma-3 hidden states → max-pooled document-level features
- **Pipeline 2**: Phrase-level SAE on F2LLM embeddings → max-pooled document-level features

Both pipelines support optional frozen-core extensions (ExtendedSAE) for domain-specific feature discovery.

---

## Architecture

### Pipeline 1: Gemma-3 Token-to-Document Encoding

**Flow**: Raw text → Gemma-3 hidden states (layer 17) → pretrained SAE encode → max-pool → document vectors

**Key files**: `saev5.py` (`run_llm_max_pool_pipeline`), `sae_shared.py`

**Implementation details**:
- **Token-level SAE**: Uses `gemma-scope-2-4b-it-res/layer_17_width_16k_l0_medium` (d=16384, k=medium sparsity)
- **Max-pooling strategy**: Per-token SAE activations are max-pooled across tokens to produce a single d=16384-dimensional vector per document
- **FrozenCoreResidualSAE / ExtendedSAE**: 
  - When `USE_FROZEN_CORE=1`, a residual SAE is trained on the residual $e = x - \text{decode}(\text{core\_sae}(x))$
  - This captures domain-specific (French text) features not explained by the pretrained SAE
  - Results in d=16384+1024 dimensional vectors when enabled
  - Shared by reference: `ExtendedSAE` inherits from `FrozenCoreResidualSAE`, uses core decode + extra encoder

**Dimensionality handling**:
- Core features: d_core=16384 (from pretrained SAE)
- Extra features: D_EXTRA=1024 (trained on French residuals)
- Total when frozen: d_total=17408

**Metric evaluation**:
- `compute_metrics()` evaluates reconstruction FVE/NMSE/L0 on document-level SAE vectors
- When input is a SAE code vector (shape=[B, d_sae]), uses decode→encode→compare in latent space
- Detects SAE vectors by checking if tensor dimension matches `model.cfg.d_sae`

---

### Pipeline 2: F2LLM Phrase-to-Document Encoding

**Flow**: Text → split into phrases → F2LLM-v2 embeddings (d=320) → phrase-level SAE encode → max-pool by doc → document vectors

**Key files**: `saev5.py` (`run_f2llm_pipeline`), `sae_shared.py`

**Implementation details**:
- **Phrase splitting**: Documents split into up to `MAX_PHRASES_DOC=10` sentences using NLTK
- **Embedding model**: F2LLM-v2-80M/160M/330M (multilingual, locally cached)
- **Phrase SAE**: Custom SAE trained on phrase embeddings (d_in=320 → d_sae=8192, k=16)
  - Trained on 85% of train phrase embeddings, evaluated on 15%
  - Improves over raw embedding for downstream classification
- **Max-pooling strategy**: Same as Pipeline 1 but at phrase-level (max over phrases per document)

**Phrase-to-document mapping**:
- `phrase_to_doc` array maps each phrase index to its document index (0-indexed)
- Used in `encode_documents_with_phrase_sae()` to aggregate phrase activations
- Essential for maintaining correspondence between activation vectors and document labels

**Document-level embedding aggregation**:
- Phrase embeddings are max-pooled at document level for downstream tasks
- Helper function `pool_embeddings_by_document()` handles this aggregation
- Ensures shape consistency with document-level SAE activations for classification

---

## Script Descriptions

### `saev5.py` - Main Pipeline Orchestration

#### Entry Point
```python
if __name__ == "__main__":
    # Loads corpora (FineWeb-2 + EDF emails)
    # Runs Pipeline 1 and 2 sequentially
    # Saves results to SAVE_DIR
```

#### Key Functions

**`load_pretrained_sae()`**
- Downloads/caches Gemma Scope SAE from Hugging Face Hub
- Returns `JumpReLUSAE` model (d_sae=16384)
- Used as core in FrozenCoreResidualSAE

**`FrozenCoreResidualSAE` (class)**
- Inherits from `nn.Module`
- Wraps a pretrained SAE as frozen core
- Encodes extra features via learnable W_enc_extra / b_enc_extra
- Decode: uses core SAE decoder only
- Supports batch topk encoding for sparsity

**`ExtendedSAE` (class)**
- Extends FrozenCoreResidualSAE with PCA initialization
- Initializes extra encoder from SVD of residuals
- Improves convergence on domain-specific features

**`run_llm_max_pool_pipeline()`**
- Orchestrates Pipeline 1
- Extracts Gemma-3 hidden states
- Encodes via pretrained + optional frozen-core SAE
- Max-pools to document level
- Runs 4 analysis tasks (diffing, NPMI, clustering, retrieval)
- Evaluates metrics (FVE, L0, silhouette, downstream classification)

**`run_f2llm_pipeline()`**
- Orchestrates Pipeline 2
- Trains phrase-level SAE if not cached
- Encodes test/email documents
- Generates LLM feature labels via local Gemma-3
- Runs UMAP visualization
- Downstream classification on document-level aggregations

---

### `sae_shared.py` - Shared Utilities

#### SAE Training & Loading

**`load_or_train_sae()`**
- Wrapper for training or loading cached SAE
- Handles both sparse and dense architectures
- Evaluates on held-out split

**`load_or_train()`**
- Generic checkpoint save/load for models
- Used for ExtendedSAE training

#### Analysis Functions

**`compute_metrics()`**
- Evaluates SAE reconstruction quality (FVE, NMSE, L0)
- Key fix: Detects SAE code vectors by checking `model.cfg.d_sae`
  - If input dimension matches d_sae and model has encode/decode, treats as latent vector
  - Reconstructs via decode→encode comparison in SAE latent space
  - Otherwise falls back to model(x) forward pass
- Converts activations to model's native parameter dtype (not forced bfloat16)

**`compute_rho_sae()`**
- Estimates activation sparsity correlation (ρ metric)
- Samples activations and computes cross-correlation

**`compute_npmi()`**
- Computes Normalized Pointwise Mutual Information matrix
- Measures feature-word statistical associations

**`downstream_classification()`**
- 5-fold cross-validation on linear probe (Logistic Regression)
- Compares SAE activations vs. raw embeddings
- **Requires**: acts_by_label and (optionally) raw_emb_by_label with matching sample counts
- **Key constraint**: Both must have document-level aggregations for consistency

**`pool_embeddings_by_document()`** (newly added)
- Max-pools phrase-level embeddings by document index
- Takes phrase_embeddings [n_phrases, d] and phrase_to_doc [n_phrases]
- Returns document-level embeddings [n_docs, d]
- Essential for Pipeline 2 downstream tasks

#### Visualization

**`analyze_with_umap()`**
- Computes UMAP on SAE activations
- Generates interactive HTML with feature highlights
- Shows top activating tokens per feature

**`targeted_clustering_by_axis()`**
- Semantic clustering along a query axis (e.g., "énergie électrique")
- Uses LLM-generated soft labels

**`property_based_retrieval()`**
- Boltzmann-weighted retrieval by feature properties
- Returns ranked documents by relevance

**`local_gemma_judge()`**
- Uses local Gemma-3 to generate interpretable labels for top features
- Reduces hallucination vs. remote calls

---

## Data Flow & Dimensionality

### Pipeline 1 Example
```
Text (variable length)
  ↓
Gemma-3 layer 17: [seq_len, 3072]
  ↓
SAE.encode(): [seq_len, 16384]  (sparse)
  ↓
max(dim=0): [16384]  (document-level)
  ↓
Optional ExtendedSAE: [17408] if frozen-core enabled
```

### Pipeline 2 Example
```
Text → phrases (≤10 per doc)
  ↓
F2LLM embedding: [n_phrases_doc, 320]
  ↓
SAE.encode(): [n_phrases_doc, 8192]  (sparse)
  ↓
max(dim=0): [8192]  (document-level)
  ↓
pool_embeddings_by_document(): [1, 320]  (for downstream)
```

---

## Key Design Choices

### 1. **Max-pooling Strategy**
- **Why**: Preserves sparsity; single dominant feature per document is interpretable
- **Trade-off**: Loses relative activation magnitudes; asymmetric over aggregation

### 2. **Frozen-Core SAE**
- **Why**: Decouples model-generic (pretrained) from domain-specific features
- **Implementation**: Residual encoder only; decoder frozen to pretrained
- **Use case**: Discover French-specific phenomena without retraining core

### 3. **Document-Level Aggregation**
- **Why**: Matches label granularity (energy/sports per document, not per token/phrase)
- **Consistency requirement**: All downstream tasks must use document-level representations
- **Safety check**: `downstream_classification()` validates acts_by_label and raw_emb_by_label have equal sizes per label

### 4. **SAE Code Vector Detection in `compute_metrics()`**
- **Why**: Avoid treating max-pooled activations as raw model inputs
- **Mechanism**: Check if tensor dim matches `model.cfg.d_sae` and model has `encode`/`decode`
- **Fallback**: Use model(x) forward if dimension mismatch

### 5. **Phrase-Level Embedding Pooling**
- **Why**: Maintain consistency with SAE document-level activations in downstream tasks
- **Implementation**: `pool_embeddings_by_document()` max-pools phrase embeddings by document index
- **Safety**: Ensures both acts_by_label and raw_emb_by_label have [n_docs, d] shape

---

## Configuration

Environment variables (set in `run_sae.slurm` or manually):

```bash
# Corpus
SEED=42
N_TRAIN=3600           # training documents
N_TEST=400             # test documents

# Pipeline 1 (Gemma-3 → SAE)
LAYER=17               # which Gemma-3 layer
USE_FROZEN_CORE=1      # enable ExtendedSAE
D_EXTRA=1024           # extra feature dimensions
K_EXTRA=32             # extra sparsity target
EPOCHS_EXTRA=20        # ExtendedSAE training epochs

# Pipeline 2 (F2LLM → SAE)
MAX_PHRASES_DOC=10     # max phrases per document
D_SAE=8192             # phrase SAE dimension
K_SPARSE=16            # phrase SAE sparsity

# Analysis
N_FEATURES_TO_LABEL=50 # top features for LLM labeling
N_TOKENS_EXTRA_TRAIN=200000 # tokens for residual collection
```

---

## Common Issues & Fixes

### Issue: Dimension Mismatch in `downstream_classification()`
**Symptom**: `ValueError: Found input variables with inconsistent numbers of samples: [2708, 400]`

**Cause**: Passing phrase-level embeddings with document-level SAE activations

**Fix**: Pool phrase embeddings to document level using `pool_embeddings_by_document()`
```python
doc_emb_energy = pool_embeddings_by_document(
    phrase_embeddings[energy_phrase_mask],
    phrase_to_doc[energy_phrase_mask]
)
```

### Issue: `compute_metrics()` Crashes on Document-Level Vectors
**Symptom**: Shape mismatch or type error in SAE encode/decode

**Cause**: Model expected raw input, not SAE code vectors

**Fix**: `compute_metrics()` auto-detects SAE vectors via `model.cfg.d_sae` and uses latent-space reconstruction

### Issue: Pipeline 2 Phrase-to-Document Mismatch
**Symptom**: Inconsistent phrase counts across documents

**Cause**: Documents split into phrases inconsistently or phrase cache corrupted

**Fix**: Validate `test_p2d_arr` shape matches `test_phrase_emb` length; clear cache if needed

---

## Caching Strategy

All expensive computations are cached:

```
results_v7/
  cache/
    p1_all_doc_acts.pt              # Pipeline 1 max-pooled activations
    p1_all_doc_acts_ext_d1024.pt    # With ExtendedSAE (if enabled)
    p1_per_doc_token_data.pkl       # Token-level metadata
    p2_sae_*.pt                     # Phrase SAE weights
    test_phrase_emb_*.pt            # Phrase embeddings
    email_phrase_emb_*.pt           # Email phrase embeddings
  
  p1_*.json, p1_*.pt                # Pipeline 1 results
  p2_*.json, p2_*.pt                # Pipeline 2 results
```

Delete specific cache to force recomputation.

---

## Running the Pipeline

```bash
# Full run (GPU recommended)
cd /home/h21486/SAE
sbatch run_sae.slurm

# Local CPU test
export USE_FROZEN_CORE=0 N_TRAIN=50 N_TEST=10
python saev5.py
```

Expected runtime: ~2-3 hours on A100 with full corpus.

---

## References

- Gemma Scope SAE: https://huggingface.co/google/gemma-scope-2-4b-it-res
- F2LLM: https://huggingface.co/DDPM/F2LLM-v2-80M
- SAE-Lens: https://github.com/jbloomAus/SAELens
- Interpretability via Sparsity: https://arxiv.org/abs/2309.08600
