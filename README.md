# NPN: Identifying and Disambiguating the Italian NPN Construction in BERT's Family

Repository for the CMCL 2026 paper:

> **Gorzoni, G., Pannitto, L., & Masini, F. (2026).**  
> *“Layer su Layer”: Identifying and Disambiguating the Italian NPN Construction in BERT’s Family.*  
> Proceedings of CMCL 2026.

---

## Overview

This repository contains the code, datasets, and supplementary materials for the study of the Italian **NPN (noun–preposition–noun)** constructional family (e.g. *strato su strato*, *faccia a faccia*, *giorno dopo giorno*) within a **Construction Grammar** and **interpretability** framework.

The project investigates whether and to what extent contextual embeddings extracted from BERT-family models encode:

- constructional identity
- construction-specific semantic information
- distinctions between NPN constructions and structurally related distractors

The experiments are based on probing classifiers trained over contextual embeddings extracted across Transformer layers.

---

## Repository Structure

```bash
NPN/
│
├── data/                     # datasets and train/test splits
├── probing/                  # probing experiments
├── embeddings/               # contextual embedding extraction
├── pca/                      # PCA visualizations
├── plots/                    # figures and confusion matrices
├── annotazione.md            # annotation guidelines
└── README.md
```

---

## Dataset

The repository includes datasets for:

- construction identification
- semantic disambiguation
- distractor configurations
- cross-preposition generalization experiments

The data are based on the Italian NPN constructional family described in Masini (2024).

### Semantic Labels

The semantic disambiguation task includes the following constructional meanings:

- `succession/iteration/distributivity`
- `greater_plurality/accumulation`
- `juxtaposition/contact`

Detailed annotation guidelines are available in:

```bash
annotazione.md
```

---

## Probing Experiments

The probing pipeline:

1. extracts contextual embeddings from BERT-family models
2. probes embeddings layer-by-layer
3. evaluates:
   - construction identification
   - semantic disambiguation
4. compares:
   - `[UNK]` embeddings
   - `PREP` embeddings
   - FastText baselines
   - control classifiers

---

## PCA Visualizations

Interactive PCA visualizations of contextual embeddings across Transformer layers are available here:

🔗 https://gretagorzoni00.github.io/NPN_contextual_embeddings/

The visualizations provide a geometric exploration of the embedding space for:

- NPN constructions vs distractors
- semantic clusters
- layer-wise representation dynamics

Animated versions are also available.

---

## Models

The experiments use several BERT-family models, including:

- `dbmdz/bert-base-italian-cased`
- `UmBERTo`
- multilingual BERT variants

Embeddings are extracted using Hugging Face Transformers.

---

## Related Resources

### Paper

[CMCL 2026 paper — forthcoming]

### Dataset (Zenodo)

[Zenodo dataset link]

### Poster

[![CMCL 2026 Poster](assets/poster_preview.png)](assets/poster_cmcl2026.pdf)

Click on the image to open the full poster PDF.

---

## Citation

```bibtex
@inproceedings{gorzoni2026layersulayer,
  title={“Layer su Layer”: Identifying and Disambiguating the Italian NPN Construction in BERT’s Family},
  author={Gorzoni, Greta and Pannitto, Ludovica and Masini, Francesca},
  booktitle={Proceedings of CMCL 2026},
  year={2026}
}
```

---
