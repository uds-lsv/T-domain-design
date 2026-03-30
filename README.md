# T-domain-design

Code and workflows accompanying the preprint:

**Generative AI designs functional thiolation domains for reprogramming non-ribosomal peptide synthetases**  
bioRxiv (2026), DOI: https://doi.org/10.64898/2026.03.03.709401  

## Project Overview

This repository focuses on AI-assisted design and evaluation of thiolation (T) domains in NRPS systems.

The codebase currently provides:

- sequence generation workflows for T-domain redesign with fixed A/C context
- surrogate modeling workflows for protein sequence activity prediction
- utilities for sequence/structure handling and evaluation metrics
- a vendored ProteinMPNN submodule for structure-conditioned design experiments

## Repository Layout

- `data/`:
	- example inputs and generated sequence files used by scripts/notebooks
	- includes `dataset_gb1.csv` and `GB1_WT.fasta` for surrogate-model examples
- `generation/`:
	- generation workflows using ESM3, EvoDiff, and ProteinMPNN
	- see `generation/README.md`
- `surrogates/`:
	- notebook tutorial for surrogate modeling (one-hot, PLM embeddings, zero-shot, fine-tuning)
	- see `surrogates/surrogates.ipynb` and `surrogates/README.md`
- `src/`:
	- reusable Python modules:
		- `src/esm/` model wrappers for ESM2, ESMC, ESM3
		- `src/models/` supervised and fine-tuning model implementations
		- `src/eval/` evaluation utilities
		- `src/protein/`, `src/utils/` protein and helper utilities
- `ProteinMPNN/`:
	- git submodule mirror of ProteinMPNN

## Installation

### 1) Clone and initialize submodules

```bash
git clone <your-fork-or-this-repo-url>
cd T-domain-design
git submodule update --init --recursive
```

### 2) Create Python environment

```bash
conda create -n tdomain python=3.10.13 pip
conda activate tdomain
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

The main `requirements.txt` is now profile-oriented with comments so you can switch PLM stacks:

- ESMC/ESM3 path (recommended default): keep `esm`
- ESM2 path: use `fair-esm` instead of `esm`

Do not install both `esm` and `fair-esm` in the same environment.

## Quick Start

### Generation workflows

From `generation/`:

- ESM3 generation:

```bash
python generate_esm3.py
```

- EvoDiff generation:

```bash
python generate_evodiff.py
```

- ProteinMPNN workflow:
	- open and run `generation/generate_pmpnn.ipynb`

See `generation/README.md` for domain-index configuration and expected input files.

### Surrogate modeling workflow

Open and run:

- `surrogates/surrogates.ipynb`

This notebook covers:

1. one-hot encoding based regressors
2. PLM embedding based regressors
3. zero-shot PLM scores
4. PLM fine-tuning (regression and contrastive)

Default PLM in surrogate workflows is **ESMC**.

## Data Notes

Included example files under `data/`:

- `dataset_gb1.csv`: cleaned GB1 fitness data used in surrogate tutorial
- `GB1_WT.fasta`: GB1 wild-type sequence for zero-shot/fine-tuning sections
- `gxps_ATC_AF.pdb`: example structure used by generation scripts

Generated FASTA files in `data/` (for quick inspection/examples):

- `esm3_gen.fasta`
- `evodiff_gen.fasta`
- `pmpnn_gen.fasta`

## Citation

If you use this repository or build on the methods, please cite the associated preprint:

- Bülbül EF, Bang S, George K, et al. Generative AI designs functional thiolation domains for reprogramming non-ribosomal peptide synthetases. bioRxiv (2026)

BibTeX:

```bibtex
@article{buelbuel2026generative,
  title={Generative AI designs functional thiolation domains for reprogramming non-ribosomal peptide synthetases},
  author={Buelbuel, Emre F and Bang, Seounggun and Geroge, Kevin and Bianchi, Gabriele and Raj, Prateek and Chung, Seonyong and Pauline, Vincent and Hochstrasser, Ramon and Minas, Hannah A and Elgaher, Walid AM and others},
  journal={bioRxiv},
  pages={2026--03},
  year={2026},
  publisher={Cold Spring Harbor Laboratory}
}
```

## Notes and Limitations

- Several scripts/notebooks are research-grade and may contain hardcoded file paths or environment assumptions.
- GPU is assumed in parts of the generation and PLM workflows.
- For ESM2 vs ESMC/ESM3 usage, prefer separate conda environments.

