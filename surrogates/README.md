# Surrogate Modeling for Protein Sequence Activity

This folder contains the notebook workflow for training and evaluating surrogate models on protein fitness data.

Primary notebook:
- `surrogates.ipynb`

## Scope

The notebook demonstrates four model categories for sequence-to-activity prediction:

1. One-hot encoding based supervised models
2. PLM embedding based supervised models
3. Zero-shot scoring from PLMs
4. PLM fine-tuning (regression and contrastive)

The default and primary PLM in this workflow is **ESMC**.

## Example Dataset

The tutorial uses the cleaned GB1 dataset from:
- `../data/dataset_gb1.csv`

It also uses the wild-type GB1 sequence for zero-shot and contrastive workflows:
- `../data/GB1_WT.fasta`

## Environment Setup

From the repository root:

```bash
pip install -r surrogates/requirements.txt
```

Then launch Jupyter and open:

- `surrogates/surrogates.ipynb`

## ESMC vs ESM2 Dependency Note

This folder is configured for ESMC by default (`esm` package in `requirements.txt`).

If you want to use **ESM2** instead:

1. Update dependencies in `surrogates/requirements.txt`:
   - remove `esm`
   - add `fair-esm`
2. Update notebook/model wrapper usage to the ESM2 path in `src/esm/esm2.py` where needed.

Using separate virtual environments for ESMC and ESM2 is recommended to avoid package/version conflicts.

## What the Notebook Covers

- Data loading and split masks (`train/val/test`) based on `split_id`
- One-hot feature extraction and baseline regressors (Ridge, Random Forest, MLP)
- ESMC embedding extraction (`mean` or `concat`) with supervised heads
- Zero-shot scores (`wt_marginal`, `masked_marginal`, `pseudolikelihood`)
- PLM fine-tuning with:
  - `ESMCLoraRegression`
  - `ESMCConFit`

For implementation details of model classes and training wrappers, see:
- `../src/models/`
- `../src/esm/`
