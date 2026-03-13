# Generation

This folder contains generation workflows for designing T-domain sequences while keeping A and C domains fixed.

The scripts use residue-index domain definitions from [src/utils/constants.py](../src/utils/constants.py). You can update those definitions or add new ones for your own protein/domain layout.

## What Is In This Folder

- [generate_esm3.py](generate_esm3.py): T-domain generation with ESM3.
- [generate_evodiff.py](generate_evodiff.py): T-domain generation with EvoDiff inpainting.
- [generate_pmpnn.ipynb](generate_pmpnn.ipynb): T-domain generation with ProteinMPNN.

## Domain Definitions (A, T, C)

Domain boundaries are index lists stored in [src/utils/constants.py](../src/utils/constants.py), for example:

- `A_gxps_atc = [0..488]`
- `T_gxps_atc = [505..574]`
- `C_gxps_atc = [604..1033]`

How to adapt for your case:

1. Open [src/utils/constants.py](../src/utils/constants.py).
2. Create or edit index lists for your new protein.
3. Use those symbols in generation scripts/notebook.
4. Ensure indexing matches your loaded sequence/PDB residue order.

or 

1. Create as lists in the script directly and use them

## Input Structure (PDB)

All generation workflows read a PDB structure and apply masking/design based on the domain indices.

- Provided sample structure: [data/gxps_ATC_AF.pdb](../data/gxps_ATC_AF.pdb)
- For your use case, replace the input PDB path and define matching A/T/C index ranges.

## Environment Setup (Main)

Use this environment for most workflows (including EvoDiff and ProteinMPNN notebook dependencies):

```bash
conda create -n myenv python=3.10.13 pip
conda activate myenv
pip install -r requirements.txt
```

## ProteinMPNN Workflow

```bash
git submodule update --init --recursive
```

ProteinMPNN is included as a git submodule. After submodule initialization, follow [generate_pmpnn.ipynb](generate_pmpnn.ipynb) for ProteinMPNN generation.

## EvoDiff Workflow

Before first generation run, download EvoDiff weights once:

```python
from evodiff.pretrained import OA_DM_640M
checkpoint = OA_DM_640M()
```

This downloads and caches the model weights.

Then run the script from this folder:

```bash
python generate_evodiff.py
```

Notes:

- The script currently assumes CUDA (`model = model.cuda()`).
- It masks the region between end of A and start of C, and checks a default masked length of 115 for the sample setup.

## ESM3 Workflow (Separate Environment Recommended)

Because ESM3/ESM2 setups can conflict around package naming/versioning, use a separate environment:

```bash
conda create -n myenv_esm3 python=3.10.13 pip
conda activate myenv_esm3
pip install esm
```

You need a Hugging Face access token to download the ESM3 model. Follow ESM instructions:

- https://github.com/evolutionaryscale/esm

Run once to authenticate and cache model weights:

```python
from huggingface_hub import login
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient

login()
model: ESM3InferenceClient = ESM3.from_pretrained("esm3-open").to("cuda")  # or "cpu"
```

Then run:

```bash
python generate_esm3.py
```

Notes:

- Script currently loads sample PDB `../data/gxps_ATC_AF.pdb`.
- Script currently uses CUDA by default (`to("cuda")`).
- Generated sequences are appended to FASTA output in `../data/esm3_gen.fasta`.

## Typical Adaptation Checklist

1. Prepare your PDB file and place it under [data](../data) (or update script path).
2. Define A/T/C indices in [src/utils/constants.py](../src/utils/constants.py).
3. Confirm masked/design region length is correct for your protein.
4. Run generation script/notebook for your chosen model.
5. Review FASTA outputs in [data](../data).
