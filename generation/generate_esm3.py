import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append('..')

import torch

from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig
from esm.utils.structure.protein_chain import ProteinChain

from src.utils import helper
from src.utils.constants import *

protein = ProteinChain.from_pdb('../data/gxps_ATC_AF.pdb')

sequence_prompt = ''.join([protein[i].sequence if i in A_gxps_atc + C_gxps_atc else '_' for i in range(len(protein))])
structure_prompt = torch.tensor(protein.atom37_positions)

model: ESM3InferenceClient = ESM3.from_pretrained("esm3_sm_open_v1").to("cuda")

fasta_file = '../data/esm3_gen.fasta' ## file loc

N_GENERATIONS = 10
temperature = 0.5
print(f'T domain: {protein[T_gxps_atc].sequence}')
for idx in range(N_GENERATIONS):
    
    sequence_prediction_config = GenerationConfig(
        track="sequence", 
        num_steps=sequence_prompt.count("_") // 2, 
        temperature=temperature
    )
    esm_protein = ESMProtein(sequence=sequence_prompt, coordinates=structure_prompt)
    generated_protein = model.generate(esm_protein, sequence_prediction_config)

    print(f"T domain: {''.join([generated_protein.sequence[i] for i in range(len(generated_protein.sequence)) if i in T_gxps_atc])}")

    assert protein[A_gxps_atc].sequence == ''.join([generated_protein.sequence[i] for i in range(len(generated_protein.sequence)) if i in A_gxps_atc])
    assert protein[C_gxps_atc].sequence == ''.join([generated_protein.sequence[i] for i in range(len(generated_protein.sequence)) if i in C_gxps_atc])

    seq_dict = {}
    gen_idx = f'gxps_ATC_esm3_str_gen_{idx}'
    seq_dict[gen_idx] = generated_protein.sequence

    print(gen_idx)

    helper.create_fasta(seq_dict, fasta_file, append=True)