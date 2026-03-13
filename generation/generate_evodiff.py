import sys
sys.path.append('..')

import torch

from evodiff.pretrained import OA_DM_640M
from evodiff.conditional_generation import inpaint_simple 

torch.hub.set_dir('/data/users/kgeorge/workspace/evodiff')

from src import BaseProtein
from src.utils import helper
from src.utils.constants import *

protein = BaseProtein(file='/nethome/kgeorge/workspace/DomainPrediction/Data/gxps/gxps_ATC_AF.pdb')

start_idx, end_idx = A_gxps_atc[-1]+1, C_gxps_atc[0]
start_idx, end_idx

idr_length = end_idx - start_idx
masked_sequence = protein.sequence[0:start_idx] + '#' * idr_length + protein.sequence[end_idx:]

assert masked_sequence.count('#') == 115 ## This is our T domain length, which is being masked out for generation

checkpoint = OA_DM_640M()
model, collater, tokenizer, scheme = checkpoint

model = model.cuda()

sequence = protein.sequence

fasta_file = '/nethome/kgeorge/workspace/DomainPrediction/Data/round_3_exp/evodiff_5000.fasta'
N_GENERATIONS = 2000
for i in range(N_GENERATIONS):
    seq_dict = {}
    sample, entire_sequence, generated_idr = inpaint_simple(model, sequence, start_idx, end_idx, tokenizer=tokenizer, device='cuda')
    id = f'gxps_ATC_evodiff_gen_{i}'
    seq_dict[id] = entire_sequence

    print(id)
    helper.create_fasta(seq_dict, fasta_file, append=True)