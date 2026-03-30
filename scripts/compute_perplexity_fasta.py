import sys
sys.path.append('..')

import os
from tqdm import tqdm

from src.esm.esm2 import ESM2
from src.eval import metrics
from src.utils import helper

esm2 = ESM2(device='gpu')

fasta_path = '../data/esm3_gen.fasta'

records = helper.read_fasta(fasta_path)
for rec in records:
    perplexity = metrics.compute_perplexity(esm2, str(rec.seq))

    print(f"Perplexity for {rec.id}: {perplexity}")