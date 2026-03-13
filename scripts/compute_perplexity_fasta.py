import sys
sys.path.append('..')

import os
from tqdm import tqdm

from DomainPrediction.esm.esm2 import ESM2
from DomainPrediction.eval import metrics
from DomainPrediction.utils import helper

model_path = '/data/users/kgeorge/workspace/esm2/checkpoints/esm2_t33_650M_UR50D.pt'
esm2 = ESM2(model_path = model_path, device='gpu')

fasta_path = '../../Data/evodiff_experiments/gxps_exp/gxps_evodiff_1000.fasta'
meta_path = '../../Data/evodiff_experiments/gxps_exp/gxps_pdbs'

records = helper.read_fasta(fasta_path)
for rec in records:
    perplexity = metrics.compute_perplexity(esm2, str(rec.seq))
    meta_file = os.path.join(meta_path, rec.id + '.meta.npz')

    print(meta_file, perplexity)
    helper.update_metadata(meta_file, 'esm2_650M_perplexity', perplexity, force=False)