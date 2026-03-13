import sys
sys.path.append('..')

import os
import torch

from DomainPrediction import BaseProtein

root = '../..'
data_path = os.path.join(root, 'Data/')

## Read Protein
# protein = BaseProtein(file=os.path.join(data_path, 'GxpS_ATC_AF.pdb'))
# A = [i for i in range(33,522)] ## 34-522
# C = [i for i in range(637,1067)] ## 638-1067
# T = [i for i in range(538, 608)] ## 539-608

from typing import Dict, List
from tqdm import tqdm
import numpy as np
import torch

from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils import OFProtein, atom14_to_atom37, to_pdb

from DomainPrediction.utils import helper

class esmFold():
    def __init__(self, device='cpu') -> None:
        self.model = EsmForProteinFolding.from_pretrained("/data/users/kgeorge/workspace/esm2/esmfold")
        self.tokenizer = AutoTokenizer.from_pretrained("/data/users/kgeorge/workspace/esm2/esmfold")
        # self.model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
        # self.tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        self.device = device

        if self.device == 'gpu':
            self.model = self.model.cuda()
            self.model.trunk.set_chunk_size(256)

    def forward(self, inputs):
        if self.device == 'gpu':
            for key in inputs:
                inputs[key] = inputs[key].cuda()
                
        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs

    def structures_from_fasta(self, file: str, save_path: str):
        records = helper.read_fasta(file)

        for rec in tqdm(records):
            outputs = self.get_structure(str(rec.seq))
            file = os.path.join(save_path, rec.id)
            self.output_to_pdb(outputs, file)

    def get_structure(self, sequence: str):
        inputs = self.tokenizer([sequence], return_tensors="pt", add_special_tokens=False)
        outputs = self.forward(inputs)

        return outputs
    
    @staticmethod
    def output_to_pdb(output: Dict, file: str, save_meta: bool = True):
        '''
            Adapted from https://github.com/huggingface/transformers/blob/979d24e7fd82a10d1457d500bef8ec3b5ddf2f8a/src/transformers/models/esm/modeling_esmfold.py#L2292
        '''
        output = {k: v.to("cpu").numpy() for k, v in output.items()}
        pdbs = []
        final_atom_positions = atom14_to_atom37(output["positions"][-1], output)
        final_atom_mask = output["atom37_atom_exists"]

        for i in range(output["aatype"].shape[0]):
            aa = output["aatype"][i]
            pred_pos = final_atom_positions[i]
            mask = final_atom_mask[i]
            resid = output["residue_index"][i] + 1

            pred = OFProtein(
                aatype=aa,
                atom_positions=pred_pos,
                atom_mask=mask,
                residue_index=resid,
                b_factors=output["plddt"][i],
            )
            pdbs.append(to_pdb(pred))

        assert output["aatype"].shape[0] == 1

        meta = {
            "predicted_aligned_error" : output["predicted_aligned_error"][0],
            "ptm" : output["ptm"]
        }

        with open(file + '.pdb', "w") as f:
            f.write(pdbs[0])

        if save_meta:
            np.savez(file + '.meta', **meta)


esmfold = esmFold(device='gpu')

## save pdbs from a fasta file
save_path = os.path.join(data_path, 'esm3_experiments/gxps_exp/gxps_pdbs')
gen = os.path.join(data_path, 'esm3_experiments/gxps_exp/gxps_esm3_1000.fasta')
esmfold.structures_from_fasta(file=gen, save_path=save_path)

