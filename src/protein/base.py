import Bio
import Bio.PDB
import os
import numpy as np

from .constants import AA20_3_TO_1

class BaseProtein():
    def __init__(self, file: str | None = None, 
                 sequence: str | None = None, 
                 id: str = 'default') -> None:
        
        if file is None and sequence is None:
            raise Exception("Provide pdb file or sequence")
        
        if file:
            if file.endswith('.pdb'):
                pdbparser = Bio.PDB.PDBParser(QUIET=True)
                self.struct = pdbparser.get_structure(id, file)

                n_chains = 0
                for chain in self.struct.get_chains():
                    n_chains += 1

                if n_chains > 1:
                    raise Exception('Method not designed for multiple chains')
                
                if id == 'default':
                    self.id = os.path.basename(file).replace('.pdb', '')
                else:
                    self.id = id
                self.sequence = ''.join([AA20_3_TO_1[res.resname] for res in chain.get_residues()])

                if sequence is not None:
                    assert self.sequence == sequence
            else:
                raise Exception(f"{file} is not a pdb file")
        
        if sequence and file is None:
            if id == 'default':
                raise Exception("Provide id")
            
            self.id = id
            self.sequence = sequence


    def get_residues(self, resnums: list):
        '''
            resnums starts from 0
        '''
        return ''.join([self.sequence[i] for i in resnums])
    

class FoldedProtein(BaseProtein):
    def __init__(self, file: str | None = None, 
                 sequence: str | None = None, 
                 id: str = 'default') -> None:
        super().__init__(file, sequence, id)
        
        self.plddts = np.array([a.get_bfactor() for a in self.struct.get_atoms()])
        self.plddt = self.plddts.mean()
        self.pTM = None
        self.pAE = None

        if os.path.isfile(file.replace('.pdb', '.meta.npz')):
            metadata = np.load(file.replace('.pdb', '.meta.npz'))
            self.pTM = metadata['ptm']
            self.pAE = metadata['predicted_aligned_error']
            self.metadata = dict(metadata)

    