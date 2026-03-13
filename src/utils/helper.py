import os
import json
from typing import List, Union
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def create_fasta(sequences: dict | List[SeqRecord], 
                 file: str = 'default.fasta', append: bool = False):
    if os.path.isfile(file) and not append:
        raise Exception(f'fasta file {file} exists. If you want to append try with append=True')
    
    if type(sequences) is dict:
        records = []
        for id in sequences:
            records.append(SeqRecord(Seq(sequences[id]), id=id, description=""))
    else:
        assert type(sequences[0]) is type(SeqRecord)
        records = sequences

    if append:
        with open(file, 'a') as f:
            SeqIO.write(records, f, "fasta")
    else:
        with open(file, 'w') as f:
            SeqIO.write(records, f, "fasta")


def read_fasta(file: str, mode: str='default') -> List[type[SeqRecord] | str]:
    records = SeqIO.parse(file, "fasta")

    if mode == 'str':
        return [str(i.seq) for i in records]
    else:
        return [i for i in records]
    
def update_metadata(file, key, value, force=False):
    if os.path.isfile(file):
        metadata = dict(np.load(file))
        if not force and key in metadata:
            raise Exception(f'key {key} found. if you want to force add then keep force=True')
        metadata[key] = value
        np.savez(file.replace('.npz', ''), **metadata)
    else:
        raise Exception(f'{file} metadata not found')
    
def update_metadata_json(json_file, protein_id, key, value, force=False):
    try:
        with open(json_file, 'r') as file:
            metadata = json.load(file)
    except:
        metadata = {}

    if protein_id in metadata:
        if not force and key in metadata[protein_id]:
            raise Exception(f'key {key} found for protein {protein_id}. If you want to force add then keep force=True')
        metadata[protein_id][key] = value
    else:
        metadata[protein_id] = {
            key: value
        }

    with open(json_file, 'w') as file:
        json.dump(metadata, file, indent=4)

def hamming_distance(seq1, seq2):
    # Ensure the sequences are of the same length
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be of equal length to compute Hamming distance.")
    
    # Count differences
    return sum(c1 != c2 for c1, c2 in zip(seq1, seq2))