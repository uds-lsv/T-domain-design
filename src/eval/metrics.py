import os
from tqdm import tqdm
from Bio import AlignIO
import re
import numpy as np
from ..utils.hmmtools import HmmerTools
from ..utils import helper
from ..utils.tmalign import TMalign

def compute_sequence_identity(wt: str, gen: str, hmm: str) -> list:
    """Compute sequence identity between wildtype and generated sequences using HMM alignment.
    
    Args:
        wt: Path to wildtype FASTA file (single sequence)
        gen: Path to generated sequences FASTA file
        hmm: Path to HMM profile for alignment
        
    Returns:
        List of sequence identity scores (0-1) for each generated sequence
        
    Raises:
        Exception: If FASTA files are empty or WT contains multiple sequences
    """
    if len(helper.read_fasta(wt, mode='str')) == 0 or len(helper.read_fasta(gen, mode='str')) == 0:
        raise Exception('One of the fasta files is empty')
    
    if len(helper.read_fasta(wt, mode='str')) > 1:
        raise Exception('Current functionality is limited to one WT sequence. Feel free to extend :D')
    
    ## need to select seq from gen and combine with wt to create a fasta file
    wt_record = helper.read_fasta(wt)[0]
    gen_records = helper.read_fasta(gen)

    ## updating with query length
    query_len = len(str(wt_record.seq))

    hmmer = HmmerTools()
    seq_id_list = []
    for rec in tqdm(gen_records):
        tmp_file = os.path.join(os.path.dirname(gen), rec.id + '.tmp.fasta')
        helper.create_fasta(
            {
                wt_record.id : str(wt_record.seq),
                rec.id : str(rec.seq)
            },
            tmp_file
        )
        hmmer.hmmalign(hmm_path=hmm, fasta_file=tmp_file)
        alignment = AlignIO.read(tmp_file.replace('.fasta', '.stockholm'), "stockholm")
        alignment_length = alignment.get_alignment_length()

        assert len(alignment) == 2

        seq_id = 0
        ## should we treat caps?
        for s1, s2 in zip(alignment[0].seq, alignment[1].seq):
            if s1 != '-' and s2 != '-' and s1 != '.' and s2 != '.' and s1 == s2:
                seq_id += 1

        seq_id_list.append(seq_id/query_len)

        os.remove(tmp_file)
        os.remove(tmp_file.replace('.fasta', '.stockholm'))

    return seq_id_list


def search_and_filter_motif(file: str, pattern: str = 'FF.{2}GG.{1}S'):
    """Search for regex pattern motifs in protein sequences.
    
    Args:
        file: Path to FASTA file containing sequences
        pattern: Regex pattern to search for (default: 'FF.{2}GG.{1}S')
        
    Returns:
        None (prints percentage of sequences containing motif)
        
    Note:
        Currently prints results. Could be extended to return matches or save to file.
    """

    records = helper.read_fasta(file)

    matches = []
    for rec in records:
        if re.search(pattern, str(rec.seq)):
            matches.append(rec)
    
    print(f'{len(matches)*100/len(records)}% records contain motif')


def compute_perplexity(model, sequence, mask_token='<mask>'):
    """Compute pseudo-perplexity of a protein sequence using language model.
    
    Formula: pseudoperplexity(x) = exp( -1/L \sum_{i=1}_{L} [log( p(x_{i}|x_{j!=i}) )] )
    Note: This function is designed specifically for ESM2 class
    
    Args:
        model: ESM2 model instance with get_prob() and tok_to_idx attributes
        sequence: Protein sequence string
        mask_token: Token used for masking positions (default: '<mask>')
        
    Returns:
        Pseudo-perplexity value (float)
    """
    
    sum_log = 0
    for pos in tqdm(range(len(sequence))):
        masked_query = list(sequence)
        assert mask_token not in masked_query
        masked_query[pos] = mask_token
        masked_query = ''.join(masked_query)
        prob = model.get_prob(sequence=masked_query)

        assert prob.shape[0] == len(sequence)

        prob_pos = np.log(prob[pos, model.tok_to_idx[sequence[pos]]])
        
        sum_log += prob_pos

    return np.exp(-1*sum_log/len(sequence))


def compute_TMscore(tm_path, pdbs_path, ref_path, prefix='', save_meta=True, force=False):
    """Compute TM-score between reference structure and multiple PDB files.
    
    Args:
        tm_path: Path to TM-align executable
        pdbs_path: Directory containing PDB files to compare
        ref_path: Path to reference PDB structure
        prefix: Prefix for metadata keys when saving
        save_meta: Whether to save metadata to .meta.npz files
        force: Whether to force overwrite existing metadata
        
    Returns:
        List of TM-scores for each PDB file in pdbs_path
        
    Note:
        Also saves alignment metadata to .meta.npz files if save_meta=True
    """
    tmalign = TMalign(tm_path)

    TM_scores = []
    for f in tqdm(os.listdir(pdbs_path)):
        file = os.path.join(pdbs_path, f)
        if f.endswith('.pdb'):
            res = tmalign.run(ref_path, file)
            TM_scores.append(res['tm_score'])

            if save_meta:
                meta_file = file.replace('.pdb', '.meta.npz')
                for key in res:
                    helper.update_metadata(meta_file, prefix+key, res[key], force=force)
            
    return TM_scores


    