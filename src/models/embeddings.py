import numpy as np
from tqdm import tqdm


def one_hot_encode(sequences) -> np.ndarray:
    """Encode a protein sequence as a one-hot array."""
    embeddings = []
    for seq in tqdm(sequences):
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        aa_to_index = {aa: i for i, aa in enumerate(amino_acids)}
        one_hot = np.zeros((len(seq), len(amino_acids)))
        for i, aa in enumerate(seq):
            if aa in amino_acids:
                one_hot[i, aa_to_index[aa]] = 1
    
        embeddings.append(one_hot.flatten())  

    embeddings = np.stack(embeddings, axis=0)

    return embeddings.astype(np.float32)