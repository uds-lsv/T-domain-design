import torch
import esm
from tqdm import tqdm
import numpy as np


class ESM2():
    """ESM2 protein language model wrapper for embeddings and scoring.
    
    Example:
        model = ESM2(model_path='/path/to/esm2_t33_650M_UR50D.pt', device='gpu')
    """
    
    def __init__(self, model_path=None, device='cpu') -> None:
        """Initialize ESM2 model.
        
        Args:
            model_path: Path to ESM2 model checkpoint
            device: Device to use ('cpu' or 'gpu')
        """
        if model_path is None:
            self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            model_path = 'facebook/esm2_t33_650M_UR50D'
        else:
            self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_path)
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval()
        self.device = device

        if 't6_8M' in model_path:
            self.rep_layer = 6
            self.emb_dim = 320
        elif 't30_150M' in model_path:
            self.rep_layer = 30
            self.emb_dim = 640
        elif 't33_650M' in model_path:
            self.rep_layer = 33
            self.emb_dim = 1280
        else:
            raise Exception('I need to work on this. Feel free to extend :)')

        if self.device == 'gpu':
            self.model.cuda()

        self.tok_to_idx = self.alphabet.tok_to_idx
        self.idx_to_tok = {v:k for k,v in self.tok_to_idx.items()}

    def get_res(self, sequence, rep_layer=None):
        """Get model representations for a single sequence.
        
        Args:
            sequence: Protein sequence string
            rep_layer: Representation layer to extract (default: model's rep_layer)
            
        Returns:
            Dictionary containing model outputs including representations
        """
        if rep_layer is None:
            rep_layer = self.rep_layer

        data = [
            ("protein1", sequence)
        ]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)

        if self.device == 'gpu':
            batch_tokens = batch_tokens.cuda()

        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[rep_layer], return_contacts=False)

        return results

    def get_res_batch(self, sequences, rep_layer=None):
        """Get model representations for multiple sequences in batch.
        
        Args:
            sequences: List of protein sequence strings
            rep_layer: Representation layer to extract (default: model's rep_layer)
            
        Returns:
            Tuple of (results dict, batch_lens)
        """
        if rep_layer is None:
            rep_layer = self.rep_layer

        data = [
            (f"P{i+1}", seq) for i, seq in enumerate(sequences)
        ]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)

        if self.device == 'gpu':
            batch_tokens = batch_tokens.cuda()

        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[rep_layer], return_contacts=True)

        return results, batch_lens

    def get_logits(self, sequence):
        """Get logits for a sequence.
        
        Args:
            sequence: Protein sequence string
            
        Returns:
            Tensor of logits
        """
        results = self.get_res(sequence)
        return results['logits']

    def get_prob(self, sequence):
        """Get amino acid probabilities for each position in sequence.
        
        Args:
            sequence: Protein sequence string
            
        Returns:
            Numpy array of probabilities (seq_len, vocab_size)
        """
        logits = self.get_logits(sequence)
        prob = torch.nn.functional.softmax(logits, dim=-1)[0, 1:-1, :] # 1st and last are start and end tokens

        return prob.cpu().numpy()
    
    def get_embeddings_mean(self, sequences):
        """Get mean-pooled sequence embeddings.
        
        Args:
            sequences: List of protein sequence strings
            
        Returns:
            Numpy array of embeddings (n_seqs, emb_dim)
        """
        embeddings = []
        for seq in tqdm(sequences):
            rep = self.get_res(sequence=seq)
            embeddings.append(rep['representations'][self.rep_layer][:,1:-1,:].mean(1).cpu().numpy())

        embeddings = np.concatenate(embeddings, axis=0)

        return embeddings
    
    def get_embeddings_flatten(self, sequences):
        """Get flattened sequence embeddings (all positions concatenated).
        
        Args:
            sequences: List of protein sequence strings
            
        Returns:
            Numpy array of flattened embeddings (n_seqs, seq_len * emb_dim)
        """
        embeddings = []
        for seq in tqdm(sequences):
            rep = self.get_res(sequence=seq)
            embeddings.append(rep['representations'][self.rep_layer][:,1:-1,:].cpu().numpy()[0].flatten())

        embeddings = np.stack(embeddings, axis=0)

        return embeddings
    
    def get_embeddings_cls(self, sequences):
        """Get CLS token embeddings for sequences.
        
        Args:
            sequences: List of protein sequence strings
            
        Returns:
            Numpy array of CLS embeddings (n_seqs, emb_dim)
        """
        embeddings = []
        for seq in tqdm(sequences):
            rep = self.get_res(sequence=seq)
            embeddings.append(rep['representations'][self.rep_layer][:, 0, :].cpu().numpy())

        embeddings = np.concatenate(embeddings, axis=0)

        return embeddings
    
    def get_embeddings_feature_pool(self, sequences, pool='mean'):
        """Get feature-pooled embeddings across embedding dimensions.
        
        Args:
            sequences: List of protein sequence strings
            pool: Pooling method ('mean' or 'sum')
            
        Returns:
            Numpy array of pooled embeddings (n_seqs, seq_len)
        """
        embeddings = []
        for seq in tqdm(sequences):
            rep = self.get_res(sequence=seq)
            if pool == 'mean':
                embeddings.append(rep['representations'][self.rep_layer][:,1:-1,:].mean(-1).cpu().numpy())
            elif pool == 'sum':
                embeddings.append(rep['representations'][self.rep_layer][:,1:-1,:].sum(-1).cpu().numpy())
            else:
                raise Exception('pool can only take values mean or sum')
            
        embeddings = np.concatenate(embeddings, axis=0)

        return embeddings
    
    def compute_perplexity(self, sequence, mask_token='<mask>'):
        """Compute pseudo-perplexity of a sequence.
        
        Formula: pseudoperplexity(x) = exp( -1/L \sum_{i=1}_{L} [log( p(x_{i}|x_{j!=i}) )] )
        
        Args:
            sequence: Protein sequence string
            mask_token: Token used for masking positions
            
        Returns:
            Pseudo-perplexity value
        """
        
        sum_log = 0
        for pos in range(len(sequence)):
            masked_query = list(sequence)
            assert mask_token not in masked_query
            masked_query[pos] = mask_token
            masked_query = ''.join(masked_query)
            prob = self.get_prob(sequence=masked_query)

            assert prob.shape[0] == len(sequence)

            prob_pos = np.log(prob[pos, self.tok_to_idx[sequence[pos]]])
            
            sum_log += prob_pos

        return np.exp(-1*sum_log/len(sequence))
    
    def pseudolikelihood(self, mt_sequence, mask_token='<mask>'):
        """Compute pseudolikelihood score for a sequence.
        
        Args:
            mt_sequence: Mutant protein sequence string
            mask_token: Token used for masking positions
            
        Returns:
            Pseudolikelihood score
        """
        score = 0
        for i, aa_mt in enumerate(mt_sequence):

            masked_query_mt = list(mt_sequence)
            masked_query_mt[i] = mask_token
            masked_sequence_mt = ''.join(masked_query_mt)
            masked_log_prob_mt = self.get_log_prob(sequence=masked_sequence_mt)

            idx_mt = self.tok_to_idx[aa_mt]
            score += masked_log_prob_mt[i, idx_mt]

        return score
    
    def get_log_prob(self, sequence):
        """Get log probabilities for each position in sequence.
        
        Args:
            sequence: Protein sequence string
            
        Returns:
            Numpy array of log probabilities (seq_len, vocab_size)
        """
        logits = self.get_logits(sequence)
        log_prob = torch.log_softmax(logits, dim=-1)[0,1:-1,:]

        return log_prob.cpu().numpy()
    
    def get_wildtype_marginal(self, mt_sequence, wt_sequence, wt_log_prob=None):
        """Compute wildtype marginal likelihood score.
        
        Args:
            mt_sequence: Mutant sequence string
            wt_sequence: Wildtype sequence string
            wt_log_prob: Pre-computed wildtype log probabilities (optional)
            
        Returns:
            Tuple of (score, number_of_mutations)
        """
        if wt_log_prob is None:
            assert len(wt_sequence) == len(mt_sequence)
            wt_log_prob = self.get_log_prob(sequence=wt_sequence)

        assert wt_log_prob.shape[0] == len(wt_sequence) == len(mt_sequence)

        n_muts = 0
        score = 0
        for i, (aa_mt, aa_wt) in enumerate(zip(mt_sequence, wt_sequence)):
            if aa_wt != aa_mt:
                ## mutation pos
                n_muts += 1

                idx_mt = self.tok_to_idx[aa_mt]
                idx_wt = self.tok_to_idx[aa_wt]
                score += wt_log_prob[i, idx_mt] - wt_log_prob[i, idx_wt]


        return score, n_muts
    
    def get_masked_marginal(self, mt_sequence, wt_sequence, mask_token = '<mask>'):
        """Compute masked marginal likelihood score.
        
        Args:
            mt_sequence: Mutant sequence string
            wt_sequence: Wildtype sequence string
            mask_token: Token used for masking mutation positions
            
        Returns:
            Tuple of (score, number_of_mutations)
        """
        assert len(wt_sequence) == len(mt_sequence)

        n_muts = 0
        mask_positions = []
        for i, (aa_mt, aa_wt) in enumerate(zip(mt_sequence, wt_sequence)):
            if aa_wt != aa_mt:
                ## mutation pos
                n_muts += 1
                mask_positions.append(i)

        assert len(mask_positions) == n_muts
        masked_query = list(wt_sequence)
        for _pos in mask_positions:
            masked_query[_pos] = mask_token
        masked_sequence = ''.join(masked_query)

        masked_log_prob = self.get_log_prob(sequence=masked_sequence)
        
        score = 0
        _idx = 0
        for i, (aa_mt, aa_wt) in enumerate(zip(mt_sequence, wt_sequence)):
            if aa_wt != aa_mt:
                ## mutation pos

                assert mask_positions[_idx] == i
                _idx += 1

                idx_mt = self.tok_to_idx[aa_mt]
                idx_wt = self.tok_to_idx[aa_wt]
                score += masked_log_prob[i, idx_mt] - masked_log_prob[i, idx_wt]


        return score, n_muts