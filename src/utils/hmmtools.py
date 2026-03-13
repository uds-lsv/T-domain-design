import subprocess
import os
from sklearn.preprocessing import StandardScaler
import pandas as pd


class HmmerTools():
    def __init__(self) -> None:
        self.scaler = StandardScaler()

    def hmmalign(self, hmm_path: str, fasta_file: str, 
                  outfile: str | None = None, 
                  trim: bool = False,
                  verbose: bool = False):
        if outfile is None:
            outfile = fasta_file.replace('.fasta', '.stockholm')

        if trim:
            with open(outfile, "w") as fout:
                subprocess.run(['hmmalign', '--trim', hmm_path, fasta_file], stdout=fout)
        else:
            with open(outfile, "w") as fout:
                subprocess.run(['hmmalign', hmm_path, fasta_file], stdout=fout)

        if verbose:
            print(f'alignment created: {outfile}')

    def hmmsearch(self, hmm_path: str, fasta_file: str, 
                  outfile: str | None = None, 
                  verbose: bool = False,
                  save: bool = False,
                  return_df: bool = True) -> pd.DataFrame | None: 
        if outfile is None:
            outfile = fasta_file.replace('.fasta', '.hmmsearch.out')

        tblfile = fasta_file.replace('.fasta', '.hmmsearch.tbl.out')

        if save:
            with open(outfile, "w") as fout:
                subprocess.run(['hmmsearch', '--tblout', tblfile, hmm_path, fasta_file], stdout=fout)
        else:
            subprocess.run(['hmmsearch', '--tblout', tblfile, hmm_path, fasta_file], stdout = subprocess.DEVNULL)

        if verbose:
            if save:
                print(f'hmmsearch out: {outfile}')
            print(f'hmmsearch tbl out: {tblfile}')

        if return_df:
            df = self.parse_hmmsearch_tblout(tblfile)
            if save :
                os.remove(outfile)
            os.remove(tblfile)

            return df


    def parse_hmmsearch_tblout(self, filename):
        with open(filename) as fin:
            tmp = []
            for line in fin:
                if not line.startswith('#'):
                    tmp.append(line.strip().split()[:7])

        df = pd.DataFrame(tmp, columns=['name', '-', 'domain', 'domain_id', 'Evalue', 'score', 'bias'])
        df['Evalue'] = df['Evalue'].astype(float)
        df['score'] = df['score'].astype(float)
        df['bias'] = df['bias'].astype(float)

        return df

    def sort_by_Eval(self, df, top=None):
        df_tmp = df[['name']].drop_duplicates().sort_values(by=['name']).reset_index(drop=True)
        
        for domain in df['domain'].unique():
            print(f'domain {domain} {df[df["domain"] == domain].shape[0]}')
            df_tmp[domain] = df[df['domain'] == domain].sort_values(by=['name'])['Evalue'].to_numpy()

        df_tmp['sum'] = df_tmp[df['domain'].unique()].sum(axis=1)

        df_tmp['norm_sum'] = self.scaler.fit_transform(df_tmp[df['domain'].unique()]).sum(axis=1)

        df_sorted = df_tmp.sort_values(by=['norm_sum'])
        if top is None:
            return df_sorted['name'].to_numpy()
        else:
            return df_sorted['name'].to_numpy()[:top]
        
    def sort_by_Eval_domain(self, df, domain, top=None):

        df_tmp = df[df['domain'] == domain]
        df_sorted = df_tmp.sort_values(by=['Evalue'])

        if top is None:
            return df_sorted['name'].to_numpy()
        else:
            return df_sorted['name'].to_numpy()[:top]