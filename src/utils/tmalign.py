import subprocess
import re

class TMalign():
    def __init__(self, path) -> None:
        self.tmalign = path

    def run(self, prot_a, prot_b):
        out = subprocess.check_output([self.tmalign, prot_a, prot_b])
        data = str(out).split("\\n")
        for d in data:
            x = re.sub(r"\s\s+", " ", d).split(' ')
            if x[0] == 'Aligned':
                rmsd = float(x[4][:-1])
                seq_id = float(x[6])
            elif x[0] == 'TM-score=':
                tm_score = float(x[1])
                break

        return {
            'tm_rmsd': rmsd,
            'tm_seq_id': seq_id,
            'tm_score': tm_score
        }