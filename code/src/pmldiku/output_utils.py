from typing import List, Dict
from pathlib import Path
import pandas as pd

class FIDScoreOutput:
    def __init__(self, n_decimals: int =3) -> None:
        self.memory: Dict[str: List[float]] = dict()
        self.n_decimals = n_decimals

    def add(self, name:str, score:float) -> None:
        self.memory[name] = [score]

    def generate_table(self):
        table = pd.DataFrame(self.memory, index=['score'])
        return table.to_latex(float_format="%.{}f".format(self.n_decimals))

    def __repr__(self):
        return str(self.memory)

    
def save_fig(fig, path, name) -> None:
    fig.savefig(path / Path(f'{name}.png'))