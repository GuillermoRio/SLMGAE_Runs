import pandas as pd
from pathlib import Path
import scipy.sparse as sp


carpetaInput = '../data_aux/'
adjs = []

archivos = sorted(Path(carpetaInput).glob("*.parquet"))
    
for i, archivo in enumerate(archivos[:5]):
    print(f'Vista{i+1}: {archivo.name}')
    df = pd.read_parquet(archivo)
    adjs.append(sp.coo_matrix(df.values))

