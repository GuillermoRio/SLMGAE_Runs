import pandas as pd
import numpy as np

me = pd.read_csv('F1_F2_me_for_train.txt', sep='\t', header=None).to_numpy()

min_max = me[me > 0]

minV = min_max.min()
maxV = min_max.max()

epsilon = 1e-6

me_norm = np.where(
    me == 0,
    0,
    epsilon + (1 - epsilon) * (me - minV) / (maxV - minV)
)

sol = me_norm[me_norm > 0]
df_norm = pd.DataFrame(me_norm)

df_norm.to_csv('F1_F2_me_for_train.txt', header=None, index=False, sep='\t')
