import pandas as pd
import glob
import os
import numpy as np

df1 = pd.read_csv(
    'data_aux/F1_F2_coexpr_for_train.txt',
    sep='\t',
    header=None
)
df1.columns = df1.columns.astype(str)
df1.to_parquet('data_aux/F1_F2_coexpr_for_train.parquet', index=False)

df2 = pd.read_csv(
    'data_aux/F1_F2_me_for_train.txt',
    sep='\t',
    header=None
)
df2.columns = df2.columns.astype(str)
df2.to_parquet('data_aux/F1_F2_me_for_train.parquet', index=False)

df3 = pd.read_csv(
    'data_aux/F1_F2_pathway_for_train.txt',
    sep='\t',
    header=None
)
df3.columns = df3.columns.astype(str)
df3.to_parquet('data_aux/F1_F2_pathway_for_train.parquet', index=False)

df4 = pd.read_csv(
    'data_aux/F1_F2_ppi_for_train.txt',
    sep='\t',
    header=None
)
df4.columns = df4.columns.astype(str)
df4.to_parquet('data_aux/F1_F2_ppi_for_train.parquet', index=False)

df5 = pd.read_csv(
    'data_aux/F1_F2_proteincomplex_for_train.txt',
    sep='\t',
    header=None
)
df5.columns = df5.columns.astype(str)
df5.to_parquet('data_aux/F1_F2_proteincomplex_for_train.parquet', index=False)

neg_edge = np.load("data_aux/neg_edge_binary.npy")
pos_edge = np.load("data_aux/pos_edge_binary.npy")

np.savez("datos.npz", pos_edge=pos_edge, neg_edge=neg_edge)