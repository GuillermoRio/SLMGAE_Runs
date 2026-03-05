import pandas as pd

#Pasar una matriz ya puesta en formato para SLMGAE
coex = pd.read_csv('PC_data_2/F1_F2_coexpr_for_train.txt', sep='\t', header=None)

coex = coex /1000

coex.to_csv('PC_data_2/F1_F2_coexpr_for_train.txt', sep='\t', index=False, header=None)