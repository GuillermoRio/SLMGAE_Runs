import pandas as pd

def index_to_gen():
    genes = pd.read_csv("../data/List_Proteins_in_SL_panc.txt", header=None, names=['gene'])
    pairs = pd.read_csv('results/False.csv', sep='\t')[['gen1','gen2','score']]

    dicionario = {i: name for i, name in enumerate(genes['gene'])}

    pairs_filtered = []

    for _, row in pairs.iterrows():
        gen1_num = row['gen1']
        gen2_num = row['gen2']
        score = row['score']
        pairs_filtered.append({
                'gen1': dicionario[gen1_num],
                'gen2': dicionario[gen2_num],
                'score': score
            })
        
    return pd.DataFrame(pairs_filtered)