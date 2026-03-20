import codecs
import pandas as pd
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from pathlib import Path

flags = tf.app.flags
FLAGS = flags.FLAGS


def log(message):
    result_file_name = FLAGS.log_file #Coge los logs para enseñar que es lo que esta haciendo el programa
    log_file = result_file_name
    codecs.open(log_file, mode='a', encoding='utf-8').write(message + "\n")
    print(message)

def load_PC_data(carpeta):
    print("loading sl data...")
    adjs = []
    adjs.append(sp.coo_matrix(np.loadtxt(f'{carpeta}F1_F2_ppi_for_train.txt')))    
    adjs.append(sp.coo_matrix(np.loadtxt(f'{carpeta}F1_F2_coexpr_for_train.txt')))
    adjs.append(sp.coo_matrix(np.loadtxt(f'{carpeta}F1_F2_me_for_train.txt')))
    adjs.append(sp.coo_matrix(np.loadtxt(f'{carpeta}F1_F2_proteincomplex_for_train.txt')))
    adjs.append(sp.coo_matrix(np.loadtxt(f'{carpeta}F1_F2_pathway_for_train.txt')))

    pos_edge = np.load(f'{carpeta}pos_edge_binary.npy').astype(np.int32)
    neg_edge = np.load(f'{carpeta}neg_edge_binary.npy').astype(np.int32)
    return pos_edge, neg_edge, adjs

def sparse_to_tuple(sparse_mx): #Convierte una matriz sparse a coordenadas, valores y shape
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()#La pasa a COO si no lo es ya
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data#Coge los valores de la matriz
    shape = sparse_mx.shape
    return coords, values, shape
'''
COO                 vstack
row = [0, 1, 1]     coords =
col = [1, 0, 2]        [[0 1]
data = [5, 3, 1]        [1 0]
                        [1 2]]
'''

def mean_confidence_interval(data, confidence=0.95):
    import scipy as sp
    import scipy.stats
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1 + confidence) / 2., n - 1)
    return m, h


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)#Pone la matriz en formato COO
    adj_ = adj + sp.eye(adj.shape[0])#Los nodos se conectan a si mismos(diagonales)
    rowsum = np.array(adj_.sum(1))#Suma el numero de conexiones de cada nodo para saber el total de conexiones que tiene cada nodo
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())#DUDA
    # adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    adj_normalized = degree_mat_inv_sqrt.dot(adj_)#Normaliza la matriz por la izq
    adj_normalized = adj_normalized.dot(degree_mat_inv_sqrt).tocoo()#Por la derecha
    #Es bueno para evitar que los nodos con muchas conexiones dominen el proceso de aprendizaje, lo mejor para GCNs
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(support, features, adj_orig, placeholders):
    feed_dict = dict()#Crea un diccionario vacio
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})#
    feed_dict.update({placeholders['features']: features})# 
    feed_dict.update({placeholders['adj_orig']: adj_orig})# 
    return feed_dict
