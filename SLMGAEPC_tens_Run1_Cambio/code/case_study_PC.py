from __future__ import division
from __future__ import print_function

import time
import os
import csv
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

from objective import *
from metrics import *
from SLMGAEPC_tens_Run1_Cambio.code.models_origin import *
from utils import *
from res_false import *


os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Set random seed
seed = 8
np.random.seed(seed)
tf.set_random_seed(seed)

# tensorflow config
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('log_file', "log/drop_PC.txt", 'log file name.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 300, 'Number of epochs to train.')
flags.DEFINE_integer('eva_epochs', 25, 'Number of epochs to evaluate')
flags.DEFINE_integer('hidden1', 512, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 256, 'Number of units in hidden layer 2.')
flags.DEFINE_float('dropout', 0.20, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('Alpha', 2.0, 'Coefficient of support view loss.')
flags.DEFINE_float('Coe', 2.0, 'Coefficient of support view loss.')
flags.DEFINE_float('Beta', 4.0, 'Coefficient of final loss.')
flags.DEFINE_integer('early_stopping', 20, 'Tolerance for early stopping (# of epochs).')

log('drop:'+str(FLAGS.dropout))
log('Alpha:'+str(FLAGS.Alpha))
log('Coe:'+str(FLAGS.Coe))
log('Beta:'+str(FLAGS.Beta))

pos_edge, neg_edge, adjs_orig = load_PC_data() #628 - 650 - 85
num_adjs_orig = adjs_orig[0].shape[0]

# Coge los pares con SL y les pone un 1
adj_orig = sp.csr_matrix((np.ones(len(pos_edge)), (pos_edge[:, 0], pos_edge[:, 1])), shape=(num_adjs_orig, num_adjs_orig))#85
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()#85

auc_pair, aupr_pair, f1_pair, train_time = [], [], [], []

tf.reset_default_graph()

row = pos_edge[:, 0]
col = pos_edge[:, 1]
val = np.ones(len(pos_edge))#628 unos
#matriz 85*85 con 1s por SL
adj = sp.csr_matrix((val, (row, col)), shape=(num_adjs_orig, num_adjs_orig))
adj = adj + adj.T#simetria de 628 a 1256 SL
adjs = adjs_orig[: 5]#0 Coex, 1 exclusion mutua, 2 pathway, 3 protComplex, 4 PPI hasta 5 por que si pones 4 es excluyente de una

adjs.append(adj)# adjs 6 matrices 85*85

# load features
features = sparse_to_tuple(adj)#Paso la vista de matriz de SL a tupla 85*85
num_features = features[2][1]#0: pares de genes,1: array con 1 de valores,2: 85*85, coge 85
#pos_edge tiene 628 SL al haber hecho una matriz simetrica, esto será 1256
features_nonzero = features[1].shape[0]

num_nodes = adj.shape[0]#85
num_edges = adj.sum()#1256 que es 628*2

# build training set
x, y = np.triu_indices(num_nodes, k=1)#indices  de filas y columnas de la parte superior de matriz
train_index = set(zip(x, y))#Lo terminos fil-col juntos
train_index = np.array(list(train_index))#Una lista con indices de la parte superior dematriz 85*85
train_edges = pos_edge#Lista de SL con sus indices y el valor(1)

# Some preprocessing
supports = []
for a in adjs:#las 6 matrices
    #Para usar GCN normaliza y crea una tupla para utilzar sparsetensor
    supports.append(preprocess_graph(a))
num_supports = len(supports)

placeholders = {
    'support': [tf.sparse_placeholder(tf.float32, name='adj_{}'.format(_)) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, name='features'),
    'adj_orig': tf.sparse_placeholder(tf.float32, name='adj_orig'),
    'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
}

# Create model
model = SLMGAE_PC(placeholders, num_features, features_nonzero, num_nodes, num_supports - 1,
               name='SLMGAE')

# Create optimizer
with tf.name_scope('optimizer'):
    opt = Optimizer(
        supp=model.support_recs,
        main=model.main_rec,
        preds=model.reconstructions,
        labels=tf.sparse_tensor_to_dense(placeholders['adj_orig'], validate_indices=False),
        num_nodes=num_nodes,
        num_edges=num_edges,
        index=train_index
    )

# Initialize session
sess = tf.Session(config=config)#Empezar la sesion con un uso de GPU dinamico
sess.run(tf.global_variables_initializer())

# m_saver = tf.train.Saver()
# writer = tf.summary.FileWriter('board/SamGAE_{}'.format(train_round), sess.graph)

adj_label = sparse_to_tuple(adj)#Ya es sparse pero se le pasa a un formato tensor

# Construct feed dictionary
feed_dict = construct_feed_dict(supports, features, adj_label, placeholders)

# Train model
eva_score, cost_val, epoch = [], [], 0
tt = time.time()
for epoch in range(FLAGS.epochs):
    t = time.time()

    feed_dict = construct_feed_dict(supports, features, adj_label, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # One update of parameter matrices
    _, avg_cost = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)

    cost_val.append(avg_cost)

    print("Epoch: " + '%04d' % (epoch + 1) +
          " train_loss=" + "{:.5f}".format(avg_cost) +
          " time= " + "{:.5f}".format(time.time() - t))

    # if (epoch + 1) % FLAGS.eva_epochs == 0 and epoch > 5:
    #     feed_dict.update({placeholders['dropout']: 0})
    #     adj_rec = sess.run(model.predict(), feed_dict=feed_dict)
    #     roc, aupr, f1 = evalution(adj_rec, train_edges, test_edges)
    #     eva_score.append([(epoch + 1), roc, aupr, f1])
    #     log("Test by evalution:\n" +
    #         "Train_Epoch = %04d\t" % (epoch + 1) +
    #         'Test ROC score: {:.5f}\t'.format(roc) +
    #         'Test Aupr score: {:.5f}\t'.format(aupr) +
    #         'Test F1 score: {:.5f}\t'.format(f1))

    # if epoch > 199 and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping + 1): -1]):
    #     print('Early stopping...')
    #     break

train_time.append(time.time() - tt)
print('Optimization Finished!')
feed_dict.update({placeholders['dropout']: 0})
adj_rec = sess.run(model.predict(), feed_dict=feed_dict)

num = adj_rec.shape[0]

x, y = np.triu_indices(num, k=1)

c_set = set(zip(x, y)) - set(zip(row, col))

slMapping = {}
with open('../data/List_Proteins_in_SL_panc.txt', 'r') as inf:
    id = 0
    for line in inf:
        slMapping[id] = line.replace('\n', '')
        id += 1

inx = np.array(list(c_set))
prediction = []
for x, y, z in zip(inx[:, 0], inx[:, 1], adj_rec[inx[:, 0], inx[:, 1]]):
    prediction.append([slMapping[x], slMapping[y], z])

prediction.sort(key=lambda x: x[2], reverse=True)
print(prediction[:10])
    
# with open('top.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows()

df = pd.DataFrame(data=prediction[:5000])
df.to_csv('results/Pred.csv', sep='\t', index=False ,header=['gen1','gen2','score'])

'''# -----------------------------
# MATRIZ DE CONFUSIÓN 
# -----------------------------
threshold = 0.5   # puedes cambiarlo según tus métricas

y_true = []
y_pred = []

# POSITIVOS
for i, j, _ in pos_edge:
    score = adj_rec[i, j]
    y_true.append(1)
    y_pred.append(1 if score >= threshold else 0)

predFalse = set()
# NEGATIVOS
for i, j, _ in neg_edge:
    score = adj_rec[i, j]
    y_true.append(0)
    y_pred.append(1 if score >= threshold else 0)
    if score >= threshold:
        predFalse.add((i,j,score)) 

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

predFalse = pd.DataFrame(predFalse, columns=['gen1','gen2','score'])
predFalse.to_csv('results/False.csv', sep='\t', index=False)

pairs_filtered = index_to_gen()
pairs_filtered.to_csv('results/False.csv', sep='\t', index=False)'''

sess.close()