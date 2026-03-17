from __future__ import division
from __future__ import print_function

import time
import os
import pandas as pd
import random
from sklearn.model_selection import KFold

from objective import *
from metrics import *
from models import *
from utils import *

# Las GPUs no son deterministas por lo que habria cambios en los results
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Set random seed con maxima reproducibilidad
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)


# N de las shapes
shapeViews = 85
supportViews = 5

# Nombre carpetas donde coger los  datos 
# Mia screening
carpetaInput = '../data/'
carpetaOutput = '../resultados/1/'

# tensorflow config
config = tf.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1
)
config.gpu_options.allow_growth = True

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('log_file', f"{carpetaOutput}/log/SLMGAE_PC.txt", 'log file name.')
flags.DEFINE_float('learning_rate', 0.0005, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 150, 'Number of epochs to train.')
flags.DEFINE_integer('eva_epochs', 100, 'Number of epochs to evaluate')
flags.DEFINE_integer('hidden1', 256, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 128, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('nn_size', 20, 'Number of K for the KNN')
flags.DEFINE_float('dropout', 0.4, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('Alpha', 0.5, 'Coefficient of support view loss.')
flags.DEFINE_float('Coe', 1.0, 'Coefficient of support view loss.')
flags.DEFINE_float('Beta', 2.0, 'Coefficient of final loss.')
flags.DEFINE_integer('early_stopping', 15, 'Tolerance for early stopping (# of epochs).')


pos_edge, neg_edge, adjs_orig = load_PC_data(carpetaInput, supportViews)#Matriz principal y las supports

# Mezclar aleatoriamente, para que los folds de validaciÓn cruza sean representativos
np.random.shuffle(pos_edge)
np.random.shuffle(neg_edge)

# Matriz n * n dispersa poniendo 1 como SL 
adj_orig = sp.csr_matrix((np.ones(len(pos_edge)), (pos_edge[:, 0], pos_edge[:, 1])), shape=(shapeViews, shapeViews))
# Eliminar conexiones con uno mismo y eliminación de ceros para hacerla sparse de verdad
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()

# K-Fold Test (Test 20%, Train 80%)
k_round = 0
kf = KFold(n_splits=5, shuffle=True, random_state=seed)
pos_edge_kf = kf.split(pos_edge)
# Misma lenth con los negativos que con los positivos
neg_edge = neg_edge[:len(pos_edge)]
neg_edge_kf = kf.split(neg_edge)

# Listas para guardar metricas
auc_pair, aupr_pair, f1_pair, train_time = [], [], [], []
training_loss, testing_loss = [], []

name = 1
for train_pos, test_pos in pos_edge_kf:
    # Obtener los indices de prueba para las aristas negativas
    _, test_neg = next(neg_edge_kf)

    # Resetear el grafo de ts para cada fold, evita que interfieran variables anteriores
    tf.reset_default_graph()
    k_round += 1
    print("Training in the %02d fold..." % k_round)

    # Extraer aristas positivas: row genes origen o 0 y col genes destino o 1
    row = pos_edge[train_pos, 0]
    col = pos_edge[train_pos, 1]
    # Siempre 1s
    val = np.ones(len(train_pos))
    # Matriz dispersa y simetrica con solo 1s
    adj = sp.csr_matrix((val, (row, col)), shape=(shapeViews, shapeViews))
    adj = adj + adj.T

    # Supports del 0 al 4, la 5 no la coge y mete despues la principal
    adjs = adjs_orig[0: supportViews] 
    adjs.append(adj)

    # Numero de genes, numero de aristas (SL)
    num_nodes = adj.shape[0]
    num_edges = adj.sum()

    # Aristas de test
    test_edges = pos_edge[test_pos]
    # Tantas Aristas test negativas como positivas haya
    neg_test_edges = neg_edge[test_neg[:len(test_pos)]]
    # Combinar los dos 
    test_set = np.vstack((test_edges, neg_test_edges))

    '''
    test_edge = [  4  12   1]
                [ 85 165   1]
                [  0 205   1]

    neg_test_edge = [266 299   0]
                    [193 267   0]
                    [ 34 306   0]
    
    test_set = [[  4  12   1]
                [ 85 165   1]
                [  0 205   1]
                [266 299   0]
                [193 267   0]
                [ 34 306   0]]

    test_set[:,0] = [4 85 0 266 193 34]
    test_set[:,1] = [12 165 205 299 267 306]
    test_set[:,2] = [1 1 1 0 0 0]

    '''

    # Todos los indices de la matriz(k=1, Sin diagonal)
    x, y = np.triu_indices(num_nodes, k=1)

    # Coger todos los indices que tenemos en el test
    test_x, test_y = test_set[:, 0], test_set[:, 1]

    # Todos los pares posibles - los del test = train
    train_index = set(zip(x, y)) - set(zip(test_x, test_y))

    # Train - los que son SL = negativos para entrenar
    train_neg_index = train_index - set(zip(row, col))

    # Convertirlos array np para usar con ts
    train_index = np.array(list(train_index))
    train_neg_index = np.array(list(train_neg_index))

    # Pasamos vista principal a tuplas
    features = sparse_to_tuple(adj)
    num_features = features[2][1]
    # Ver cuantas variables hay que no son 0
    features_nonzero = features[1].shape[0]

    supports = []
    
    for a in adjs:
        supports.append(preprocess_graph(a))
    num_supports = len(supports)

    placeholders = {
            'support': [tf.sparse_placeholder(tf.float32, name='adj_{}'.format(_)) for _ in range(num_supports)],
            'features': tf.sparse_placeholder(tf.float32, name='features'),
            'adj_orig': tf.sparse_placeholder(tf.float32, name='adj_orig'),
            'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
        }

    # Create model
    model = SLMGAE_PC(placeholders, num_features, features_nonzero, num_nodes, num_supports,
                   name=f'SLMGAE_{k_round}')

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
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    adj_label = sparse_to_tuple(adj_orig)

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

    train_time.append(time.time() - tt)
    print('Optimization Finished!')
    feed_dict.update({placeholders['dropout']: 0})
    adj_rec = sess.run(model.predict(), feed_dict=feed_dict)

    # roc, aupr, f1 = evalution_cmf(adj_rec, test_set)
    roc, aupr, f1 = evalution_bal(adj_rec, test_edges, neg_test_edges)
    eva_score.append([(epoch + 1), roc, aupr, f1, train_time[-1]])

    print("Test by Evalution:")
    for e in eva_score:
        print('Train_Epoch = %04d,'
              ' Test ROC score: %5f,'
              ' Test AUPR score:%5f,'
              ' Test F1 score: %5f' % (e[0], e[1], e[2], e[3]))

    auc_pair.append(eva_score[-1][1])
    aupr_pair.append(eva_score[-1][2])
    f1_pair.append(eva_score[-1][3])
    
    #Print y CSV
    num = adj_rec.shape[0]
    x, y = np.triu_indices(num, k=1)

    c_set = set(zip(x, y)) - set(zip(row, col))
    slMapping = {}
    with open(f'{carpetaInput}gene_list.txt', 'r') as inf:
        id = 0
        for line in inf:
            slMapping[id] = line.replace('\n', '')
            id += 1

    inx = np.array(list(c_set))
    prediction = []
    for x, y, z in zip(inx[:, 0], inx[:, 1], adj_rec[inx[:, 0], inx[:, 1]]):
        prediction.append([slMapping[x], slMapping[y], z])

    prediction.sort(key=lambda x: x[2], reverse=True)

    df = pd.DataFrame(data=prediction[:5000])
    df.to_csv(f'{carpetaOutput}Top_nonPred{name}.csv')
    name = name + 1

    m1, sdv1 = mean_confidence_interval(auc_pair)
    m2, sdv2 = mean_confidence_interval(aupr_pair)
    m3, sdv3 = mean_confidence_interval(f1_pair)
    m4, sdv4 = mean_confidence_interval(train_time)

    log("Average metrics over pairs:\n" +
        " time_mean:%.5f, time_sdv:%.5f\n" % (m4, sdv4) +
        " auc_mean:%.5f, auc_sdv:%.5f\n" % (m1, sdv1) +
        " aupr_mean:%.5f, aupr_sdv:%.5f\n" % (m2, sdv2) +
        " f1_mean: %.5f, f1_sdv: %.5f\n" % (m3, sdv3))

#print(adj_rec[test_set[:, 0], tesSt_set[:, 1]])