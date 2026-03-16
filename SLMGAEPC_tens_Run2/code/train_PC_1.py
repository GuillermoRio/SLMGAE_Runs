from __future__ import division
from __future__ import print_function

import time
import os
import pandas as pd
import csv

from sklearn.model_selection import KFold

from objective import *
from metrics import *
from models import *
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# N de las shapes
shapeViews = 330
# Nombre carpetas donde coger los  datos
carpetaInput = '../PC_data_1/'
carpetaOutput = '../resultados_1/3/'

nombres_vistas = [
    "PPI",           # Vista 0 - Interacciones proteína-proteína
    "Co-expression", # Vista 1 - Co-expresión
    "ME",            # Vista 2 - (Mutual Exclusivity?)
    "ProteinComplex",# Vista 3 - Complejos proteicos
    "Pathway"        # Vista 4 - Rutas metabólicas
]

# tensorflow config
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('log_file', f"{carpetaOutput}/log/SLMGAE_PC.txt", 'log file name.')
flags.DEFINE_float('learning_rate', 0.0008, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 250, 'Number of epochs to train.')
flags.DEFINE_integer('eva_epochs', 100, 'Number of epochs to evaluate')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 64, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('nn_size', 50, 'Number of K for the KNN')
flags.DEFINE_float('dropout', 0.3, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('Alpha', 0.6, 'Coefficient of support view loss.')
flags.DEFINE_float('Coe', 1.2, 'Coefficient of support view loss.')
flags.DEFINE_float('Beta', 2.0, 'Coefficient of final loss.')
flags.DEFINE_integer('early_stopping', 20, 'Tolerance for early stopping (# of epochs).')


pos_edge, neg_edge, adjs_orig = load_PC_data(carpetaInput)#Matriz principal y las supports

np.random.shuffle(pos_edge)
np.random.shuffle(neg_edge)

# Store original adjacency matrix (without diagonal entries) for later
adj_orig = sp.csr_matrix((np.ones(len(pos_edge)), (pos_edge[:, 0], pos_edge[:, 1])), shape=(shapeViews, shapeViews))
# Eliminar conexiones con uno mismo
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()

# K-Fold Test (Test 20%, Train 80%)
k_round = 0
kf = KFold(n_splits=5, shuffle=True, random_state=123)
pos_edge_kf = kf.split(pos_edge)
neg_edge_kf = kf.split(neg_edge)

# AUC Probabilidad de que un positivo aleatorio tenga mas score que un neg aleatorio
# AUPR 
# Score entre precision y recall
auc_pair, aupr_pair, f1_pair, train_time = [], [], [], []
training_loss, testing_loss = [], []


# Ver lo losses de cada matriz en sus epochs

csv_filename = f'{carpetaOutput}entrenamientos/perdidas_folds.csv'
with open(csv_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    cabecera = ['epoch', 'costo_total', 'main_loss', 'preds_loss']
    # Añade una columna por cada vista
    for i, nombre in enumerate(nombres_vistas):
        cabecera.append(f'vista_{i}_{nombre}')
    writer.writerow(cabecera)

name = 1

for train_pos, test_pos in pos_edge_kf:
    _, test_neg = next(neg_edge_kf)

    tf.reset_default_graph()
    k_round += 1
    print("Training in the %02d fold..." % k_round)

    # Matriz de adjacencia (Todos los que si tienen SL)
    row = pos_edge[train_pos, 0]
    col = pos_edge[train_pos, 1]
    val = np.ones(len(train_pos))
    adj = sp.csr_matrix((val, (row, col)), shape=(shapeViews, shapeViews))
    adj = adj + adj.T #Matriz principal

    adjs = adjs_orig[0: 5] # Supports
    adjs.append(adj)

    num_nodes = adj.shape[0]
    num_edges = adj.sum() #Dos por arista por la simetría

    # build test set
    # Aristas positivas
    test_edges = pos_edge[test_pos]
    # Tantas Aristas negativas como positivas haya
    neg_test_edges = neg_edge[test_neg[:len(test_pos)]]
    test_set = np.vstack((test_edges, neg_test_edges))
    '''
    test_edge = [  4  12   1]
                [ 85 165   1]
                [  0 205   1]
    neg_test_edge = [266 299   0]
                    [193 267   0]
                    [ 34 306   0]
    test_set[:,0] = [4 85 0 266 193 34]
    test_set[:,1] = [12 165 205 299 267 306]
    test_set[:,2] = [1 1 1 0 0 0]
    test_set = [[  4  12   1]
                [ 85 165   1]
                [  0 205   1]
                [266 299   0]
                [193 267   0]
                [ 34 306   0]]
    '''

    # Indices de la matriz(Sin diagonal)
    x, y = np.triu_indices(num_nodes, k=1)
    test_x, test_y = test_set[:, 0], test_set[:, 1]
    # Todos los pares posibles - los del test
    train_index = set(zip(x, y)) - set(zip(test_x, test_y))
    # Train pares posibles - los que ya sabemos que son SL del train quedadonos con los negativos a predecir.
    train_neg_index = train_index - set(zip(row, col))
    # Pasamos a lista todos los pares que no se saben del test
    train_index = np.array(list(train_index))
    # Pasamos a lista todos los pares que no sabemos de SL
    train_neg_index = np.array(list(train_neg_index))

    # build features, pasar las vistas a tuple
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
    model = SLMGAE_PC(placeholders, num_features, features_nonzero, num_nodes, num_supports - 1,
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

        tensores_a_ejecutar = [
        opt.opt_op,           # Operación de optimización
        opt.cost,             # Costo total
        opt.loss_main,        # Pérdida vista principal
        opt.loss_preds,       # Pérdida predicción final
        *opt.loss_supp_individual  # TODAS las pérdidas individuales (el * desempaqueta la lista)
        ]
    
        # Ejecutamos y recibimos todos los valores
        resultados = sess.run(tensores_a_ejecutar, feed_dict=feed_dict)
        
        # Desempaquetamos los resultados
        # resultados[0] = opt_op (no nos interesa el valor)
        # resultados[1] = cost
        # resultados[2] = loss_main
        # resultados[3] = loss_preds
        # resultados[4] en adelante = loss de cada vista de soporte
        
        avg_cost = resultados[1]
        loss_main_val = resultados[2]
        loss_preds_val = resultados[3]
        losses_supp_val = resultados[4:]  # Todas las pérdidas de soporte

        cost_val.append(avg_cost)
        
        # DENTRO del bucle de entrenamiento, después de calcular las pérdidas:
        if (epoch + 1) % 10 == 0:
            print(f"\nEpoch: {epoch + 1:04d}")
            print(f"  Costo total: {avg_cost:.5f}")
            print(f"  Main loss: {loss_main_val:.5f}")
            print(f"  Preds loss: {loss_preds_val:.5f}")
            print("  Vistas de soporte:")
            for i, loss_val in enumerate(losses_supp_val):
                print(f"    Vista {i}: {loss_val:.5f}")
            print(f"  Tiempo: {time.time() - t:.5f}")
            
            # GUARDAR EN CSV
            with open(csv_filename, 'a', newline='') as f:
                writer = csv.writer(f)
                fila = [epoch + 1, avg_cost, loss_main_val, loss_preds_val] + losses_supp_val
                writer.writerow(fila)
    
    
    # Ahora sacaremos el % de las losses
    log(f"PORCENTAJES DE MEJORA - FOLD {k_round}\n")
    df_completo = pd.read_csv(csv_filename)

    filas_por_fold = FLAGS.epochs/10
    inicio = int((k_round - 1) * filas_por_fold)
    fin = int(inicio + filas_por_fold)

    print(filas_por_fold, inicio, fin)
    df_fold = df_completo.iloc[inicio:fin]
    print(f"  Filas en este fold: {len(df_fold)} (de {inicio} a {fin-1})\n")
    # Main upgraded
    main_inicial = df_fold['main_loss'].iloc[0]
    main_final = df_fold['main_loss'].iloc[-1]
    main_mejora = (main_inicial - main_final) / main_inicial * 100
    log(f"\tMain upgraded: {main_mejora:.1f}%")

    # Preds upgraded
    preds_inicial = df_fold['preds_loss'].iloc[0]
    preds_final = df_fold['preds_loss'].iloc[-1]
    preds_mejora = (preds_inicial - preds_final) / preds_inicial * 100
    log(f"\tPreds upgraded: {preds_mejora:.1f}%")

    # Cada vista
    for i, nombre in enumerate(nombres_vistas):
        columna = f'vista_{i}_{nombre}'
        vista_inicial = df_fold[columna].iloc[0]
        vista_final = df_fold[columna].iloc[-1]
        vista_mejora = (vista_inicial - vista_final) / vista_inicial * 100
        log(f"\t{nombre}: {vista_mejora:.1f}%")

    # Metricas de evaluacion
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
    with open('../PC_data_1/gene_list.txt', 'r') as inf:
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



