from inits import *
import tensorflow as tf

def dropout_sparse(x, keep_prob, num_nonzero_elems):
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1. / keep_prob)


class GraphConvolution():
    """Basic graph convolution layer for undirected graph without edge labels."""

    def __init__(self, input_dim, output_dim, adj, name, dropout=0., act=tf.nn.relu, norm=False, is_train=False):
        self.name = name
        self.vars = {}
        self.issparse = False
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name='weights')
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.norm = norm
        self.is_train = is_train

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            x = inputs # De la capa anterior (N*hidden1)
            # Como x es denso, usamos el driout normal de ts
            x = tf.nn.dropout(x, 1 - self.dropout)
            # Combierte el (N*hidden1) + (hidden1*hidden2) = (N*hidden2)
            x = tf.matmul(x, self.vars['weights'])
            # Cada nodo mezcla su información con la de sus vecinos
            # (N×N) × (N×hidden2) = (N×hidden2)
            x = tf.sparse_tensor_dense_matmul(self.adj, x)
            # Activacion (ReLU o LeakyReLU)
            outputs = self.act(x)
            # Normalizacion
            if self.norm:
                outputs = tf.layers.batch_normalization(outputs, training=self.is_train)

        return outputs


class GraphConvolutionSparse():
    """Graph convolution layer for sparse inputs."""

    def __init__(self, input_dim, output_dim, adj, features_nonzero, name, dropout=0., act=tf.nn.relu, norm=False, is_train=False):
        self.name = name
        self.vars = {}
        self.issparse = False
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name='weights')
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero
        self.norm = norm
        self.is_train = is_train

    # Creacion de embeddings
    def __call__(self, inputs):
        with tf.name_scope(self.name):
            x = inputs # Matriz de adjacencia sparse

            x = dropout_sparse(x, 1 - self.dropout, self.features_nonzero)
            # cConvierte (N*N) * (N*hidden1) = (N*hidden1)
            # Pasa de tener N caracteristicas a tener hidden1 
            x = tf.sparse_tensor_dense_matmul(x, self.vars['weights'])
            # Cada nodo mezcla su informacion con la de sus vecinos
            # (N*N) * (N*hidden1) = (N*hidden1)
            x = tf.sparse_tensor_dense_matmul(self.adj, x)
            # Activacion (ReLU o LeakyReLU)
            outputs = self.act(x)
            # Normalizar si los valores son muy dispares
            if self.norm:
                outputs = tf.layers.batch_normalization(outputs, training=self.is_train)

        return outputs


class AttentionRec():
    """Attention merge layer for each support view"""

    def __init__(self, output_dim, num_support, name, dropout=0., act=tf.nn.sigmoid):
        self.num_nodes = output_dim     #Numero de genes
        self.num_support = num_support  #Numero de vistas sup
        self.name = name
        self.dropout = dropout
        self.act = act

        # Se guardara cada lista ponderada
        self.attADJ = []

        with tf.variable_scope(self.name + '_attW'):
            # Van a ser 5 de dimensiones N*N y inicualizamos los pesos random de 0.9 a 1.1(parecidos)
            self.attweights = tf.get_variable("attWeights", [self.num_support, self.num_nodes, self.num_nodes],
                                             initializer=tf.random_uniform_initializer(minval=0.9, maxval=1.1))
            # Ahora ponemos que la suma de los pesos sea 1
            self.attention = tf.nn.softmax(self.attweights, 0)


    def __call__(self, recs):
        with tf.name_scope(self.name):
            for i in range(self.num_support):
                # Multiplica la matriz de atencion N*N *  la matriz reconstruida N*N
                self.attADJ.append(tf.multiply(self.attention[i], recs[i]))

            #Suma todas las matrices y saca una única matriz NxN que combina:
            # - Las predicciones de cada vista
            # - Ponderadas por la importancia aprendida para cada par de genes
            confiWeights = tf.add_n(self.attADJ)

            return confiWeights


class InnerProductDecoder():
    """Decoder model layer for link prediction."""

    def __init__(self, output_dim, name, dropout=0., act=tf.nn.sigmoid):
        self.name = name
        self.issparse = False
        self.vars = {}
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(output_dim, output_dim, name='weights')
        self.dropout = dropout
        self.act = act

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            # Dropout
            inputs = tf.nn.dropout(inputs, 1 - self.dropout)
            # (N×hidden2) × (hidden2×hidden2) = (N×hidden2)
            # Es como una "refinamiento" de los embeddings 
            x = tf.matmul(inputs, self.vars['weights'])
            # ============================================================
            # PASO 3: PRODUCTO INTERNO (¡LA CLAVE!)
            # ============================================================
            # Multiplica x por la transpuesta de inputs
            # (N×hidden2) × (hidden2×N) = (N×N)
            # 
            # Esto calcula la SIMILITUD entre cada par de nodos:
            # Para los nodos i y j: x[i] · inputs[j] (producto punto)
            # Si son similares → valor alto → probable conexión
            # Si son diferentes → valor bajo → probable no conexión
            x = tf.matmul(x, tf.transpose(inputs))
            
            outputs = self.act(x)
        return outputs

