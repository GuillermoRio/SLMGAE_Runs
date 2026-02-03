from layers import *

flags = tf.app.flags
FLAGS = flags.FLAGS

class SLMGAE_PC():
    def __init__(self, placeholders, num_features, features_nonzero, num_nodes, num_supView, name):
        self.name = name
        self.num_nodes = num_nodes
        self.num_supView = num_supView
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.adjs = placeholders['support']
        self.dropout = placeholders['dropout']
        self.inputs = placeholders['features']
        self.support_recs = []

        with tf.variable_scope(self.name):
            self.attentionLayer = AttentionRec(
                name='Attention_Layer',
                output_dim=self.num_nodes,
                num_support=self.num_supView,
                act=lambda x: x)
            self.build()

    def build(self):
        # Capas para vistas de soporte (0-4)
        support_hidden = []
        support_embeddings = []
        
        for i in range(self.num_supView):  # 0 a 4
            # Capa 1 específica por vista
            hidden1 = GraphConvolutionSparse(
                name=f'gcn_sparse_support_{i}_layer1',
                input_dim=self.input_dim,
                output_dim=FLAGS.hidden1,
                adj=self.adjs[i],  # Vista de soporte i
                features_nonzero=self.features_nonzero,
                act=tf.nn.leaky_relu,
                dropout=self.dropout)(self.inputs)
            
            # Capa 2 específica por vista
            hidden2 = GraphConvolution(
                name=f'gcn_dense_support_{i}_layer2', 
                input_dim=FLAGS.hidden1,
                output_dim=FLAGS.hidden2,
                adj=self.adjs[i],  # Misma vista
                act=tf.nn.leaky_relu,
                dropout=self.dropout)(hidden1)
            
            support_embeddings.append(hidden2)
            
            # Decoder para esta vista
            rec = InnerProductDecoder(
                name=f'support_decoder_{i}',
                output_dim=FLAGS.hidden2,
                act=lambda x: x)(hidden2)
            
            self.support_recs.append(rec)

        # Capa principal (vista 5) - SEPARADA
        self.main_hidden1 = GraphConvolutionSparse(
            name='gcn_sparse_main_layer1',
            input_dim=self.input_dim,
            output_dim=FLAGS.hidden1,
            adj=self.adjs[self.num_supView],  # Última vista = principal
            features_nonzero=self.features_nonzero,
            act=tf.nn.leaky_relu,
            dropout=self.dropout)(self.inputs)
        
        self.main_hidden2 = GraphConvolution(
            name='gcn_dense_main_layer2',
            input_dim=FLAGS.hidden1,
            output_dim=FLAGS.hidden2, 
            adj=self.adjs[self.num_supView],  # Misma vista principal
            act=tf.nn.leaky_relu,
            dropout=self.dropout)(self.main_hidden1)

        # Attention sobre reconstrucciones de soporte
        self.att = self.attentionLayer(self.support_recs)

        # Reconstrucción principal
        self.main_rec = InnerProductDecoder(
            name='main_decoder',
            output_dim=FLAGS.hidden2,
            act=lambda x: x)(self.main_hidden2)

        # Combinación final
        self.reconstructions = tf.add(self.main_rec, tf.multiply(FLAGS.Coe, self.att))

    def predict(self):
        return self.reconstructions