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

        pesos_iniciales = [1.0, 1.2, 1.8, 1.7, 0]
        #54.58, 57.52, 69.18, 67.48, 17.1
        with tf.variable_scope(self.name):
            self.attentionLayer = AttentionRec(
                name='Attention_Layer',
                output_dim=self.num_nodes,
                num_support=self.num_supView,
                initial_weights=pesos_iniciales
            )

            self.build()

    def build(self):
        
        self.hidden = []
            
        for i in range(self.num_supView):
            hidden = GraphConvolutionSparse(
                name=f'gcn_sparse_layer{i+1}',
                input_dim=self.input_dim,
                output_dim=FLAGS.hidden1,
                adj=self.adjs[i],                
                features_nonzero=self.features_nonzero,
                act=tf.nn.leaky_relu,
                dropout=self.dropout)(self.inputs)
                
            self.hidden.append(hidden)

        self.hid = []

        for i in range(self.num_supView):

            hid = GraphConvolution(
                name=f'gcn_dense_layer{i+1}',
                input_dim=FLAGS.hidden1,
                output_dim=FLAGS.hidden2,
                adj=self.adjs[i],
                act=tf.nn.leaky_relu,
                dropout=self.dropout)(self.hidden[i])
            
            self.hid.append(hid)

        for i in range(self.num_supView):

            self.support_recs.append(
                InnerProductDecoder(
                    name='gcn_decoder',
                    output_dim=FLAGS.hidden2,
                    act=lambda x: x)(self.hid[i]))
        print(self.attentionLayer(self.support_recs))
        self.att = self.attentionLayer(self.support_recs)

        self.main_rec = InnerProductDecoder(
                name='gcn_decoder',
                output_dim=FLAGS.hidden2,
                act=lambda x: x)(self.hid[-1])

        self.reconstructions = tf.add(self.main_rec, tf.multiply(FLAGS.Coe, self.att))

    def predict(self):
        return self.reconstructions