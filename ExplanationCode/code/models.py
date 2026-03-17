from layers import *

flags = tf.app.flags
FLAGS = flags.FLAGS

class SLMGAE_PC():
    def __init__(self, placeholders, num_features, features_nonzero, num_nodes, num_supView, name):
        self.name = name
        self.num_nodes = num_nodes                  #Numero de genes
        self.num_supView = num_supView              #Numero de vistas sup
        self.input_dim = num_features               #Numero de genes
        self.features_nonzero = features_nonzero    #Optimizacion sparse
        self.adjs = placeholders['support']         #Lista de supports views
        self.dropout = placeholders['dropout']      #Dropout
        self.inputs = placeholders['features']      #Caracteristicas de entrada
        self.support_recs = []                      #Guardar reconstrucciones

        with tf.variable_scope(self.name):
            # Capa de atencion que conbinara todas las vistas co la importancia de cada una
            self.attentionLayer = AttentionRec(
                name='Attention_Layer',
                output_dim=self.num_nodes,
                num_support=self.num_supView,
                act=lambda x: x)

            self.build()

    def build(self):

        self.hidden = [] # Guardar salidas de la primera capa
        
        #--------------------------------------------
        # Primera capa: GCN Sparse o Encoder
        #--------------------------------------------
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
        
        self.hid = [] # Guardar salidas de la segunda capa

        #--------------------------------------------
        # Segunda capa: GCN Dense o Encoder
        #--------------------------------------------
        for i in range(self.num_supView):

            hid = GraphConvolution(
                name=f'gcn_dense_layer{i+1}',
                input_dim=FLAGS.hidden1,
                output_dim=FLAGS.hidden2,
                adj=self.adjs[i],
                act=tf.nn.leaky_relu,
                dropout=self.dropout)(self.hidden[i])
            
            self.hid.append(hid)

        #--------------------------------------------
        # Decoder: Reconstruir cada vista
        #--------------------------------------------
        for i in range(self.num_supView):

            self.support_recs.append(
                InnerProductDecoder(
                    name='gcn_decoder',
                    output_dim=FLAGS.hidden2,
                    act=lambda x: x)(self.hid[i]))
       
        #--------------------------------------------
        # Capa de atencion: Combinar todas las vistas
        #--------------------------------------------
        self.att = self.attentionLayer(self.support_recs)

        #--------------------------------------------
        # Vita Principal
        #--------------------------------------------
        self.main_rec = InnerProductDecoder(
                name='gcn_decoder',
                output_dim=FLAGS.hidden2,
                act=lambda x: x)(self.hid[-1])
        
        #--------------------------------------------
        # Recontruccion Final
        #--------------------------------------------
        self.reconstructions = tf.add(self.main_rec, tf.multiply(FLAGS.Coe, self.att))

    def predict(self):
        return self.reconstructions
    


# Saber si embeddings estan muy dispares o no.
