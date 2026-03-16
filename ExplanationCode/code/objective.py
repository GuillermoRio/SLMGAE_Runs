from layers import *

flags = tf.app.flags
FLAGS = flags.FLAGS

# Se encargara de la perdida y optimizacion del modelo
class Optimizer():
    #Constructor
    # supp: Lista de reconstrucciones de las vistas soportes
    # main: Reconstruccion de la vista inicial
    # preds: Prediccion final combinada
    # labels: matriz original
    # index: indices de los pares de nodos a usar en el entrenamiento

    def __init__(self, supp, main, preds, labels, num_nodes, num_edges, index):
        # Obtiene todas las operaciones de actualizacion
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        #Comprueba que se ejecutan antes de continuar
        with tf.control_dependencies(update_ops):

            #Selecciona los elementos del primero segun el index que se le pasa
            labels_sub = tf.gather_nd(labels, index)
            main_sub = tf.gather_nd(main, index)
            preds_sub = tf.gather_nd(preds, index)

            # Inicializa la perdida combinada
            self.loss_supp = 0
            
            for viewRec in supp:
                # Lo mismo de antes, selecciona para cada supView los mismo indices
                viewRec_sub = tf.gather_nd(viewRec, index)
                #Calcula el error cuadratico medio (MSE) entre supView y etiquetasReales
                self.loss_supp += tf.compat.v1.keras.losses.MSE(labels_sub, viewRec_sub)

            # MSE para la vista principal
            self.loss_main = tf.compat.v1.keras.losses.MSE(labels_sub, main_sub)

            # MSE para la vista de prediccion
            self.loss_preds = tf.compat.v1.keras.losses.MSE(labels_sub, preds_sub)

            # Alpha * perdida de vistas soporte + beta * perdida de vista predicha
            self.cost = FLAGS.Alpha * self.loss_supp + \
                        FLAGS.Beta * self.loss_preds + \
                        1 * self.loss_main

            # Crea el optimizador de Adam con la tasa de aprendizaje definida en FLAGs
            self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer
            # Minimiza el costo
            self.opt_op = self.optimizer.minimize(self.cost)
            
            # Calcula los gradientes de todas las variables con respecto al costo
            self.grads_vars = self.optimizer.compute_gradients(self.cost)

