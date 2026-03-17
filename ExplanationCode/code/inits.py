import numpy as np
import tensorflow as tf

# Se utiliza Glorot para inicializar los pesos de una forma la cual
# no sean muy grandes y no se aprenda
# no sean muy pequeños y no se aprenda igualmente
# init_range = √(6 / (256 + 128)) = √(6 / 384) = √(0.0156) = 0.125
def weight_variable_glorot(input_dim, output_dim, name=""):
    # Calcula el rango de inicializacion basado en las dimensiones
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    
    # Crea valores aleatorios uniformes dentro de [-init_range, init_range]
    initial = tf.random_uniform(
        [input_dim, output_dim], minval=-init_range,
        maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)
