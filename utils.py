import numpy as np
from keras import backend as K
from keras.engine.topology import Layer

class pruned_Dense(Layer):
    def __init__(self, n_neurons_out, **kwargs):
        self.n_neurons_out = n_neurons_out
        super(pruned_Dense,self).__init__(**kwargs)

    def build(self, input_shape):
        #define the variables of this layer in the build function:
        n_neurons_in = input_shape[1]
        # print(n_neurons_in)
        # print(self.n_neurons_out)
        stdv = 1/np.sqrt(n_neurons_in)
        w = np.random.normal(size=[n_neurons_in, self.n_neurons_out], loc=0.0, scale=stdv).astype(np.float32)
        self.w = K.variable(w)
        b = np.zeros(self.n_neurons_out)
        self.b = K.variable(b)
        # w is the weight matrix, b is the bias. These are the trainable variables of this layer.
        self.trainable_weights = [self.w, self.b]
        # mask is a non-trainable weight that simulates pruning. the values of mask should be either 1 or 0, where 0 will prune a weight. We initialize mask to all ones:
        mask = np.ones((n_neurons_in, self.n_neurons_out))
        self.mask = K.variable(mask)

    def call(self, x):
        # define the input-output relationship in this layer in this function
        pruned_w = self.w * self.mask
        out = K.dot(x, pruned_w)
        out = out + self.b
        return out

    def compute_output_shape(self, input_shape):
        #define the shape of this layer's output:
        return (input_shape[0], self.n_neurons_out)

    def get_mask(self):
        #get the mask values
        return K.get_value(self.mask)

    def set_mask(self, mask):
        #set new mask values to this layer
        K.set_value(self.mask, mask)

