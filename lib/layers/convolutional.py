# this is the man, the myth the legend...

import numpy as np
#from layers.layer import Layer
from layer import Layer

class Convolutional:
    def __init__(self, input_shape: tuple, K: int, F: int, S: int, P: int = 1, **kwargs):
        """Create a convolutional layer.

        Parameters
        ----------
        
            input_shape : tuple
                The volume of the input, W1 x H1 x D1 for width, height, depth, respectively.
            K : int
                The number of filters.
            F : int
                The spacial extent, field size, or height/width of the filter.
            S : int
                The stride of the filters: how many 'pixels' or cells the filters will move  by.
            P : int, optional
                The amount of zero padding added to the input. Convolution reduces shape of the data, by
                including zero padding, this is avoided. Default is 1.
        """
        # initialize data types
        self.dtype = kwargs["dtype"] if "dtype" in kwargs else np.float64
        self.dtype_max_value = np.finfo(self.dtype) 
        #self.sigmoid_clip_value = 0.9 * np.log(self.get_max_dtype(self.dtype))

        # initialize random state 
        self.random_state = kwargs["random_state"] if "random_state" in kwargs else None
        self.rng = np.random.default_rng(seed=self.random_state)

        # initialize matrix values
        weight_init_method = kwargs["weight_init"] if "weight_init" in kwargs else "Xavier"
        #self.weights = self.initialize(weight_init_method, **kwargs)
        #self.biases = self.bias_initialize("Zero")

        # outputs required
        self.deltas: np.ndarray = np.zeros(0, dtype=self.dtype) # initialized in training to fit training data size
        #self.outputs = np.zeros(0, dtype=self.dtype)
        #self.raw_outputs = np.zeros(0, dtype=self.dtype)

        # for adam optimizer
        #self.weight_momenta = np.zeros_like(self.weights, dtype=self.dtype)
        #self.weight_variances = np.zeros_like(self.weights, dtype=self.dtype)
        #self.bias_momenta = np.zeros_like(self.biases, dtype=self.dtype)
        #self.bias_variances = np.zeros_like(self.biases, dtype=self.dtype)

        # activation methods
        self.activation_method = kwargs["activation_method"] if "activation_method" in kwargs else "sigmoid"

        # CONVOLUTIONAL PARAMETERS #
        
        # initialize input matrix shape
        self.input_shape = input_shape
        self.input_depth = input_shape[-1]
        self.kernel_num = K
        self.filter_size = F
        self.stride = S
        self.pad = P

        W2 = (input_shape[0] - F + 2*P)//S + 1
        H2 = (input_shape[1] - F + 2*P)//S + 1
        D2 = K
        self.raw_outputs = np.zeros((W2, H2, D2))
        self.outputs = np.zeros((W2, H2, D2))

        # create K filters of shape FxFxD1
        self.kernels = np.zeros((K, F, F, self.input_depth))
        self.biases = np.zeros(K)

        # TODO: get rid of temp variables and fix inheritance
        self.n_input = 10
        self.n_output = 10
        self.weights = None

    # TODO: make this a staticmethod
    def convolve(self, inputs, filter):
        for i in range(0, self.outputs.shape[0]):
            for j in range(0, self.outputs.shape[1]):
                for k, kernel in enumerate(self.kernels):
                    # slice the input into an F x F kernel shape
                    inputs_slice = inputs[self.stride*i:self.stride*i+self.filter_size, self.stride*j:self.stride*j+self.filter_size, :]
                    output = np.sum(inputs_slice * kernel) + self.biases[k]
                    self.outputs[i, j, k] = output
        return self.outputs
    
    def process(self, inputs: np.ndarray, mask=False):
        # create a tuple of tuples that ensures that only the middle two dimensions of the input are zero padded
        if len(inputs.shape) == 3:
            padding_shape = ((self.pad, self.pad), (self.pad, self.pad), (0, 0))
        else:
            raise ValueError("Input array must be 3 dimensional, for now")
        padded_inputs = np.pad(inputs, padding_shape)
        self.convolve(padded_inputs, self.kernels)
        return self.outputs

myConvArray = np.array([[[2, 1, 0], [1, 0, 1], [2, 0, 0], [2, 0, 2], [1, 0, 0]],
                        [[1, 1, 2], [1, 1, 1], [1, 0, 0], [2, 0, 1], [2, 2, 1]],
                        [[2,2,1], [0,2,1], [0,1,1], [0,2,2], [1,0,1]],
                        [[0,2,2], [0,2,1], [2,2,0], [1,0,0], [1,1,2]],
                        [[0,0,1], [1,2,1], [2,1,0], [2,2,1], [2,0,1]]])

myConvLayer = Convolutional(myConvArray.shape, 2, 3, 2, 1)
myConvLayer.kernels = np.array([[[[-1, 1, 0], [0, -1, 1,], [-1, 0, 0]],
                                 [[-1, 0, 1], [1, 1, -1], [-1, 0, -1]],
                                 [[1, -1, 0], [0, 0, 1], [-1, 1, 1]]],

                                 [[[-1, -1, 1], [0, 1, 0], [0, 0, 1]],
                                 [[0, 0, -1], [1, -1, 1], [1, -1, 1]],
                                 [[0, 0, -1], [0, 1, 0], [0, -1, 0]]]])

myConvLayer.biases[0] = 1
myConvLayer.biases[1] = 0

myConvLayer.biases = myConvLayer.biases .astype(int)

myConvLayer.process(myConvArray)

print(myConvLayer.outputs[:, :, 0])
print(myConvLayer.outputs[:, :, 1])
"""
print(myConvLayer.kernels[0, :, :, 0])
print(myConvLayer.kernels[0, :, :, 1])
print(myConvLayer.kernels[0, :, :, 2])
print("---")
print(myConvLayer.kernels[1, :, :, 0])
print(myConvLayer.kernels[1, :, :, 1])
print(myConvLayer.kernels[1, :, :, 2])
"""