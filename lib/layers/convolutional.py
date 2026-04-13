# this is the man, the myth the legend...

import numpy as np
#from layer import Layer
from layers.layer import Layer

class Convolutional(Layer):
    def __init__(self, input_shape: tuple | None, kernel_params: tuple[int, int, int, int] | None, **kwargs):
        """Create a convolutional layer.

        Parameters
        ----------
            input_shape : tuple
                The volume of the input, W1 x H1 x D1 for width, height, depth, respectively.
            kernel_params: tuple[int, int, int]
                A tuple of kernel parameters (K, F, S).Where
                K: int is the number of filters,
                F: int is the spacial extent, field size, or height/width of the filter,
                S: int is the stride of the filters: how many 'pixels' or cells the filters will move by.
            P : int, optional
                The amount of zero padding added to the input. Convolution reduces shape of the data, by
                including zero padding, this is avoided. Default is 1.
        """
        # error handling to ensure compatability with Layer and NewLayer
        if kernel_params is None: raise ValueError("kernel_params cannot be None.")
        if input_shape is None: raise ValueError("input_shape cannot be None.")
        K, F, S, P = kernel_params
        W2 = (input_shape[0] - F + 2*P)//S + 1
        H2 = (input_shape[1] - F + 2*P)//S + 1
        D2 = K
        super().__init__(sum(input_shape), W2 + H2 + D2, **kwargs) # TODO: fix inheritance
        
        # CONVOLUTIONAL PARAMETERS #
        
        # initialize input matrix shape
        if len(input_shape) == 2:
            input_shape = input_shape + (1,) 
        self.input_shape = input_shape
        self.input_depth = input_shape[-1]
        self.kernel_num = K
        self.filter_size = F
        self.stride = S
        self.pad = P

        W2 = (input_shape[0] - F + 2*P)//S + 1
        H2 = (input_shape[1] - F + 2*P)//S + 1
        D2 = K
        self.output_shape = (W2, H2, D2)
        self.raw_outputs_nd = np.zeros((1, W2, H2, D2)) # full size initialized later, since we are using batches
        self.raw_outputs = self.raw_outputs_nd.reshape((1,) + (self.raw_outputs_nd.shape))
        self.outputs_nd = np.zeros((1, W2, H2, D2)) # full size initialized later, since we are using batches

        # create K filters of shape FxFxD1
        self.kernels = np.zeros((K, F, F, self.input_depth))
        self.weights = self.kernels.ravel() # provides a flattened view of kernels
        self.deltas = np.zeros_like(self.weights)
        self.biases = np.zeros(K)
        # for adam optimizer
        self.weight_momenta = np.zeros_like(self.weights, dtype=self.dtype)
        self.weight_variances = np.zeros_like(self.weights, dtype=self.dtype)
        self.bias_momenta = np.zeros_like(self.biases, dtype=self.dtype)
        self.bias_variances = np.zeros_like(self.biases, dtype=self.dtype)

        # PARENT (Layer) OVERRIDES #
        self.shape = sum(self.weights.shape)
        self.n_input = np.prod(input_shape)
        self.n_output = W2 * H2 * D2
    
    def initialize_weights(self, init_method):
        self.weights = self.initialize(init_method)
        self.kernels = self.weights.reshape(self.kernels.shape)

    def convolve(self, inputs):
        for i in range(0, self.output_shape[0]):
            for j in range(0, self.output_shape[1]):
                for k, kernel in enumerate(self.kernels):
                    # slice the input into an F x F kernel shape
                    inputs_slice = inputs[:, self.stride*i:self.stride*i+self.filter_size, self.stride*j:self.stride*j+self.filter_size, :]
                    output = np.sum(inputs_slice * kernel, axis=(1, 2, 3)) + self.biases[k]
                    self.outputs_nd[:, i, j, k] = output
                    #print("ijk", i, j, k, "\n output:", output, "\n", inputs_slice, "\n", kernel)
                    #input()
        return self.outputs_nd
    
    def vec_convole(self, ijk):
        ...

    @staticmethod # TODO: make this the usual convolve method
    def convolve_specific(inputs, filters, biases = None, S = 1, P = 1):
        batch_size = inputs.shape[0]
        K = filters.shape[1]
        F = filters.shape[2]
        W2 = (inputs.shape[1] - F + 2*P)//S + 1
        H2 = (inputs.shape[2] - F + 2*P)//S + 1
        D2 = K

        # 14 - 28
        #print("WHD:", W2, H2, D2, "KFPS: ", K, F, P, S)

        # set up biases for None
        if biases is None:
            biases = np.zeros(K)
        output_shape = (W2, H2, D2)
        if len(inputs.shape) == 4:
            padding_shape = ((0, 0), (P, P), (P, P), (0, 0))
        elif len(inputs.shape) == 3:
            padding_shape = ((P, P), (P, P), (0, 0))
        else:
            raise ValueError("Inputs must be either 4 dimensional or 3 dimensional.")
        padded_inputs = np.pad(inputs, padding_shape)
        outputs_nd = np.zeros((batch_size, W2, H2, D2))
        for batch in range(batch_size):
            kernels = filters[batch]
            for i in range(0, output_shape[0]):
                for j in range(0, output_shape[1]):
                    for k, kernel in enumerate(kernels):
                        # slice the input into an F x F kernel shape
                        inputs_slice = padded_inputs[:, S*i:S*i+F, S*j:S*j+F, :]
                        outputs_nd[:, i, j, k] = np.sum(inputs_slice * kernel) + biases[k]
        return outputs_nd
    
    def process_nd(self, inputs: np.ndarray, mask=False):
        # create a tuple of tuples that ensures that only the middle two dimensions of the input are zero padded
        if len(inputs.shape) == 4:
            padding_shape = ((0, 0), (self.pad, self.pad), (self.pad, self.pad), (0, 0))
        else:
            raise ValueError(f"Input array must be 4 dimensional, got shape: {inputs.shape}")
        padded_inputs = np.pad(inputs, padding_shape)
        batch_shape: tuple = (inputs.shape[0],) + self.output_shape
        # initialize the size of the batch output
        self.outputs_nd = np.zeros(batch_shape)
        self.raw_outputs_nd : np.ndarray = self.convolve(padded_inputs)
        self.raw_outputs = self.raw_outputs_nd.reshape(inputs.shape[0], np.prod(self.raw_outputs_nd.shape[1:]))
        self.outputs_nd = self.activate(self.raw_outputs_nd)
        self.outputs = self.outputs_nd.reshape(inputs.shape[0], np.prod(self.raw_outputs_nd.shape[1:]))
        return self.outputs_nd
    
    def process(self, inputs: np.ndarray, mask=False):
        # process a 1D batched input, return a 1D batched input
        batch_size = inputs.shape[0]
        if len(self.input_shape) == 2: # if its 2D, add an extra empty dimension
            inputs_nd = inputs.reshape((batch_size,) + self.input_shape + (1,))
        else:
            inputs_nd = inputs.reshape((batch_size,) + self.input_shape)
        self.process_nd(inputs_nd)
        return self.outputs

