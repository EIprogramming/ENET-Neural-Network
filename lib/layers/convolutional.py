# this is the man, the myth the legend...

import numpy as np
#from layers.layer import Layer
from layer import Layer

class Convolutional(Layer):
    def __init__(self, input_shape: tuple, kernel_params: tuple[int, int, int] | None, P: int = 1, **kwargs):
        """Create a convolutional layer.

        Parameters
        ----------
        
            input_shape : tuple
                The volume of the input, W1 x H1 x D1 for width, height, depth, respectively.
            kernel_params: tuple[int, int, int]
                A tuple of kernel parameters (K, F, S). Where K: int is the number of filters,
                P: int is the spacial extent, field size, or height/width of the filter,
                S: int is the stride of the filters: how many 'pixels' or cells the filters will move by.
            P : int, optional
                The amount of zero padding added to the input. Convolution reduces shape of the data, by
                including zero padding, this is avoided. Default is 1.
        """
        if kernel_params is None: raise ValueError("kernel_params cannot be None.")
        K, F, S = kernel_params
        W2 = (input_shape[0] - F + 2*P)//S + 1
        H2 = (input_shape[1] - F + 2*P)//S + 1
        D2 = K
        super().__init__(sum(input_shape), W2 + H2 + D2, **kwargs) # TODO: fix inheritance
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
        self.output_shape = (W2, H2, D2)
        self.raw_outputs_nd = np.zeros((1, W2, H2, D2)) # full size initialized later, since we are using batches
        self.outputs_nd = np.zeros((1, W2, H2, D2)) # full size initialized later, since we are using batches

        # create K filters of shape FxFxD1
        self.kernels = np.zeros((K, F, F, self.input_depth))
        self.biases = np.zeros(K)

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
    
    def process_nd(self, inputs: np.ndarray, mask=False):
        # create a tuple of tuples that ensures that only the middle two dimensions of the input are zero padded
        if len(inputs.shape) == 3:
            padding_shape = ((self.pad, self.pad), (self.pad, self.pad), (0, 0))
        elif len(inputs.shape) == 4:
            padding_shape = ((0, 0), (self.pad, self.pad), (self.pad, self.pad), (0, 0))
        else:
            raise ValueError("Input array must be 3 or 4 dimensional.")
        padded_inputs = np.pad(inputs, padding_shape)
        batch_shape: tuple = (inputs.shape[0],) + self.output_shape
        # initialize the size of the batch output
        self.outputs_nd = np.zeros(batch_shape)
        self.raw_outputs_nd = self.convolve(padded_inputs)
        self.outputs_nd = self.activate(self.raw_outputs_nd)
        return self.outputs_nd
    
    def process(self, inputs: np.ndarray, mask=False):
        # process a 1D batched input, return a 1D batched input
        batch_size = inputs.shape[0]
        inputs_nd = inputs.reshape((batch_size,) + self.input_shape)
        self.process_nd(inputs)
        self.outputs = self.outputs_nd.reshape((batch_size, -1))
        return self.outputs

