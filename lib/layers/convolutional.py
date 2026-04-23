# this is the man, the myth the legend...

import numpy as np
#from layer import Layer
from layers.layer import Layer

class Convolutional(Layer):
    def __init__(self, input_shape: tuple, kernel_params: tuple[int, int, int, int], **kwargs):
        """Create a convolutional layer.

        Parameters
        ----------
            input_shape : tuple
                The volume of the input, W1 x H1 x D1 for width, height, depth, respectively.
            kernel_params: tuple[int, int, int, int]
                A tuple of kernel parameters (K, F, S, P).Where
                K: int is the number of filters,
                F: int is the spacial extent, field size, or height/width of the filter,
                S: int is the stride of the filters: how many 'pixels' or cells the filters will move by.
                P: int is the size of zero padding added to the input. Default is 1 (to avoid dimension reduction).
        """
        # error handling to ensure compatability with Layer and NewLayer
        self._init_kernel_parameters(*kernel_params)
        self._init_outputs(kernel_params, input_shape)
        self._init_super(input_shape, self.output_shape, **kwargs)

        self._init_input_parameters(input_shape)
        self._init_neural_parameters()
        self._init_adam_parameters()

    @staticmethod
    def get_output_shape(K: int, F: int, S: int, P: int, W1: int, H1: int):
        """Get the shape of the convolution output.

        Parameters
        ----------
            K : int
                The number of filters.
            F : int
                The spacial extent (height and width) of the filter.
            S : int
                The stride of the filter.
            P : int
                The size of zero padding.
            W1 : int
                The width of the input.
            H1 : int
                The height of the input.
        """
        W2 = (W1 - F + 2*P)//S + 1
        H2 = (H1 - F + 2*P)//S + 1
        D2 = K

        return (W2, H2, D2)

    def _init_kernel_parameters(self, K: int, F: int, S: int, P: int):
        self.kernel_num = K
        self.filter_size = F
        self.stride = S
        self.padding = P

    def _init_neural_parameters(self):
        K = self.kernel_num
        F = self.filter_size
        D1 = self.input_depth

        # create K filters of shape FxFxD1
        self.kernels = np.zeros((K, F, F, D1))
        self.weights = self.kernels.ravel() # provides a flattened view of kernels
        self.deltas = np.zeros_like(self.weights)
        self.biases = np.zeros(K)
    
    def _init_adam_parameters(self):
        self.weight_momenta = np.zeros_like(self.weights, dtype=self.dtype)
        self.weight_variances = np.zeros_like(self.weights, dtype=self.dtype)
        self.bias_momenta = np.zeros_like(self.biases, dtype=self.dtype)
        self.bias_variances = np.zeros_like(self.biases, dtype=self.dtype)

    def _init_input_parameters(self, input_shape: tuple[int, int, int]):
        self.input_shape = input_shape
        self.input_depth = input_shape[-1]

    def _init_outputs(self, kernel_params: tuple[int, int, int, int], input_shape: tuple[int, int, int]):
        K, F, S, P = kernel_params
        W1, H1 = input_shape[0], input_shape[1]
        self.output_shape = self.get_output_shape(K, F, S, P, W1, H1)
        self.raw_outputs = np.zeros(0) # full size initialized later, since we are using batches
        self.outputs = np.zeros(0) # full size initialized later, since we are using batches

    def _init_super(self, input_shape: tuple[int, int, int], output_shape: tuple[int, int, int], **kwargs):
        input_shape_flattened = sum(input_shape)
        output_shape_flattened = sum(output_shape)
        super().__init__(input_shape_flattened, output_shape_flattened, **kwargs) # TODO: fix inheritance

    def initialize_weights(self, init_method: str):
        # TODO: incorporate into __init__
        self.weights = self.initialize(init_method)
        self.kernels = self.weights.reshape(self.kernels.shape)
    
    @staticmethod
    def einsum_convolve3D(slices: np.ndarray, kernels: np.ndarray) -> np.ndarray:
        """ Perform a 3D convolution using `np.einsum`.

        Performs a convolution on a window slice broadcasted
        over each batch, each window per row and each window per column,
        and per kernel.

        Parameters
        ----------
            slices : np.ndarray, shape (n, i, j, d, h, w)
                An array of the window-view slices of the inputs.
                Shape dimensions:

                * `n` : number of samples (batch size)
                * `i` : the number of windows per row of the inputs
                * `j` : the number of windows per column of the inputs
                * `d` : depth of the window
                * `h` : height of the window
                * `w` : width of the window
            kernels : np.ndarray, shape (k, w, h, d)
           
                Shape dimensions:

                * `k` : number of kernels
                * `h` : height of the kernel (same as window)
                * `w` : width of the kernel (same as window)
                * `d` : depth of the window (same as window)
        
        Returns
        -------
        convolution : np.ndarray, shape (n, i, j, k)
            The batched convolution of the slices with the kernels.
        """
        
        return np.einsum('nijdwh, ...kwhd -> nijk...', slices, kernels, optimize=True)

    @staticmethod
    def convolve3D(inputs, filters: np.ndarray, biases: np.ndarray | None = None, stride: int = 1):
        axes = (1, 2) # apply the slice over the 1st and 2nd axes
        filter_size = filters.shape[2]

        slices = np.lib.stride_tricks.sliding_window_view(inputs, (filter_size, filter_size), axis=axes) # type: ignore
        
        # apply the stride
        slices = slices[:, ::stride, ::stride, :, :, :]
    
        output = Convolutional.einsum_convolve3D(slices, filters)

        # apply biases
        if biases is None:
            biases = np.zeros(0)
        output += biases

        return output

    @staticmethod
    def pad(inputs: np.ndarray, padding: int):
        # TODO: make this n-dimensional
        # create a tuple of tuples that ensures that only the middle two dimensions of the input are zero padded
        if len(inputs.shape) == 4:
            padding_shape = ((0, 0), (padding, padding), (padding, padding), (0, 0))
        else:
            raise ValueError(f"Input array must be 4 dimensional, got shape: {inputs.shape}")
        return np.pad(inputs, padding_shape)

    def process(self, inputs: np.ndarray):
        padded_inputs = Convolutional.pad(inputs, self.padding)
        # initialize the size of the batch output
        self.raw_outputs : np.ndarray = Convolutional.convolve3D(padded_inputs, self.kernels, self.biases, self.stride)
        self.outputs = self.activate(self.raw_outputs)
        return self.outputs
    

