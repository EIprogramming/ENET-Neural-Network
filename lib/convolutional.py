import numpy as np
from layer import Layer

class Convolutional(Layer):
    # initialize data types
    def __init__(self, n_input: int, n_output: int, kernel_shape: tuple, **kwargs):
        # initialize data types
        self.dtype = kwargs["dtype"] if "dtype" in kwargs else np.float64
        self.dtype_max_value = np.finfo(self.dtype) 
        self.sigmoid_clip_value = 0.9 * np.log(self.get_max_dtype(self.dtype))

        # initialize random state 
        self.random_state = kwargs["random_state"] if "random_state" in kwargs else None
        self.rng = np.random.default_rng(seed=self.random_state)

        # initialize matrix shape
        self.n_input = n_input
        self.n_output = n_output
        self.shape = kernel_shape

        # initialize matrix values
        weight_init_method = kwargs["weight_init"] if "weight_init" in kwargs else "Xavier"
        self.weights = self.initialize(weight_init_method, **kwargs)
        self.biases = self.bias_initialize("Zero")

        # outputs required
        self.raw_outputs = np.zeros(n_output, dtype=self.dtype)
        self.outputs = np.zeros(n_output, dtype=self.dtype)
        self.deltas = np.zeros(n_output, dtype=self.dtype)
        self.batch_deltas: np.ndarray = np.zeros(0, dtype=self.dtype) # initialized in training to fit training data size
        self.batch_outputs = np.zeros(0, dtype=self.dtype)
        self.batch_raw_outputs = np.zeros(0, dtype=self.dtype)

        # for adam optimizer
        self.weight_momenta = np.zeros_like(self.weights, dtype=self.dtype)
        self.weight_variances = np.zeros_like(self.weights, dtype=self.dtype)
        self.bias_momenta = np.zeros_like(self.biases, dtype=self.dtype)
        self.bias_variances = np.zeros_like(self.biases, dtype=self.dtype)

        # activation methods
        self.activation_method = kwargs["activation_method"] if "activation_method" in kwargs else "sigmoid"

        # CONVOLUTIONAL LAYER #
        
        self.kernel_shape = kernel_shape