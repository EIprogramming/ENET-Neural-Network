import numpy as np
from layers.layer import Layer

class Dense(Layer):
    def __init__(self, n_input: int, n_output: int, **kwargs):
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
        self.shape = (n_output, n_input)

        # initialize matrix values
        weight_init_method = kwargs["weight_init"] if "weight_init" in kwargs else "Xavier"
        self.weights = self.initialize(weight_init_method, **kwargs)
        self.biases = self.bias_initialize("Zero")

        # outputs required
        self.deltas: np.ndarray = np.zeros(0, dtype=self.dtype) # initialized in training to fit training data size
        self.outputs = np.zeros(0, dtype=self.dtype)
        self.raw_outputs = np.zeros(0, dtype=self.dtype)

        # for adam optimizer
        self.weight_momenta = np.zeros_like(self.weights, dtype=self.dtype)
        self.weight_variances = np.zeros_like(self.weights, dtype=self.dtype)
        self.bias_momenta = np.zeros_like(self.biases, dtype=self.dtype)
        self.bias_variances = np.zeros_like(self.biases, dtype=self.dtype)

        # activation methods
        self.activation_method = kwargs["activation_method"] if "activation_method" in kwargs else "sigmoid"
    
    def process(self, input: np.ndarray, mask=False):
        if mask:
            # randomly turns off some neurons
            dropout_mask = self.rng.choice([0, 1], size=self.weights.shape[0], p=[0.05, 0.95])
            self.raw_outputs = ((dropout_mask[:, np.newaxis] * self.weights) @ input.T + (dropout_mask * self.biases)[:, np.newaxis]).T
            self.outputs = self.activate(self.raw_outputs)
        else:
            self.raw_outputs = (self.weights @ input.T + self.biases[:, np.newaxis]).T
            self.outputs = self.activate(self.raw_outputs)
        return self.outputs
