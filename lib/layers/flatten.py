import numpy as np
from layers.layer import Layer

class Flatten(Layer):
    def __init__(self, shape_input: int | tuple, n_output: int, **kwargs):
        # initialize data types
        self.dtype = kwargs["dtype"] if "dtype" in kwargs else np.float64

        # initialize matrix shape
        self.shape_input = (shape_input,) if isinstance(shape_input, int) else shape_input
        self.n_output = n_output

        self.outputs = np.zeros(0, dtype=self.dtype)
        self.raw_outputs = np.zeros(0, dtype=self.dtype)

        self.deltas = np.zeros(0, dtype=self.dtype)

    def unflatten(self, batch_size, output):
        output_shape = (batch_size, ) + self.shape_input
        return output.reshape(output_shape)

    def flatten(self, input: np.ndarray):
        batch_size = input.shape[0]
        output_shape = (batch_size, self.n_output)
        self.raw_outputs = input.reshape(output_shape)
        self.outputs = self.raw_outputs


        return self.outputs
    
    def process(self, input: np.ndarray, mask = None):
        return self.flatten(input)
