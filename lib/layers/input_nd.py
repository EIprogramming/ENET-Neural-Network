import numpy as np
from layers.input import Input

class InputND(Input):
    def __init__(self, shape: tuple, **kwargs):
        self.dtype = kwargs["dtype"] if "dtype" in kwargs else np.float64

        # activation methods
        self.activation_method = "None"
        self.shape = shape
    
    def process(self, inputs: np.ndarray):
        self.raw_outputs = inputs.reshape(self.shape)
        self.outputs = inputs.reshape(self.shape)
        return self.outputs
