import numpy as np
from layers.layer import Layer

class Input(Layer):
    def __init__(self, **kwargs):
        self.dtype = kwargs["dtype"] if "dtype" in kwargs else np.float64

        # activation methods
        self.activation_method = "None"
    
    def process(self, inputs: np.ndarray, mask=False):
        self.outputs = inputs
        return self.outputs
