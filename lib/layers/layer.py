import numpy as np

class Layer:
    def Xavier_initialization(self):
        limit = np.sqrt(6 / (self.n_input + self.n_output))

        return self.rng.uniform(-limit, limit, size=self.shape).astype(self.dtype)
    
    def He_initialization(self):
        mu = 0
        sigma = np.sqrt(2 / self.n_input)
        return self.rng.normal(mu, sigma, size=self.shape).astype(self.dtype)

    def initialize(self, init_method: str="Xavier", **kwargs) -> np.ndarray:
        if init_method == "Xavier":
            return self.Xavier_initialization()
        elif init_method == "He":
            return self.He_initialization()
        else:
            print("No weight init specified...", init_method)
            return np.zeros(shape=self.shape, dtype=self.dtype)
        
    def initialize_weights(self, init_method):
        self.weights = self.initialize(init_method)

    def bias_initialize(self, init_method: str=""):
        if init_method == "Zero":
            return np.zeros(shape=self.n_output, dtype=self.dtype)
        else:
            print("No bias init specified...", init_method)
            return np.zeros(shape=self.n_output, dtype=self.dtype)

    def get_max_dtype(self, dtype):
        return np.finfo(dtype).max if issubclass(dtype, np.floating) else np.iinfo(dtype).max

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
    
    def __str__(self) -> str:
        self_str = ""
        for neuron_index in range(self.n_output):
            self_str += "{("
            for weight in self.weights[neuron_index]:
                self_str += f"{weight:.3g}, "
            self_str += f") + {self.biases[neuron_index]:.3g}}}, "
        return self_str
    
    def __repr__(self) -> str:
        return str(self) + "\n"

    def sigmoid(self, X: np.ndarray | float):
        return 1 / (1 + np.exp(-np.clip(X, -self.sigmoid_clip_value, self.sigmoid_clip_value)))

    def ReLu(self, X: np.ndarray | float):
        return np.where(X > 0, X, 0)
    
    def ReLu_derivative(self, X: np.ndarray | float):
        return np.where(X > 0, 1, 0)
    
    def softmax(self, X: np.ndarray | float):
        # source: https://stats.stackexchange.com/questions/304758/softmax-overflow
        # to prevent overflow, subtract by the maximum m = x_i

        m = np.max(self.raw_outputs, axis=1, keepdims=True)

        result = np.exp(X - m)/np.sum(np.exp(self.raw_outputs - m), axis=-1)[:, np.newaxis]

        return result

    def activate(self, X: np.ndarray):
        if self.activation_method == "sigmoid":
            return self.sigmoid(X)
        elif self.activation_method == "ReLu":
            return self.ReLu(X)
        elif self.activation_method == "softmax":
            return self.softmax(X)
        elif self.activation_method == "logit":
            return X
        elif self.activation_method == "None":
            return X
        else:
            print("No activation specified...")
            return X
    
    def activation_derivative(self, X: float | np.ndarray):
        if self.activation_method == "sigmoid":
            return self.sigmoid(X) * (1 - self.sigmoid(X))
        elif self.activation_method == "ReLu":
            return self.ReLu_derivative(X)
        elif self.activation_method == "logit":
            return np.ones_like(X, dtype=self.dtype)
        else:
            print("No activation derivative specified...")
            return np.ones_like(X, dtype=self.dtype) # for default no activation function, derivative is 1

    def process(self, input: np.ndarray, mask=False):
        raise NotImplementedError # abstract base class ()

