import numpy as np

class Layer:
    def xavier_initialization(self):
        mu = 0 # mean is 0 for Xavier_init
        sigma = np.sqrt(6 / self.n_input + self.n_output)

        return self.rng.normal(mu, sigma, size=self.shape)

    def initialize(self, init_method: str="Xavier", **kwargs) -> np.ndarray:
        if init_method == "Xavier":
            return self.xavier_initialization()
        else:
            return np.zeros(shape=self.shape)

    def bias_initialize(self, init_method: str=""):
        if init_method == "Zero":
            return np.zeros(shape=self.n_output)
        else:
            return np.zeros(shape=self.n_output)

    def __init__(self, n_input: int, n_output: int, **kwargs):
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
        self.biases = self.bias_initialize()

        # outputs required
        self.raw_outputs = np.zeros(n_output)
        self.outputs = np.zeros(n_output)
        self.deltas = np.zeros(n_output)
        self.batch_deltas: np.ndarray = np.zeros(0) # initialized in training to fit training data size
        self.batch_outputs = np.zeros(0)
        self.batch_raw_outputs = np.zeros(0)

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
        return 1 / (1 + np.exp(-X))

    def ReLu(self, X: np.ndarray | float):
        return np.where(X > 0, X, 0)
    
    def ReLu_derivative(self, X: np.ndarray | float):
        return np.where(X > 0, 1, 0)

    def activate(self, X: np.ndarray):
        if self.activation_method == "sigmoid":
            return self.sigmoid(X)
        elif self.activation_method == "ReLu":
            return self.ReLu(X)
        else:
            return X
    
    def activation_derivative(self, X: float | np.ndarray):
        if self.activation_method == "sigmoid":
            return self.sigmoid(X) * (1 - self.sigmoid(X))
        elif self.activation_method == "ReLu":
            return self.ReLu_derivative(X)
        else:
            return np.ones_like(X) # for default no activation function, derivative is 1

    def process(self, input: np.ndarray):
        if (input.shape[0] != (self.n_input)):
            raise ValueError(f"Input shape ({input.shape}, n) must equal layer input shape ({self.shape})")
        
        self.raw_outputs = self.weights @ input + self.biases
        self.outputs = self.activate(self.raw_outputs)

        return self.outputs

    def batch_process(self, input: np.ndarray):
        self.batch_raw_outputs = self.weights @ input + self.biases[:, np.newaxis]
        self.batch_outputs = self.activate(self.batch_raw_outputs)
        return self.batch_outputs