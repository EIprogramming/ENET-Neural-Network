import time

import numpy as np
from sklearn.metrics import accuracy_score
from layers.layer import Layer
from layers.convolutional import Convolutional
from layers.dense import Dense
from layers.input import Input
from layers.input_nd import InputND
from layers.layer import Layer
from layers.new_layer import NewLayer
import h5py

class NeuralNet:
    def __init__(self, layers: list[NewLayer], learning_rate: float = 0.01, **kwargs): # TODO: refactor docstring
        """Initialize a neural network object.

        Parameters
        ----------

        layer_sizes :
            A tuple of the network's layer sizes.
            First value denotes input size, final value denotes output size. Must have at least an input and output size.

        learning_rate : optional
            The multiplier applied to the weight change during backpropagation. Default is 0.01.

        loss_method : { "CE", "BCE", "MSE" }, optional
            The method used to calculate loss. Default is "CE".

        random_state : {int or None}, optional
            The random state of the random number generator. Default is None (seed generated from operating system).

        dtype : optional
            The data type used in the network. Default is np.float64.

            
        Examples
        --------
        >>> NeuralNet((28**2, 256, 10)) # initialize a network with size 784 input layer, 256 hidden layer, and 10 output layer
        >>> NeuralNet((2, 16, 2), learning_rate = 0.005, loss_method = "MSE", random_state=42, dtype=np.float32)
        """
        if len(layers) < 2:
            raise ValueError(f"Neural Network must have an input and output layer.")
        
        # data type
        self.dtype = kwargs["dtype"] if "dtype" in kwargs else np.float64

        # report values
        self.report = {
            "epochs" : 0,
            "loss" : np.array([]),
            "test accuracy" : np.array([]),
        }

        # learning params
        self.learning_rate = learning_rate
        self.loss_method = kwargs["loss_method"] if "loss_method" in kwargs else "CE"
        
        self.random_state = kwargs["random_state"] if "random_state" in kwargs else None
        self.rng = np.random.default_rng(self.random_state)

        self.layers: list[Layer] = self.create_layers(layers)

        # initialize math
        self.loss = self.CE
        self.loss_derivative = self.CE_prime

        self.set_loss(self.loss_method)
    
    def __str__(self):
        self_str = ""
        for layer in reversed(self.layers):
            self_str += str(layer) + "\n"
        return self_str

    def create_layers(self, new_layers: list[NewLayer]) -> list[Layer]:
        layers: list[Layer] = []
        prev_layer_shape: tuple = ()
        for new_layer in new_layers:
            layer_shape = new_layer.shape
            match new_layer.layer_type:
                case "Dense":
                    layers.append(Dense(prev_layer_shape[0], layer_shape[0], dtype=self.dtype, random_state=self.random_state))
                case "Input":
                    pass # we dont need to specify an input layer in this case
                case "InputND":
                    layers.append(InputND(layer_shape))
                case "Convolutional":
                    #layers.append(Convolutional(layer_shape, 2, 3, 2, 1))
                    ... # TODO: add convolutional back to Layer inheritance
                case _:
                    raise ValueError(f"Invalid layer type {new_layer.layer_type}")
            prev_layer_shape = layer_shape
        return layers

    def set_loss(self, loss_method):
        """Set the method used for the loss function.
        
        Parameters
        ----------

        loss_method: {"CE", "BCE", "MSE"}
            The method used to calculate the loss.
        """
        match loss_method:
            case "CE":
                self.loss = self.CE
                self.loss_derivative = self.CE_prime
            case "BCE":
                self.loss = self.BCE
                self.loss_derivative = self.BCE_prime
            case "MSE":
                self.loss = self.MSE
                self.loss_derivative = self.MSE_prime
            case _:
                raise ValueError(f"loss_method must be 'CE', 'BCE', or 'MSE', got {loss_method}")

    def predict(self, inputs: np.ndarray, mask=False):
        current_inputs = inputs
        for layer in self.layers:
            current_inputs = layer.process(current_inputs, mask)
        return current_inputs

    def BCE(self, y_pred_all: np.ndarray, y_exp_all: np.ndarray):
        return self.CE(y_pred_all, y_exp_all)

    def BCE_prime(self, y_pred, y_exp):
        epsilon = 1e-8
        return (y_pred - y_exp) / (y_pred * (1 - y_pred) + epsilon)

    def CE(self, y_pred: np.ndarray, y_exp: np.ndarray):
        n: int = len(y_pred)
        epsilon = 1e-8

        loss_CE = - np.sum(y_exp * np.log(y_pred + epsilon), axis=0) / n

        return loss_CE

    # same as BCE prime but repeated to reduce overhead
    def CE_prime(self, y_pred, y_exp):
        epsilon = 1e-8
        return (y_pred - y_exp) / (y_pred * (1 - y_pred) + epsilon)
    
    def MSE(self, y_pred: np.ndarray, y_exp: np.ndarray):
        n: int = len(y_pred)

        loss_MSE = np.sum((y_pred - y_exp)**2, axis=0) / n

        return loss_MSE

    def MSE_prime(self, y_pred, y_exp):
        return 2 * (y_pred - y_exp)
    
    def grad_softmax_CE(self, y_pred, y_exp):
        return y_pred - y_exp

    def momentum(self, gradient, prev_momentum, beta = 0.9):
        # m = B * m_{t-1} + (1 - B) * dL/dw
        return beta * prev_momentum + (1 - beta) * gradient

    def RMSprop_variance(self, gradient, prev_variance, beta = 0.999):
        return beta * prev_variance + (1 - beta) * gradient ** 2

    def adam_optimize(self, t, gradient, prev_momentum, prev_variance, beta_1 = 0.9, beta_2 = 0.999, epsilon=1e-8):
        # momentum / mean
        #print(gradient.shape, prev_momentum.shape, prev_variance.shape)
        #print(gradient.flags)
        #print(prev_momentum.flags)
        #print(prev_variance.flags)
        first_moment = self.momentum(gradient, prev_momentum, beta_1)

        # variance
        second_moment = self.RMSprop_variance(gradient, prev_variance, beta_2)

        #step_size = 1 / (1 - beta_1**t)
        #bias_correction2 = (1 - beta_2**t) ** 0.5

        first_moment_corrected = first_moment / (1 - beta_1**(t + 1))
        second_moment_corrected = second_moment / (1 - beta_2**(t + 1))
        optimized_loss = first_moment_corrected / (np.sqrt(second_moment_corrected) + epsilon)

        return optimized_loss, first_moment, second_moment
    
    def initialize_report(self, epochs):
        self.report = {
            "epochs": epochs,
            "loss": np.zeros(epochs),
            "accuracy": np.zeros(epochs),
        }

    def export(self, filename):
        if filename[-3:] != ".h5":
            raise ValueError("File must be in .h5")
        
        with h5py.File(filename, 'w') as file:
            for index, layer in enumerate(self.layers):
                group = file.create_group(f"layer{index}")
                group.create_dataset("weights", data=layer.weights)
                group.create_dataset("biases", data=layer.biases)
                group.attrs["activation"] = layer.activation_method
        
        print(f"Neural Network exported to {filename}")
    
    def load(self, filename):
        if filename[-3:] != ".h5":
            raise ValueError("File must be in .h5")

        with h5py.File(filename, 'r') as file:
            for index, layer in enumerate(self.layers):
                layer.weights = np.array(file[f"layer{index}/weights"])
                layer.biases = np.array(file[f"layer{index}/biases"])
                layer.activation_method = file[f"layer{index}"].attrs["activation"]

    def print_epoch_report(self, epoch, epoch_loss, epoch_losses, accuracy_train, prev_accuracy_train, epoch_time):
        str_epoch = f"EPOCH: {epoch + 1}"
        str_loss = f"LOSS: {epoch_loss:6g}"
        loss_change = epoch_loss - epoch_losses[epoch - 1] if epoch > 0 else 0
        if loss_change < 0:
            str_loss_change = f"LOSS CHANGE: \033[1;34m{loss_change:.4g}\033[0m"
        elif loss_change == 0:
            str_loss_change = f"LOSS CHANGE: {loss_change:.4g}"
        else:
            str_loss_change = f"LOSS CHANGE: \033[1;31m{loss_change:.4g}\033[0m"

        str_accuracy_train = f"TEST ACCURACY: {accuracy_train:.4g}"

        accuracy_change = accuracy_train - prev_accuracy_train

        if accuracy_change > 0:
            str_accuracy_change = f"TEST ACCURACY CHANGE \033[1;32m{accuracy_change:.4g}\033[0m"
        elif accuracy_change == 0:
            str_accuracy_change = f"TEST ACCURACY CHANGE {accuracy_change:.4g}"
        else:
            str_accuracy_change = f"TEST ACCURACY CHANGE \033[1;31m{accuracy_change:.4g}\033[0m"
        
        str_epoch_time = f"TIME: {epoch_time:.2g} s"

        print(str_epoch, str_loss, str_loss_change, str_accuracy_train, str_accuracy_change, str_epoch_time, sep="    ")

    def train(self, X: np.ndarray, y_exp: np.ndarray, epochs = 10, batch_size=32, validate = None,
              display=True, lr_scheduling = False, **kwargs):

        # set parameters:
        noise_normalize = kwargs["noise_normalize"] if "noise_normalize" in kwargs else False
        dropout = kwargs["dropout"] if "dropout" in kwargs else False

        # cast data types:
        X = X.astype(self.dtype)
        y_exp = y_exp.astype(self.dtype)

        self.initialize_report(epochs)
        
        epoch_losses = np.zeros(epochs)
        prev_accuracy_train = 0
        indices = np.arange(X.shape[0])
        shuffled_indices = np.array([self.rng.permutation(indices) for _ in range(epochs)])

        # for adam optimization
        adam_t = 0
        initial_learning_rate = self.learning_rate
        lr_k = 0.000000230/batch_size # learning rate decay constant factor

        for epoch in range(epochs):
            TIMER_epoch = time.time()
            losses = []
            Y_pred = []
            shuffled_indices_epoch = shuffled_indices[epoch]
            X = X[shuffled_indices_epoch]
            y_exp = y_exp[shuffled_indices_epoch]
            
            for batch_number in range(len(X)//batch_size):
                # slice a batch of the shuffled X
                X_i = X[batch_size * batch_number : batch_size * (batch_number + 1)]
                if noise_normalize:
                    mu, sigma = 0.05, 0.05
                    X_i += self.rng.normal(mu, sigma, size=X_i.shape).clip(0, 1)
                y_exp_i = y_exp[batch_size * batch_number : batch_size * (batch_number + 1)]
                y_pred = self.predict(X_i, mask=dropout)
                Y_pred.append(y_pred)

                # update adam optimizer t value
                adam_t += 1

                # backpropogation
                for i in reversed(range(len(self.layers))):
                    # for each layer, starting from the last, go through each node and calculate the deltas
                    raw_output = self.layers[i].raw_outputs
                    if i == len(self.layers) - 1:
                        if (self.layers[i].activation_method == "softmax" and (self.loss_method == "CE" or self.loss_method == "BCE")):
                            delta_j = self.grad_softmax_CE(y_pred, y_exp_i)
                        else:
                            delta_j = self.loss_derivative(y_pred, y_exp_i) * self.layers[i].activation_derivative(raw_output)
                        self.layers[i].deltas = delta_j
                    else:
                        # sum along the weights and the previous deltas along their respective axes
                        sum_delta_weights = self.layers[i + 1].deltas @ self.layers[i + 1].weights
                        delta_j = sum_delta_weights * self.layers[i].activation_derivative(raw_output)
                        self.layers[i].deltas = delta_j
                    if i == 0:
                        output_k: np.ndarray = X_i
                    else:
                        output_k: np.ndarray = self.layers[i - 1].outputs

                    # update weights
                    weight_decay = 0.01
                    grad_loss = delta_j.T @ output_k / output_k.shape[0]
                    optimized_loss, \
                        self.layers[i].weight_momenta, \
                        self.layers[i].weight_variances = self.adam_optimize(adam_t,
                                                                             grad_loss,
                                                                             self.layers[i].weight_momenta,
                                                                             self.layers[i].weight_variances)

                    # update weights in place
                    self.layers[i].weights *= (1 - self.learning_rate * weight_decay)
                    self.layers[i].weights -= self.learning_rate * optimized_loss
                    
                    # update biases
                    grad_bias = np.mean(delta_j, axis=0)
                    optimized_delta, \
                        self.layers[i].bias_momenta, \
                        self.layers[i].bias_variances = self.adam_optimize(adam_t, 
                                                                           grad_bias, 
                                                                           self.layers[i].bias_momenta,
                                                                           self.layers[i].bias_variances)
                    
                    # update biases in place
                    self.layers[i].biases *= (1 - self.learning_rate * weight_decay)
                    self.layers[i].biases -= self.learning_rate * optimized_delta
                
                    if lr_scheduling:
                        self.learning_rate = initial_learning_rate * np.exp(- lr_k * adam_t)
                loss = self.loss(y_pred, y_exp_i)
                losses.append(loss)
            
            TIMER_epoch = time.time() - TIMER_epoch

            # reshape Y_pred:
            Y_pred = np.array(Y_pred, dtype=self.dtype)
            Y_pred = np.concatenate(Y_pred, axis=-1)

            # calculate loss
            epoch_loss = np.mean(losses)
            epoch_losses[epoch] = epoch_loss
            
            self.report["loss"][epoch] = epoch_loss
            # display epoch
            if display and epoch % 1 == 0:
                if validate is not None and len(validate) == 2:
                    X_test = validate[0]
                    y_test = validate[1]
                    y_test_pred = self.predict(X_test)
                    if ("binary" in kwargs and kwargs["binary"] == True) or ("categorical" in kwargs and kwargs["categorical"] == True):
                        y_test_pred = np.where(y_test_pred >= 0.5, 1, 0)
                    accuracy_train = accuracy_score(y_test_pred, y_test)
                    self.report["accuracy"][epoch] = accuracy_train
                    self.print_epoch_report(epoch, epoch_loss, epoch_losses, accuracy_train, prev_accuracy_train, TIMER_epoch)
                    prev_accuracy_train = accuracy_train
                else:
                    if epoch % 20 == 0: print(f"EPOCH: {epoch + 1}    LOSS: {epoch_loss}    LOSS CHANGE: {epoch_loss - epoch_losses[epoch - 1] if epoch > 0 else 0:.4g}")
        return epoch_losses, np.array(Y_pred)
