import time
from typing import Type

import numpy as np
from sklearn.metrics import accuracy_score
from layers.layer import Layer
from layers.convolutional import Convolutional
from layers.dense import Dense
from layers.layer import Layer
import h5py
from layers.flatten import Flatten

class NeuralNet:
    def __init__(self, layers: list[Layer] | list[tuple[Type, tuple, tuple]], learning_rate: float = 0.01, **kwargs): # TODO: refactor docstring
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
        >>> neural_net = NeuralNet([
                Dense(28*28, 256, activation="ReLu", init="He"),
                Dense(256, 10, activation="softmax")],
                learning_rate=0.001, random_state=42, dtype=np.float32)
                # initialize a network with size 784 input layer, 256 hidden layer, and 10 output layer
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

        self.layers: list[Layer] = layers

        # initialize math
        self.loss = self.CE
        self.loss_derivative = self.CE_prime

        self.set_loss(self.loss_method)
    
    def __str__(self):
        self_str = ""
        for layer in reversed(self.layers):
            self_str += str(layer) + "\n"
        return self_str
    
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

    @staticmethod
    def flatten(input: np.ndarray):
        batch_size = input.shape[0]
        return input.reshape((batch_size, -1))

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
        first_moment = self.momentum(gradient, prev_momentum, beta_1)

        # variance
        second_moment = self.RMSprop_variance(gradient, prev_variance, beta_2)

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

    @staticmethod
    def batch (X: np.ndarray, y: np.ndarray, batch_size: int, batch_number: int, num_batches: int) -> tuple[np.ndarray, np.ndarray]:
        batch_start = batch_size * batch_number
        batch_end = batch_size * (batch_number + 1) if batch_number != num_batches - 1 else None

        X_i = X[batch_start : batch_end]
        y_i = y[batch_start : batch_end]
        return X_i, y_i
    
    def noise_normalize(self, X: np.ndarray, mu = 0.05, sigma = 0.05):
        # add some small amount of noise to half of X, randomly
        mask = self.rng.integers(0, 1, size=X.shape)
        noise = self.rng.normal(mu, sigma, size=X.shape).clip(0, 1)
        X += noise * mask
        return X

    def adamW(self, layer: Layer, adam_t: int, output_k: np.ndarray, batch_size: int):
        # update weights
        weight_decay = 0.01
        
        grad_loss = layer.deltas.T @ output_k / batch_size

        optimized_loss, \
            layer.weight_momenta, \
            layer.weight_variances = self.adam_optimize(adam_t,
                                                                    grad_loss,
                                                                    layer.weight_momenta,
                                                                    layer.weight_variances)
        # update weights in place
        layer.weights *= (1 - self.learning_rate * weight_decay)
        layer.weights -= self.learning_rate * optimized_loss
        # update biases
        grad_bias = np.mean(layer.deltas, axis=0)
        optimized_delta, \
            layer.bias_momenta, \
            layer.bias_variances = self.adam_optimize(adam_t, 
                                                                grad_bias, 
                                                                layer.bias_momenta,
                                                                layer.bias_variances)
        # update biases in place
        layer.biases *= (1 - self.learning_rate * weight_decay)
        layer.biases -= self.learning_rate * optimized_delta

    def dense_backpropagate(self, i, layer: Dense, X_i, y_exp_i, y_pred, adam_t, batch_size):
        raw_output = layer.raw_outputs
        if i == len(self.layers) - 1:
            if (layer.activation_method == "softmax" and (self.loss_method == "CE" or self.loss_method == "BCE")):
                layer.deltas = self.grad_softmax_CE(y_pred, y_exp_i)
            else:
                layer.deltas = self.loss_derivative(y_pred, y_exp_i) * layer.activation_derivative(raw_output)
        else:
            # sum along the weights and the previous deltas along their respective axes
            next_layer = self.layers[i + 1]
            if isinstance(next_layer, Dense):
                sum_delta_weights = next_layer.deltas @ next_layer.weights
                layer.deltas = sum_delta_weights * layer.activation_derivative(raw_output)
            elif isinstance(next_layer, Flatten):
                sum_delta_weights = next_layer.deltas
                layer.deltas = sum_delta_weights * layer.activation_derivative(raw_output)
        if i == 0:
            output_k: np.ndarray = X_i
        else:
            output_k: np.ndarray = self.layers[i - 1].outputs

        self.adamW(layer, adam_t, output_k, batch_size)

    def flatten_backpropagate(self, i, layer: Flatten, batch_size):
        # sum along the weights and the previous deltas along their respective axes
        next_layer = self.layers[i + 1]
        sum_delta_weights = next_layer.deltas @ next_layer.weights
        layer.deltas = layer.unflatten(batch_size, sum_delta_weights)

    def convolutional_backpropagate(self, i, layer: Convolutional, X_i, y_exp_i, y_pred, adam_t, batch_size):
        raw_output = layer.raw_outputs
        if i == len(self.layers) - 1:
            raise NotImplementedError("Convolutional layer as final layer not yet implemented.")
        else:
            # sum along the weights and the previous deltas along their respective axes
            next_layer = self.layers[i + 1]
            if isinstance(next_layer, Dense):
                sum_delta_weights = next_layer.deltas @ next_layer.weights
                        #print(sum_delta_weights.shape, raw_output.shape)
                layer.deltas = sum_delta_weights * layer.activation_derivative(raw_output)
            elif isinstance(next_layer, Flatten):
                sum_delta_weights = next_layer.deltas
                        #print(sum_delta_weights.shape, raw_output.shape)
                layer.deltas = sum_delta_weights * layer.activation_derivative(raw_output)
        if i == 0:
            output_k: np.ndarray = X_i
        else:
            output_k: np.ndarray = self.layers[i - 1].outputs

        # convert 2D input data to 3D
        if(len(output_k.shape) == 3):
            shape_3D = output_k.shape + (1,)
            output_k = output_k.reshape(shape_3D)
        output_k = layer.pad(output_k, 1)
        #print("layer_deltas before:", layer.deltas.shape)
        #layer.deltas = np.mean(layer.deltas, axis=0)
        layer.deltas = layer.deltas.reshape((layer.deltas.shape[0],) + (1,) + layer.deltas.shape[1:])
        
        output_k = output_k.reshape((output_k.shape[0],) + (1,) + output_k.shape[1:])

        #print("layer_deltas after:", layer.deltas.shape)
        #print("Convolved: ", output_k.shape, layer.deltas.shape)
        grad = layer.convolve3D(output_k, layer.deltas, None, 1, True, axes=(2,3))
        #print("Grad: ", grad.shape)
        grad_avg = np.mean(grad, axis=(-1))
       # grad_avg = grad_avg.reshape((1,) + grad_avg.shape)
        #print("grad_avg: ", grad_avg.shape)
        #print("kernels: ", layer.kernels.shape, f"\n{"-"*16}")
        #print(np.sum(layer.kernels))
        layer.kernels -= self.learning_rate * grad_avg
            
        #self.adamW(layer, adam_t, output_k, batch_size)

    def backpropagate(self, X_i, y_exp_i, y_pred, adam_t, batch_size):
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            # for each layer, starting from the last, go through each node and calculate the deltas
            if isinstance(layer, Dense):
                self.dense_backpropagate(i, layer, X_i, y_exp_i, y_pred, adam_t, batch_size)
            elif isinstance(layer, Flatten):
                self.flatten_backpropagate(i, layer, batch_size)
            elif isinstance(layer, Convolutional):
                self.convolutional_backpropagate(i, layer, X_i, y_exp_i, y_pred, adam_t, batch_size)
        

    def modify_inputs(self, X, **kwargs):
        noise_normalize = kwargs["noise_normalize"] if "noise_normalize" in kwargs else False
        if noise_normalize:
            X = self.noise_normalize(X)
        return X

    def train(self, X: np.ndarray, y_exp: np.ndarray, epochs = 10, batch_size=32, validate = None,
              display=True, **kwargs):

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

        for epoch in range(epochs):
            TIMER_epoch = time.time()
            shuffled_indices_epoch = shuffled_indices[epoch]
            X = X[shuffled_indices_epoch]
            y_exp = y_exp[shuffled_indices_epoch]
            
            num_batches = len(X)//batch_size
            if (len(X) % batch_size) != 0: num_batches += 1

            losses = np.zeros(num_batches)
            # batch_number ranges from 
            for batch_number in range(num_batches):
                # slice a batch of the shuffled X
                X_i, y_exp_i = NeuralNet.batch(X, y_exp, batch_size, batch_number, num_batches)

                X_i = self.modify_inputs(X_i, noise_normalize=noise_normalize)

                y_pred = self.predict(X_i, mask=dropout)

                # update adam optimizer t value
                adam_t += 1

                # backpropogation
                self.backpropagate(X_i, y_exp_i, y_pred, adam_t, batch_size)

                losses[batch_number] = np.mean(self.loss(y_pred, y_exp_i))
            TIMER_epoch = time.time() - TIMER_epoch

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
                    if ("continuous" not in kwargs or kwargs["binary"] == False):
                        y_test_pred = np.where(y_test_pred >= 0.5, 1, 0)
                    accuracy_train = accuracy_score(y_test_pred, y_test)
                    self.report["accuracy"][epoch] = accuracy_train
                    self.print_epoch_report(epoch, epoch_loss, epoch_losses, accuracy_train, prev_accuracy_train, TIMER_epoch)
                    prev_accuracy_train = accuracy_train
                else:
                    if epoch % 20 == 0: print(f"EPOCH: {epoch + 1}    LOSS: {epoch_loss}    LOSS CHANGE: {epoch_loss - epoch_losses[epoch - 1] if epoch > 0 else 0:.4g}")
        return epoch_losses, self.report
