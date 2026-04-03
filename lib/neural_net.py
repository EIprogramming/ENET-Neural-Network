import csv
import time

import numpy as np
from sklearn.metrics import accuracy_score
from layer import Layer
import h5py

class NeuralNet:
    def __init__(self, layer_sizes: tuple, learning_rate: float = 0.01, **kwargs):
        if len(layer_sizes) < 2:
            raise ValueError(f"Shape must have at least two layers")
        
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
        self.loss_method = kwargs["loss_method"] if "loss_method" in kwargs else ""
        
        self.random_state = kwargs["random_state"] if "random_state" in kwargs else None
        self.rng = np.random.default_rng(self.random_state)

        self.layers: list[Layer] = []
        prev_layer_size = 0
        for index, layer_size in enumerate(layer_sizes):
            if (index == 0):
                prev_layer_size = layer_size
                continue # skip the input layer (we already take inputs as they are)
            self.layers.append(Layer(prev_layer_size, layer_size, dtype=self.dtype, random_state=self.random_state))
            prev_layer_size = layer_size
    
    def __str__(self):
        self_str = ""
        for layer in reversed(self.layers):
            self_str += str(layer) + "\n"
        return self_str
    
    def process(self, inputs: np.ndarray, mask=False):
        current_inputs = inputs
        for layer in self.layers:
            current_inputs = layer.process(current_inputs, mask)
        
        return current_inputs

    def batch_eval(self, inputs: np.ndarray, mask=False):
        current_inputs = inputs.T # shape (2000, 2 becomes) (2, 2000)
        for layer in self.layers:
            current_inputs = layer.batch_process(current_inputs, mask)
        return current_inputs

    # TODO: fix transform issues
    def predict(self, inputs: np.ndarray):
        return self.batch_eval(inputs).T
    
    def eval(self, inputs: np.ndarray, mask=False):
        outputs = []
        for i in range(len(inputs)):
            outputs.append(self.process(inputs[i], mask))
        return np.array(outputs, dtype=self.dtype)

    def BCE(self, y_pred_all: np.ndarray, y_exp_all: np.ndarray):
        n: int = len(y_pred_all)
        epsilon = 0.00001
        loss_BCE = 0

        for y_pred, y_exp in zip(y_pred_all, y_exp_all):
            loss_BCE += y_exp * np.log(y_pred + epsilon) + (1 - y_exp) * np.log(1 - y_pred + epsilon)

        loss_BCE *= - 1 / n

        return loss_BCE

    def BCE_prime(self, y_pred, y_exp):
        epsilon = 0.000001
        return (y_pred - y_exp) / (y_pred * (1 - y_pred) + epsilon)
    
    def grad_softmax_CE(self, y_pred, y_exp):
        return y_pred - y_exp

    def momentum(self, gradient, prev_momentum, beta = 0.9):
        # m = B * m_{t-1} + (1 - B) * dL/dw
        return beta * prev_momentum + (1 - beta) * gradient

    def RMSprop_variance(self, gradient, prev_variance, beta = 0.999):
        return beta * prev_variance + (1 - beta) * gradient ** 2

    def adam_optimize(self, n, gradient, prev_momentum, prev_variance, beta_1 = 0.9, beta_2 = 0.999, epsilon=1e-8):
        # momentum / mean
        first_moment = self.momentum(gradient, prev_momentum, beta_1)

        # variance
        second_moment = self.RMSprop_variance(gradient, prev_variance, beta_2)

        first_moment_corrected = first_moment / (1 - beta_1**(n + 1))
        second_moment_corrected = second_moment / (1 - beta_2**(n + 1))
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
                print(layer.weights.shape, layer.activation_method)

    def print_epoch_report(self, epoch, epoch_loss, epoch_losses, accuracy_train, prev_accuracy_train):
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

        print(str_epoch, str_loss, str_loss_change, str_accuracy_train, str_accuracy_change, sep="    ")

    def train(self, X: np.ndarray, y_exp: np.ndarray, epochs = 10, batch_size=32, validate = None,
              display=True, verbose=False, lr_scheduling = False, **kwargs):
        # cast data types:
        X = X.astype(self.dtype)
        y_exp = y_exp.astype(self.dtype)

        self.initialize_report(epochs)

        noise_normalize = kwargs["noise_normalize"] if "noise_normalize" in kwargs else False
        partial_randomization = kwargs["partial_randomization"] if "partial_randomization" in kwargs else False
        dropout = kwargs["dropout"] if "dropout" in kwargs else False
        
        epoch_losses = np.zeros(epochs)
        prev_accuracy_train = 0
        indices = np.arange(X.shape[0])
        shuffled_indices = np.array([self.rng.permutation(indices) for _ in range(epochs)])

        # for adam optimization
        adam_t = 0
        initial_learning_rate = self.learning_rate
        lr_k = 0.000000230/batch_size # learning rate decay constant

        for epoch in range(epochs):
            #TOTAL_TIMER_epoch = 0
            #TOTAL_TIMER_batch = 0
            #TOTAL_TIMER_backprop = 0
            #TOTAL_TIMER_backprop1 = 0
            #TOTAL_TIMER_1 = 0
            #TIMER_epoch = time.time()
            losses = []
            Y_pred = []
            shuffled_indices_epoch = shuffled_indices[epoch]#self.rng.permutation(X.shape[0])
            X_shuffled = X[shuffled_indices_epoch]
            y_exp_shuffled = y_exp[shuffled_indices_epoch]
            # if using partial randomization (i.e. 80% of train data randomly)
            if partial_randomization:
                n = 0.25
                n_percent_of_X = int(n * len(X_shuffled))
                X_shuffled = X_shuffled[:-n_percent_of_X]
                y_exp_shuffled = y_exp_shuffled[:-n_percent_of_X]
            
            for batch_number in range(len(X_shuffled)//batch_size):
                #TIMER_batch = time.time()
                X_i = X_shuffled[batch_size * batch_number : batch_size * (batch_number + 1)]
                if noise_normalize:
                    mu, sigma = 0.05, 0.05
                    X_i += self.rng.normal(mu, sigma, size=X_i.shape).clip(0, 1)
                    #X_i += 0.05*self.rng.random(X_i.shape)
                y_exp_i = y_exp_shuffled[batch_size * batch_number : batch_size * (batch_number + 1)].T
                y_pred = self.batch_eval(X_i, mask=dropout)
                #TIMER_1 = time.time()
                Y_pred.append(y_pred)
                #TOTAL_TIMER_1 += time.time() - TIMER_1

                adam_t += 1

                # backpropogation
                for i in reversed(range(len(self.layers))): # TODO: fix this weird reversed loop
                    #TIMER_backprop = time.time()
                    # for each layer, starting from the last, go through each node and calculate the deltas
                    raw_output = self.layers[i].batch_raw_outputs
                    if i == len(self.layers) - 1:
                        if (self.layers[i].activation_method == "softmax"):
                            delta_j = self.grad_softmax_CE(y_pred, y_exp_i)
                        else:
                            delta_j = self.BCE_prime(y_pred, y_exp_i) * self.layers[i].activation_derivative(raw_output)
                        self.layers[i].batch_deltas = delta_j
                    else:
                        # sum along the weights and the previous deltas along their respective axes
                        sum_delta_weights = np.einsum('ij,ik->jk', self.layers[i + 1].weights, self.layers[i + 1].batch_deltas, optimize=True)
                        delta_j = sum_delta_weights * self.layers[i].activation_derivative(raw_output)
                        self.layers[i].batch_deltas = delta_j
                    if i == 0:
                        output_k: np.ndarray = X_i.T
                    else:
                        output_k: np.ndarray = self.layers[i - 1].batch_outputs

                    #TIMER_backprop1 = time.time()
                    # update weights
                    grad_loss = (np.einsum('kb,jb->jk', output_k, delta_j, optimize=True) / output_k.shape[1])
                    #TOTAL_TIMER_backprop1 += time.time() - TIMER_backprop1
                    weight_momenta = self.layers[i].weight_momenta
                    weight_variances = self.layers[i].weight_variances
                    optimized_loss, weight_momenta, weight_variances = self.adam_optimize(adam_t, grad_loss, weight_momenta, weight_variances)
                    self.layers[i].weights += -self.learning_rate * optimized_loss

                    # update previous weight momenta / variances
                    self.layers[i].weight_momenta = weight_momenta
                    self.layers[i].weight_variances = weight_variances

                    # update biases
                    grad_bias = np.mean(delta_j, axis=1)
                    bias_momenta = self.layers[i].bias_momenta
                    bias_variances = self.layers[i].bias_variances
                    optimized_delta, bias_momenta, bias_variances = self.adam_optimize(adam_t, grad_bias, bias_momenta, bias_variances)
                    self.layers[i].biases -= self.learning_rate * optimized_delta
                
                    # update previous bias momenta / variances
                    self.layers[i].bias_momenta = bias_momenta
                    self.layers[i].bias_variances = bias_variances

                    if lr_scheduling:
                        self.learning_rate = initial_learning_rate * np.exp(- lr_k * adam_t)

                    #TOTAL_TIMER_backprop += time.time() - TIMER_backprop
                loss = self.BCE(y_pred, y_exp_i)
                losses.append(loss)
                #TOTAL_TIMER_batch += time.time() - TIMER_batch
            
            # reshape Y_pred:
            Y_pred = np.array(Y_pred, dtype=self.dtype)
            Y_pred = np.concatenate(Y_pred, axis=-1)

            # calculate loss
            epoch_loss = np.mean(losses)
            epoch_losses[epoch] = epoch_loss
            
            self.report["loss"][epoch] = epoch_loss

            # display epoch
            if display and epoch % 1 == 0:
                if verbose: print(f"EPOCH: {epoch + 1}    LOSS: {epoch_loss}    LOSS CHANGE: {epoch_loss - epoch_losses[epoch - 1] if epoch > 0 else 0:.4g}")
                else:
                    if validate is not None and len(validate) == 2:
                        X_test = validate[0]
                        y_test = validate[1]
                        y_test_pred = self.predict(X_test)
                        if ("binary" in kwargs and kwargs["binary"] == True) or ("categorical" in kwargs and kwargs["categorical"] == True):
                            y_test_pred = np.where(y_test_pred >= 0.5, 1, 0)
                            if (y_test.shape != y_test_pred.shape):
                                y_test_pred = np.argmax(y_test_pred, axis = -1)
                        accuracy_train = accuracy_score(y_test_pred, y_test)
                        self.report["accuracy"][epoch] = accuracy_train
                        self.print_epoch_report(epoch, epoch_loss, epoch_losses, accuracy_train, prev_accuracy_train)
                        prev_accuracy_train = accuracy_train
                    else:
                        
                        if epoch % 20 == 0: print(f"EPOCH: {epoch + 1}    LOSS: {epoch_loss}    LOSS CHANGE: {epoch_loss - epoch_losses[epoch - 1] if epoch > 0 else 0:.4g}")

            #TOTAL_TIMER_epoch += time.time() - TIMER_epoch
            #print(f"Timer backprop: {TOTAL_TIMER_backprop:.4g}")
            #print(f"Timer backprop1: {TOTAL_TIMER_backprop1:.4g}")
            #print(f"Timer batch: {TOTAL_TIMER_batch:.4g}")
            #print(f"Timer epoch: {TOTAL_TIMER_epoch:.4g}")
            #print(f"Timer 1: {TOTAL_TIMER_1:.4g}")
        return epoch_losses, np.array(Y_pred)
