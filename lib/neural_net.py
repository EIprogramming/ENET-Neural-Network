import time

import numpy as np
from layer import Layer

class NeuralNet:
    def __init__(self, layer_sizes: tuple, learning_rate: float = 0.01, **kwargs):
        if len(layer_sizes) < 2:
            raise ValueError(f"Shape must have at least two layers")

        # learning params
        self.learning_rate = learning_rate
        
        self.shape = layer_sizes
        self.random_state = kwargs["random_state"] if "random_state" in kwargs else None
        self.rng = np.random.default_rng(self.random_state)

        self.layers: list[Layer] = []
        prev_layer_size = 0
        for index, layer_size in enumerate(layer_sizes):
            if (index == 0):
                prev_layer_size = layer_size
                continue # skip the input layer (we already take inputs as they are)
            self.layers.append(Layer(prev_layer_size, layer_size, random_state=self.random_state))
            prev_layer_size = layer_size
    
    def __str__(self):
        self_str = ""
        for layer in reversed(self.layers):
            self_str += str(layer) + "\n"
        return self_str
    
    def process(self, inputs: np.ndarray):
        current_inputs = inputs
        for layer in self.layers:
            current_inputs = layer.process(current_inputs)
        
        return current_inputs

    def batch_eval(self, inputs: np.ndarray):
        current_inputs = inputs.T # shape (2000, 2 becomes) (2, 2000)
        for layer in self.layers:
            current_inputs = layer.batch_process(current_inputs)
        return current_inputs

    # TODO: fix transform issues
    def predict(self, inputs: np.ndarray):
        return self.batch_eval(inputs).T
    
    def eval(self, inputs: np.ndarray):
        outputs = []
        for i in range(len(inputs)):
            outputs.append(self.process(inputs[i]))
        return np.array(outputs)

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

    def train(self, X: np.ndarray, y_exp: np.ndarray, epochs = 10, batch_size=32, display=True, verbose=False):
        epoch_losses = np.zeros(epochs)
        for epoch in range(epochs):
            TOTAL_TIMER_epoch = 0
            TOTAL_TIMER_batch = 0
            TOTAL_TIMER_backprop = 0
            TOTAL_TIMER_backprop1 = 0
            TIMER_epoch = time.time()
            losses = []
            Y_pred = []
            # TODO: shuffle the batches in the other batch train too
            shuffled_indices = self.rng.permutation(X.shape[0])
            X_shuffled = X[shuffled_indices]
            y_exp_shuffled = y_exp[shuffled_indices]
            for batch_number in range(len(X)//batch_size):
                TIMER_batch = time.time()
                X_i = X_shuffled[batch_size * batch_number : batch_size * (batch_number + 1)]
                y_exp_i = y_exp_shuffled[batch_size * batch_number : batch_size * (batch_number + 1)].T
                y_pred = self.batch_eval(X_i)
                Y_pred.append(y_pred)

                # backpropogation
                for i in reversed(range(len(self.layers))): # TODO: fix this weird reversed loop
                    TIMER_backprop = time.time()
                    # for each layer, starting from the last, go through each node and calculate the deltas
                    raw_output = self.layers[i].batch_raw_outputs
                    if i == len(self.layers) - 1:
                        if (self.layers[i].activation_method != "softmax"):
                            delta_j = self.BCE_prime(y_pred, y_exp_i) * self.layers[i].activation_derivative(raw_output)
                        else:
                            delta_j = self.grad_softmax_CE(y_pred, y_exp_i)
                        self.layers[i].batch_deltas = delta_j
                    else:
                        sum_delta_l = np.sum(self.layers[i + 1].batch_deltas)
                        delta_j = sum_delta_l * self.layers[i].activation_derivative(raw_output)
                        self.layers[i].batch_deltas = delta_j
                    if i == 0:
                        output_k: np.ndarray = X_i.T
                    else:
                        output_k: np.ndarray = self.layers[i - 1].batch_outputs

                    TIMER_backprop1 = time.time()
                    #print(output_k.shape, delta_j.shape)
                    # update weights
                    #grad_loss = np.mean(output_k[:, np.newaxis, :] * delta_j[np.newaxis, :, :], axis = -1).T
                    #print(f"old shape: {grad_loss.shape}")
                    grad_loss = (np.einsum('kb,jb->jk', output_k, delta_j) / output_k.shape[1])
                    #print(f"new shape: {grad_loss.shape}")
                    TOTAL_TIMER_backprop1 += time.time() - TIMER_backprop1
                    weight_momenta = self.layers[i].weight_momenta
                    weight_variances = self.layers[i].weight_variances
                    optimized_loss, weight_momenta, weight_variances = self.adam_optimize(epoch, grad_loss, weight_momenta, weight_variances)
                    weight_change = -self.learning_rate * optimized_loss
                    self.layers[i].weights += -self.learning_rate * optimized_loss

                    # update previous weight momenta / variances
                    self.layers[i].weight_momenta = weight_momenta
                    self.layers[i].weight_variances = weight_variances

                    # update biases
                    grad_bias = np.mean(delta_j, axis=1)
                    bias_momenta = self.layers[i].bias_momenta
                    bias_variances = self.layers[i].bias_variances
                    optimized_delta, bias_momenta, bias_variances = self.adam_optimize(epoch, grad_bias, bias_momenta, bias_variances)
                    self.layers[i].biases -= self.learning_rate * optimized_delta
                
                    # update previous bias momenta / variances
                    self.layers[i].bias_momenta = bias_momenta
                    self.layers[i].bias_variances = bias_variances

                    TOTAL_TIMER_backprop += time.time() - TIMER_backprop
                loss = self.BCE(y_pred, y_exp_i)
                losses.append(loss)
                TOTAL_TIMER_batch += time.time() - TIMER_batch
            
            # reshape Y_pred:
            Y_pred = np.array(Y_pred)
            Y_pred = np.concatenate(Y_pred, axis=-1)

            # calculate loss
            epoch_loss = np.mean(losses)
            epoch_losses[epoch] = epoch_loss

            # display epoch
            if display and epoch % 1 == 0:
                if verbose: print(f"EPOCH: {epoch + 1}    LOSS: {epoch_loss}    Delta: {epoch_loss - epoch_losses[epoch - 1] if epoch > 0 else 0:.4g}")
                else: print(f"EPOCH: {epoch + 1}    LOSS: {epoch_loss}    Delta: {epoch_loss - epoch_losses[epoch - 1] if epoch > 0 else 0:.4g}")
            
            TOTAL_TIMER_epoch += time.time() - TIMER_epoch
            print(f"Timer backprop: {TOTAL_TIMER_backprop:.4g}")
            print(f"Timer backprop1: {TOTAL_TIMER_backprop1:.4g}")
            print(f"Timer batch: {TOTAL_TIMER_batch:.4g}")
            print(f"Timer epoch: {TOTAL_TIMER_epoch:.4g}")
        return epoch_losses, np.array(Y_pred)