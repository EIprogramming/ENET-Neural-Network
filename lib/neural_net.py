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

    # TODO: create vectorized train based on X_i, y_exp_i (batch processing)
    def train(self, X: np.ndarray, y_exp: np.ndarray, epochs = 10, display=True, verbose=False):
        epoch_losses = np.zeros(epochs)
        momentum = 1
        momentum_degree = 1 # the running count of momentum
        prev_loss = 9999
        prev_prev_loss = 9999
        for epoch in range(epochs):
            losses = []
            direction = 1
            Y_pred = []
            for X_i, y_exp_i in zip(X, y_exp):
                y_pred = self.process(X_i)
                Y_pred.append(y_pred)

                # backpropogation
                for i in reversed(range(len(self.layers))): # TODO: fix this weird reversed loop
                    # for each layer, starting from the last, go through each node and calculate the deltas
                    
                    for j in range(len(self.layers[i].weights)):
                        raw_output_j = self.layers[i].raw_outputs[j]
                        if i == len(self.layers) - 1:
                            delta_j = self.BCE_prime(y_pred[j], y_exp_i[j]) * self.layers[i].activation_derivative(raw_output_j)
                            self.layers[i].deltas[j] = delta_j
                        else:
                            sum_delta_l = np.sum([delta_l for delta_l in self.layers[i + 1].deltas])
                            delta_j = sum_delta_l * self.layers[i].activation_derivative(raw_output_j)
                            self.layers[i].deltas[j] = delta_j
                        
                        for k in range(len(self.layers[i].weights[j])):
                            if i != 0:
                                output_k = self.layers[i - 1].outputs[k]
                            else:
                                output_k = X_i[k]
                            big_delta_weight = momentum * -self.learning_rate * output_k * delta_j
                            self.layers[i].weights[j, k] += big_delta_weight
                            self.layers[i].biases[j] -= delta_j
                        # big Delta = - learning_rate * node_output_i * delta_j
                        # if outer neuron: delta_j = BCE_prime * logistic_prime
                        # if inner neuron: delta_j = (sum of delta_l for every l in the layer size L) * (logistic_prime)
                        # what we need: a matrix of outputs to fill and a matrix of deltas to fill
                loss = self.BCE(y_pred, y_exp_i)
                #if (y_exp_i == 0 and epoch > 150 and epoch % 10 == 0 and y_pred > 0.5): print(y_pred, y_exp_i, loss)
                losses.append(loss)
            if (epoch == epochs - 1): print(self)
            epoch_loss = np.mean(losses)
            epoch_losses[epoch] = epoch_loss
            greatest_momentum_since = 0
            if prev_loss < epoch_loss and prev_prev_loss < epoch_loss:
                greatest_momentum_since = momentum
                momentum_degree *= 0.95
                momentum = 0.5 if momentum > 1 else momentum_degree
            else:
                momentum *= 1.02
            prev_prev_loss = prev_loss
            prev_loss = epoch_loss
            if display and epoch % 1 == 0:
                if verbose: print(f"EPOCH: {epoch}    LOSS: {epoch_loss}    MOMENTUM: {momentum}    GM: {greatest_momentum_since}    X_i = {X_i} y_pred = {y_pred}")
                else: print(f"EPOCH: {epoch}    LOSS: {epoch_loss}    MOMENTUM: {momentum}")
        return epoch_losses, np.array(Y_pred)
    
    def train_optimized(self, X: np.ndarray, y_exp: np.ndarray, epochs = 10, display=True, verbose=False):
        epoch_losses = np.zeros(epochs)
        momentum = 1
        momentum_degree = 1 # the running count of momentum
        prev_loss = 9999
        prev_prev_loss = 9999
        for epoch in range(epochs):
            losses = []
            direction = 1
            Y_pred = []
            for X_i, y_exp_i in zip(X, y_exp):
                y_pred = self.process(X_i)
                Y_pred.append(y_pred)

                # backpropogation
                for i in reversed(range(len(self.layers))): # TODO: fix this weird reversed loop
                    # for each layer, starting from the last, go through each node and calculate the deltas
                    #print("1")
                    raw_output = self.layers[i].raw_outputs
                    #print("2")
                    if i == len(self.layers) - 1:
                        delta_j = self.BCE_prime(y_pred, y_exp_i) * self.layers[i].activation_derivative(raw_output)
                        #print("3a")
                        self.layers[i].deltas = delta_j
                        #print("3b")
                    else:
                        sum_delta_l = np.sum(self.layers[i + 1].deltas)
                        #print("4a")
                        delta_j = sum_delta_l * self.layers[i].activation_derivative(raw_output)
                        #print("4b")
                        self.layers[i].deltas = delta_j
                        #print("4c")
                        
                    #print("5")
                    if i == 0:
                        output_k = X_i
                        #print("6a")
                    else:
                        output_k = self.layers[i - 1].outputs
                        #print("7a")
                    #print("8")
                    #print("delta vs output:", delta_j[:, np.newaxis].shape, output_k[:, np.newaxis].shape)
                    big_delta_weight = momentum * -self.learning_rate * (delta_j[:, np.newaxis] @ output_k[np.newaxis, :])
                    #print("9")
                    self.layers[i].weights += big_delta_weight
                    #print("10")
                    self.layers[i].biases -= delta_j
                    #print("11")
                loss = self.BCE(y_pred, y_exp_i)
                #if (y_exp_i == 0 and epoch > 150 and epoch % 10 == 0 and y_pred > 0.5): print(y_pred, y_exp_i, loss)
                losses.append(loss)
            if (epoch == epochs - 1): print(self)
            epoch_loss = np.mean(losses)
            epoch_losses[epoch] = epoch_loss
            greatest_momentum_since = 0
            if prev_loss < epoch_loss and prev_prev_loss < epoch_loss:
                greatest_momentum_since = momentum
                momentum_degree *= 0.95
                momentum = 0.5 if momentum > 1 else momentum_degree
            else:
                momentum *= 1.02
            prev_prev_loss = prev_loss
            prev_loss = epoch_loss
            if display and epoch % 1 == 0:
                if verbose: print(f"EPOCH: {epoch}    LOSS: {epoch_loss}    MOMENTUM: {momentum}    GM: {greatest_momentum_since}    X_i = {X_i} y_pred = {y_pred}")
                else: print(f"EPOCH: {epoch}    LOSS: {epoch_loss}    MOMENTUM: {momentum}")
        return epoch_losses, np.array(Y_pred)

    def batch_train_optimized(self, X: np.ndarray, y_exp: np.ndarray, epochs = 10, batch_size=32, display=True, verbose=False):
        epoch_losses = np.zeros(epochs)
        momentum = 1
        momentum_degree = 1 # the running count of momentum
        prev_loss = 9999
        prev_prev_loss = 9999
        for epoch in range(epochs):
            losses = []
            direction = 1
            Y_pred = []
            # TODO: shuffle the batches
            for batch_number in range(len(X)//batch_size):
                X_i = X[batch_size * batch_number : batch_size * (batch_number + 1)]
                y_exp_i = y_exp[batch_size * batch_number : batch_size * (batch_number + 1)].T
                y_pred = self.batch_eval(X_i)
                Y_pred.append(y_pred)
                #print(f"exp vs pred: {y_exp_i.shape}, {y_pred.shape}")

                # backpropogation
                for i in reversed(range(len(self.layers))): # TODO: fix this weird reversed loop
                    # for each layer, starting from the last, go through each node and calculate the deltas
                    #print("1")
                    raw_output = self.layers[i].batch_raw_outputs
                    #print("2")
                    if i == len(self.layers) - 1:
                        delta_j = self.BCE_prime(y_pred, y_exp_i) * self.layers[i].activation_derivative(raw_output)
                        #print("delta_j_size: ", delta_j.shape, self.BCE_prime(y_pred, y_exp_i).shape, "*",  self.layers[i].activation_derivative(raw_output).T.shape, y_pred.shape, y_exp_i.shape)
                        #print("3a")
                        self.layers[i].batch_deltas = delta_j
                        #print("3b")
                    else:
                        sum_delta_l = np.sum(self.layers[i + 1].batch_deltas)
                        #print("4a")
                        delta_j = sum_delta_l * self.layers[i].activation_derivative(raw_output)
                        #print("4b")
                        self.layers[i].batch_deltas = delta_j
                        #print("4c")
                        
                    #print("5")
                    if i == 0:
                        output_k = X_i.T
                        #print("6a")
                    else:
                        output_k = self.layers[i - 1].batch_outputs
                        #print("7a")
                    #print("8")
                    #print(f"{output_k.shape} * {delta_j.shape}")
                    #print(f"{output_k[:, np.newaxis, :].shape} * {delta_j[np.newaxis, :, :].shape}")
                    #print(f"result:{np.mean(output_k[:, np.newaxis, :] * delta_j[np.newaxis, :, :], axis = -1).shape}")
                    big_delta_weight = momentum * -self.learning_rate * np.mean(output_k[:, np.newaxis, :] * delta_j[np.newaxis, :, :], axis = -1).T

                    #print("9")
                    #print(f"weights: ", self.layers[i].weights.shape)
                    self.layers[i].weights += big_delta_weight
                    #print("10")
                    #print("delta vs output", delta_j.shape, output_k.shape)
                    self.layers[i].biases -= np.mean(delta_j, axis=1)
                    #print("11")
                loss = self.BCE(y_pred, y_exp_i)
                #if (y_exp_i == 0 and epoch > 150 and epoch % 10 == 0 and y_pred > 0.5): print(y_pred, y_exp_i, loss)
                losses.append(loss)
            if (epoch == epochs - 1): print(self)
            epoch_loss = np.mean(losses)
            epoch_losses[epoch] = epoch_loss
            greatest_momentum_since = 0
            if prev_loss < epoch_loss and prev_prev_loss < epoch_loss:
                greatest_momentum_since = momentum
                momentum_degree *= 0.95
                momentum /= 0.5 if momentum > 1 else momentum_degree
            else:
                momentum *= 1.02
            prev_prev_loss = prev_loss
            prev_loss = epoch_loss
            if display and epoch % 1 == 0:
                if verbose: print(f"EPOCH: {epoch}    LOSS: {epoch_loss}    MOMENTUM: {momentum}    GM: {greatest_momentum_since}    X_i = {X_i} y_pred = {y_pred}")
                else: print(f"EPOCH: {epoch}    LOSS: {epoch_loss}    MOMENTUM: {momentum}")
        return epoch_losses, np.array(Y_pred)

    def batch_train(self, X: np.ndarray, y_exp: np.ndarray, epochs = 10, batch_size=32, display=True, verbose=False):
        epoch_losses = np.zeros(epochs)
        momentum = 1
        momentum_degree = 1 # the running count of momentum
        prev_loss = 9999
        prev_prev_loss = 9999
        for epoch in range(epochs):
            losses = []
            direction = 1
            Y_pred = []
            for i in range(len(X)//batch_size):
                # variables to look out for: X_i, y_pred, Y_pred, y_exp_i
                X_batch = X[batch_size*i:batch_size*(i+1)]
                y_exp_batch = y_exp[batch_size*i:batch_size*(i+1)]
                y_pred = self.batch_eval(X_batch)
                # backpropogation
                for i in reversed(range(len(self.layers))): # TODO: fix this weird reversed loop
                    # for each layer, starting from the last, go through each node and calculate the deltas
                    
                    for j in range(len(self.layers[i].weights)):
                        # initialize batch deltas
                        if len(self.layers[i].batch_deltas) == 0:
                            self.layers[i].batch_deltas = np.zeros((X_batch.shape[0], len(self.layers[i].weights)))
                        raw_output_j = self.layers[i].batch_raw_outputs[j]
                        if i == len(self.layers) - 1:
                            # TODO: make individual multidimensional deltas for batch training to multiply by output k
                            delta_j: np.ndarray = self.BCE_prime(y_pred[:, j], y_exp_batch[:, j]) * self.layers[i].activation_derivative(raw_output_j)
                            self.layers[i].batch_deltas[:, j] = delta_j
                        else:
                            sum_delta_l = np.sum([delta_l for delta_l in self.layers[i + 1].batch_deltas], axis=1)
                            delta_j: np.ndarray = sum_delta_l * self.layers[i].activation_derivative(raw_output_j)
                            self.layers[i].batch_deltas[:, j] = delta_j
                        for k in range(len(self.layers[i].weights[j])):
                            if i != 0:
                                output_k = self.layers[i - 1].batch_outputs[k]
                            else:
                                output_k = X_batch[:, k]
                            big_delta_weight =momentum * -self.learning_rate * np.mean(output_k * delta_j)
                            self.layers[i].weights[j, k] += big_delta_weight
                            self.layers[i].biases[j] -= np.mean(delta_j)
                    
                loss = self.BCE(y_pred, y_exp_batch)
                #if (y_exp_i == 0 and epoch > 150 and epoch % 10 == 0 and y_pred > 0.5): print(y_pred, y_exp_i, loss)
                losses.append(loss)
                Y_pred.append(y_pred)
            # end of old non-batch NOTE
            #if (epoch == epochs - 1): print(self)
            epoch_loss = np.mean(losses)
            epoch_losses[epoch] = epoch_loss
            greatest_momentum_since = 0
            if prev_loss < epoch_loss and prev_prev_loss < epoch_loss:
                greatest_momentum_since = momentum
                momentum_degree *= 0.95
                momentum = 0.5 if momentum > 1 else momentum_degree
            else:
                momentum *= 1.02
            prev_prev_loss = prev_loss
            prev_loss = epoch_loss
            if display and epoch % 1 == 0:
                if verbose: print(f"EPOCH: {epoch}    LOSS: {epoch_loss}    MOMENTUM: {momentum}    GM: {greatest_momentum_since}    X_i = {X[-1]} y_pred = {y_pred[-1]}")
                else: print(f"EPOCH: {epoch}    LOSS: {epoch_loss}    MOMENTUM: {momentum}")
        return epoch_losses, y_pred
                