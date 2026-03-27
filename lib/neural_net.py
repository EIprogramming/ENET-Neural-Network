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
                        if (self.layers[i].activation_method != "softmax"):
                            delta_j = self.BCE_prime(y_pred, y_exp_i) * self.layers[i].activation_derivative(raw_output)
                        else:
                            delta_j = self.grad_softmax_CE(y_pred, y_exp_i)
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
                    self.layers[i].biases -= self.learning_rate * delta_j
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
            elif momentum < 300:
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

                # backpropogation
                for i in reversed(range(len(self.layers))): # TODO: fix this weird reversed loop
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
                        output_k = X_i.T
                    else:
                        output_k = self.layers[i - 1].batch_outputs
                    big_delta_weight = momentum * -self.learning_rate * np.mean(output_k[:, np.newaxis, :] * delta_j[np.newaxis, :, :], axis = -1).T
                    self.layers[i].weights += big_delta_weight
                    self.layers[i].biases -= self.learning_rate * np.mean(delta_j, axis=1) # TODO: multiply by learning rate?
                loss = self.BCE(y_pred, y_exp_i)
                losses.append(loss)
            
            # reshape Y_pred:
            Y_pred = np.array(Y_pred)
            Y_pred = np.concatenate(Y_pred, axis=-1)

            # calculate loss
            epoch_loss = np.mean(losses)
            epoch_losses[epoch] = epoch_loss

            # process momentum
            greatest_momentum_since = 0
            if prev_loss < epoch_loss and prev_prev_loss < epoch_loss:
                greatest_momentum_since = momentum
                momentum_degree *= 0.75
                momentum = momentum_degree if momentum > 1 else momentum_degree
            else:
                momentum *= 1.02
            prev_prev_loss = prev_loss
            prev_loss = epoch_loss
           
            # display epoch
            if display and epoch % 1 == 0:
                if verbose: print(f"EPOCH: {epoch}    LOSS: {epoch_loss}    MOMENTUM: {momentum}    GM: {greatest_momentum_since}    X_i = {X_i} y_pred = {y_pred}")
                else: print(f"EPOCH: {epoch}    LOSS: {epoch_loss}    MOMENTUM: {momentum}")
        return epoch_losses, np.array(Y_pred)
