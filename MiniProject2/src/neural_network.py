import numpy as np
import math
from src.layer import Layer

class NeuralNetwork:

    def __init__(self, input_size, n_hidden_layers, hidden_size, n_outputs, act_type="ReLU", verbose=False):
        self.input_size = input_size
        self.n_hidden_layers = n_hidden_layers
        self.hidden_size = hidden_size
        self.n_outputs = n_outputs
        self.act_type = act_type
        self.verbose = verbose

        self._init_network()
        return

    def predict(self, inputs):

        layer_in = inputs
        for layer in self.network:
            layer.forward(layer_in)
            layer_in = layer.output

        outputs = layer_in

        return outputs

    def train(self, inputs, targets, lrate):

        if self.verbose:
            print(f"Beginning neural network model training...")

        dataset_size = len(targets)
        accuracy = 0
        for i, X in enumerate(inputs):
            if self.verbose:
                print(f"#{i}/{dataset_size}")

            output = self.predict(X)
            self._backward_prop(output, targets[i], lrate)

            output = np.argmax(output, axis=0)
            expected = np.argmax(targets[i], axis=0)

            if (output == expected):
                accuracy += 1

            if self.verbose:
                print(f"Prediction: {output}. Actual: {expected}. ")
                print(f"Accuracy: {accuracy}")

        accuracy = accuracy / len(inputs)
        return accuracy

    def test(self, inputs, targets):

        error_count = 0
        dataset_size = len(inputs)
        predictions = []
        
        for i, X in enumerate(inputs):
            if self.verbose:
                print(f"#{i}/{dataset_size}")

            predicted = self.predict(X)
            predicted = np.argmax(predicted, axis=0)
            expected = targets[i]

            predictions.append(predicted)

            if self.verbose:
                print(f"Prediction: {predicted}. Actual: {expected}.")
            
        print(f"Error rate on testing data: {error_count / len(inputs)}")

        return predictions


    # --------------- PRIVATE MEMBERS ----------------- #
    def _init_network(self):

        if self.verbose:
            print(f"Creating network with {self.n_hidden_layers} hidden layers of size {self.hidden_size}, and {self.n_outputs} outputs")
        
        self.network = []

        input_size = self.input_size
        # CREATE NETWORK
        for n in range(self.n_hidden_layers):
            self.network.append(Layer(input_size, self.hidden_size, act_type=self.act_type))
            input_size = self.network[n].size
            
        self.network.append(Layer(input_size, self.n_outputs, act_type=self.act_type))

        if self.verbose:
            print(f"Finished creating network")

        return

    def _backward_prop(self, output, target, lrate):

        # Reverts one-hot back to actual digits
        for i in reversed(range(len(self.network))):
            layer = self.network[i]
            if i != len(self.network)-1:
                error = layer.backward(error, target, is_output_layer=False, lrate=lrate)
            else:
                error = layer.backward(output, target, is_output_layer=True, lrate=lrate)

        return