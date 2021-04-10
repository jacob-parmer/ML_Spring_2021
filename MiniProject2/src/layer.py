import numpy as np
import math

class Layer:

    def __init__(self, input_size, n_neurons, act_type="ReLU"):
        self.weights = 0.1 * np.random.randn(input_size, n_neurons)
        self.size = n_neurons
        self.biases = np.zeros((1, n_neurons))
        self.act_type = act_type

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs
        self._activation()

    def backward(self, outputs, targets, is_output_layer, lrate):
        
        errors = list()
        if is_output_layer:
            for i in range(len(outputs)):
                errors.append((targets[i] - outputs[i]) * self._activation_derivative(outputs[i]))
        else:
            for i in range(len(self.weights[0])):
                error = 0.0
                for j in range(len(outputs)):
                    error += (self.weights[i][j] * outputs[j]) * self._activation_derivative(outputs[j])
                errors.append(error)

        for i in range(len(self.weights)):
            for j in range(len(self.weights[0])):
                self.weights[i][j] = self.weights[i][j] + lrate * errors[j] * self.inputs[i]

        return errors

    # --------------- PRIVATE MEMBERS ----------------- #
    def _activation(self):
        if self.act_type == "ReLU":
            self.output = self._ReLU()
        elif self.act_type == "sigmoid":
            self.output = self._sigmoid()
        else:
            raise ValueError(f"Invalid activation function selected. {self.act_type} not supported.")

    def _step(self):
        return 1 if self.output > 0 else 0

    def _ReLU(self):
        temp_out = [0 for _ in range(len(self.output.T))]
        for i in range(len(self.output[0])):
            temp_out[i] = self.output[0][i] if self.output[0][i] > 0 else 0

        return temp_out

    def _sigmoid(self):
        temp_out = [0 for _ in range(len(self.output.T))]
        for i in range(len(self.output[0])):
            temp_out[i] = 1 / (1 + math.exp(-self.output[0][i]))

        return temp_out

    def _activation_derivative(self, output):
        if self.act_type == "ReLU":
            return self._dReLU(output)
        elif self.act_type == "sigmoid":
            return self._dSigmoid(output)
        else:
            raise ValueError(f"Invalid activation function selected. {self.act_type} not supported.")

    def _dReLU(self, output):
        return 1 if output > 0 else 0

    def _dSigmoid(self, output):
        return output * (1.0 - output)

