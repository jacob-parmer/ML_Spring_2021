class Perceptron:

    def __init__(self, x, init_w, bias=0):
        self.x = x
        self.w = init_w
        self.bias = bias

    def predict(self, x=None, w=None, bias=0):

        # Function supports calling on the object, by leaving x and w empty, as well as calling on a specific set of
        # x and w by specifying values.
        if x != None and w != None:
            activation = -bias
            for i in range(len(x)):
                activation += w[i] * x[i]

        else:
            activation = -self.bias
            for i in range(len(self.x)):
                activation += self.w[i] * self.x[i]

        return 1.0 if activation >= 0.0 else -1.0

    def train_weights(self, training_data, training_labels, lrate=0.001, epochs=100, verbose=False):

        for epoch in range(epochs):
            sum_error = 0.0
            for i in range(len(training_data)):
            
                prediction = self.predict(x=training_data[i], w=self.w, bias=self.bias) 
                error = training_labels[i] - prediction
                sum_error += error**2

                self.bias = self.bias + (lrate * error)
                for j in range(len(self.w)):
                    self.w[j] = self.w[j] + (lrate * error * training_data[i][j])

            #if verbose:
                #print(f">epoch={epoch}, lrate={lrate}, error={sum_error}")

        if verbose:
            print(f"Final weights: {self.w}")
            print(f"Final bias: {self.bias}")

    def set_x(self, x):
        self.x = x

