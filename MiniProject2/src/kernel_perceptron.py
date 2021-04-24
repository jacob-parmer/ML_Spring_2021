import numpy as np

class KernelPerceptron:

    def __init__(self, x, y, kernel, bias=0):
        self.x = x
        self.y = y
        self.kernel = kernel
        self.alpha = np.zeros(x.shape[0])
        self.bias = bias
        pass

    def polynomial_kernel(self, x, z, a=1, b=1, d=2):
        x = np.transpose(x)
        return (a + (b * (np.dot(x,z))))**d
    
    def predict(self, x_i):
        
        sum_out = 0
        for n in range(len(self.x)):
            sum_out += self.alpha[n] * self.kernel(self, x=x_i, z=self.x[n])

        return 1 if sum_out > 0 else -1
    
    def fit(self, stepsize=1, epochs=100, verbose=False):

        x_len = len(self.x)
        for epoch in range(epochs):
            sum_error = 0.0
            accuracy = 0
            for i in range(x_len):
                K = np.zeros(x_len)
                for j in range(x_len):
                    K[j] = self.alpha[j] * self.kernel(self, x=self.x[j], z=self.x[i])

                if np.sum(K) > self.bias:
                    prediction = 1
                else:
                    prediction = -1

                error = self.y[i] - prediction
                sum_error += error**2

                if error != 0:
                    self.alpha[i] = self.alpha[i] + (stepsize * self.y[i])
                else:
                    accuracy += 1
                
                if verbose:
                    print(f"epoch {epoch+1}/{epochs} - #{i}/{x_len}")
                    print(f"Prediction: {prediction}, Actual: {self.y[i]}")

            accuracy = accuracy / x_len
            if verbose:
                print(f">epoch={epoch}, stepsize={stepsize}, error={sum_error}, accuracy={accuracy}")

        if verbose:
            print(f"Final weights: {self.alpha}")
            print(f"Final bias: {self.bias}")
