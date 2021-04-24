import numpy as np
import pandas as pd
import random

from pudb import set_trace

class SVM:

    def __init__(self, X, y, epochs=1000, lrate=0.001):
        self.W = np.zeros(len(X[0])) # +1 is the bias
        self.X = X
        self.y = y

        self.epochs = epochs
        self.lrate = lrate
        return

    def predict(self, X_i):
        prediction = 0
        
        for j in range(len(X_i)):
            prediction += X_i[j] * self.W[j]

        return round((prediction),1)

    def train(self, verbose=False):

        if verbose:
            print(f"Beginning SVM model training...")

        order = np.arange(0, len(self.X), 1)

        avg_accuracy = 0
        for epoch in range(1, self.epochs+1):

            accuracy = 0

            eta = 1 / (self.lrate * epoch)
            fac = (1 - (eta*self.lrate))*self.W
            random.shuffle(order)
            for i in order:

                prediction = self.predict(self.X[i])

                if (self.y[i]*prediction < 1):
                    self.W = fac + eta * self.y[i] * self.X[i]
                    
                else:
                    self.W = fac
                    accuracy += 1

            accuracy = accuracy / len(order)
            avg_accuracy += accuracy
        
            if verbose:
                print(f">epoch: {epoch}. Accuracy: {accuracy}")

        avg_accuracy = avg_accuracy / self.epochs
        if verbose:
            print(f"Training Accuracy: {avg_accuracy}")

        return

    def test(self, X, y, verbose=False):

        error = 0
        for i in range(len(X)):
            prediction = self.predict(self.X[i])

            if (prediction != self.y[i]):
                error += 1

        if verbose:
            print(f"Error: {error}")

    