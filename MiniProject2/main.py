import numpy as np
import pandas as pd
import argparse
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
import sys
from pudb import set_trace

from src.kernel_perceptron import KernelPerceptron

np.set_printoptions(threshold=sys.maxsize)

def main(args):
    
    # LOAD DATA
    path = os.getcwd() + "/data/mnist.scale.bz2"
    x, y = load_svmlight_file(path)
    x = x.toarray()
    y = y.astype(int)

    for i in range(len(y)):
        if y[i] == 0:
            y[i] = 1
        else:
            y[i] = -1

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.05)
    kp = KernelPerceptron(X_test, y_test, kernel=KernelPerceptron.polynomial_kernel)

    #set_trace()

    kp.fit(stepsize=1, epochs=1000, verbose=args.verbose)

    for i in range(len(X_test)):
        print(f"Prediction {kp.predict(X_test[i])} . Actual: {y_test[i]}")




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Mini-project 2")
    parser.add_argument('--verbose', '-v', help='show processing information', action='store_true')

    args = parser.parse_args()
    main(args)