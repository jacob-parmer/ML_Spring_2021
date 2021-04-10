import numpy as np
import pandas as pd
import argparse
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
import sys
import builtins
from pudb import set_trace

from src.kernel_perceptron import KernelPerceptron
from src.neural_network import NeuralNetwork
from src.time_logs import TimerLog

#np.set_printoptions(threshold=sys.maxsize)

# Displays time in program before every print statement
_print = print
stopwatch = TimerLog()

# ------------- HELPER FUNCTIONS --------------- #
def timed_print(*args):
    print_str = ""
    for arg in args:
        print_str += str(arg) + " "

    _print(f"{stopwatch.get_elapsed()}\t| {print_str}")

# ---------- MAIN PROGRAM EXECUTION ----------- #
def main(args):
    
    if args.verbose:
        print(f"Beginning program execution in mode {args.mode}")
        print(f"-------------------------------------------")

    if args.timed:
        builtins.print = timed_print

    # LOAD DATA
    if args.verbose:
        print("Loading data...")

    path = os.getcwd() + "/data/mnist.scale.bz2"
    X, y = load_svmlight_file(path)
    X = X.toarray()
    y = y.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    if args.verbose:
        print(f"Finished loading data")

    if args.mode == "kp":
        # Multi-classification set to binary classification
        for i in range(len(y)):
            if y[i] == 0:
                y[i] = 1
            else:
                y[i] = -1

        kp = KernelPerceptron(X_train, y_train, kernel=KernelPerceptron.polynomial_kernel)

        # FIT DATA
        kp.fit(stepsize=1, epochs=20, verbose=args.verbose)

        # TEST DATA
        for i in range(len(X_test)):
            print(f"Prediction {kp.predict(X_test[i])} . Actual: {y_test[i]}")

    elif args.mode == "SVM":
        print("Not yet implemented")

    elif args.mode == "NN":
        
        # Encodes output data to one-hot arrays for multi-classification
        np_y = np.array(y_train)
        new_y_train = np.zeros((np_y.size, np_y.max() + 1))
        new_y_train[np.arange(np_y.size), np_y] = 1

        nn = NeuralNetwork(len(X_train[0]), n_hidden_layers=1, hidden_size=100, n_outputs=10, act_type="sigmoid", verbose=args.verbose)
        nn.train(X_train, new_y_train, lrate=args.lrate)
        nn.test(X_test, y_test)

    else:
        raise ValueError(f"Mode {args.mode} not supported. Use argument -m 'kp', 'SVM', or 'NN'.")




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Mini-project 2")
    parser.add_argument('--verbose', '-v', help='show processing information', action='store_true')
    parser.add_argument('--mode', '-m', default=None, help="Select algorithm to run for program")
    parser.add_argument('--timed', '-t', help="Display timing information", action='store_true')
    parser.add_argument('--lrate', '-l', default=0.01, help="Sets learning rate")

    args = parser.parse_args()
    main(args)