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
from src.SVM import SVM
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

def convert_to_ecoc(y):
    ecoc_y = list()
    for i in range(len(y)):
        new_y = []
        for j in '{0:04b}'.format(y[i]):
            if (int(j) == 0):
                new_y.append(-1)
            else:
                new_y.append(1)
        ecoc_y.append(new_y)

    ecoc_y = np.array(ecoc_y)
    return ecoc_y

def get_confusion_matrix_and_accuracy(predicted_values, actual_values):
    c_mtrx = np.zeros((10,10))
    accuracy = 0
    for i in predicted_values:
        for j in actual_values:
            c_mtrx[i][j] += i*j
            if i == j:
                accuracy += 1
    
    accuracy = accuracy / (len(predicted_values)**2)
    return c_mtrx, accuracy

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

    # ---------- KERNEL PERCEPTRON ----------- #
    if args.mode == "KP":
        # Multi-classification set to binary classification
        ecoc_y = list()
        for i in range(len(y)):
            ecoc_y.append([int(j) for j in '{0:04b}'.format(y[i])])

        kp = KernelPerceptron(X_train, y_train, kernel=KernelPerceptron.polynomial_kernel)

        # FIT DATA
        kp.fit(stepsize=1, epochs=20, verbose=args.verbose)

        # TEST DATA
        for i in range(len(X_test)):
            print(f"Prediction {kp.predict(X_test[i])} . Actual: {y_test[i]}")


    # ---------- SUPPORT VECTOR MACHINE ----------- #
    elif args.mode == "SVM":

        ecoc_y_train = convert_to_ecoc(y_train).T
        ecoc_y_test = convert_to_ecoc(y_test).T
        
        # FIT DATA
        SVMs = []
        for i in range(len(ecoc_y_train)):
            model = SVM(X_train, ecoc_y_train[i], epochs=200)
            model.train(verbose=args.verbose)
            SVMs.append(model)

        # TEST DATA
        predictions = []
        for j in range(len(X_test)):
            output = []
            for k in range(len(ecoc_y_test)):
                prediction = SVMs[k].predict(X_test[j])
                output.append(np.sign(prediction))

            predictions.append(output)

        # convert predictions back to decimals
        for l, pred in enumerate(predictions):
            for m in range(len(pred)):
                if pred[m] == -1:
                    pred[m] = 0
                else:
                    pred[m] = 1

            predictions[l] = int("".join(str(n) for n in pred),2)

            # Changes all classifications greater than 9 to 9
            if predictions[l] > 9:
                predictions[l] = 9

        print(predictions)
        c_mtrx, accuracy = get_confusion_matrix_and_accuracy(predictions, y_test)

        if args.verbose:
            print(f"Confusion Matrix: {c_mtrx}")
            print(f"Accuracy: {accuracy}")
        



    # ---------- NEURAL NETWORK ----------- #
    elif args.mode == "NN":
        
        # Encodes output data to one-hot arrays for multi-classification
        np_y = np.array(y_train)
        new_y_train = np.zeros((np_y.size, np_y.max() + 1))
        new_y_train[np.arange(np_y.size), np_y] = 1

        nn = NeuralNetwork(len(X_train[0]), n_hidden_layers=1, hidden_size=100, n_outputs=10, act_type="sigmoid", verbose=args.verbose)
        train_acc = nn.train(X_train, new_y_train, lrate=args.lrate)
        
        if args.verbose:
            print(f"Training Accuracy: {train_acc}")
        
        predictions = nn.test(X_test, y_test)

        c_mtrx, accuracy = get_confusion_matrix_and_accuracy(predictions, y_test)

        if args.verbose:
            print(f"Confusion Matrix: {c_mtrx}")
            print(f"Accuracy: {accuracy}")

    else:
        raise ValueError(f"Mode {args.mode} not supported. Use argument -m 'KP', 'SVM', or 'NN'.")




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Mini-project 2")
    parser.add_argument('--verbose', '-v', help='show processing information', action='store_true')
    parser.add_argument('--mode', '-m', default=None, help="Select algorithm to run for program")
    parser.add_argument('--timed', '-t', help="Display timing information", action='store_true')
    parser.add_argument('--lrate', '-l', default=0.01, help="Sets learning rate")

    args = parser.parse_args()
    main(args)