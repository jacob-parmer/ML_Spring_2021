from load_data import Data
from kNN import kNN
from perceptron import Perceptron
from decision_tree import DecisionTree
from display import Display
import time
import sys
import random as rd
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pudb import set_trace

IRIS_FEATURES = 4
A4A_FEATURES = 14

def main(args):

    a4a = Data("a4a", A4A_FEATURES)
    a4a_testing = Data("a4a.t", A4A_FEATURES)
    iris = Data("iris.scale", IRIS_FEATURES)
    iris_testing = Data("iris.t", IRIS_FEATURES)

    if not args.algorithm:
        raise AssertionError("Please specify which ML Algorithm you would like to use with -a. Exiting...")

    if args.algorithm == 'perceptron' or args.algorithm == 'Perceptron':

        # Can specify max_lrate and max_epochs with -l and -e
        if not args.lrate:
            max_lrate = 0.1
        else:
            max_lrate = args.lrate

        if not args.epochs:
            epochs = 1000
        else:
            epochs = args.epochs

        if args.verbose:
            print("Beginning perceptron categorization... \n")
            
        for lrate in np.arange(0, max_lrate, 0.001):

            # --------------------------------------------------------------------- IRIS ----------------------------------------------------------------- #
            init_w_iris = [1 for _ in range(IRIS_FEATURES)]
            
            if args.verbose:
                print("\nIRIS:\n")
            """
                Perceptron makes two passes for multi-classification.
                First pass sets datapoints with labels 2 or 3 as -1, and classifies a point as either 1, or 2/3.
                Second pass distinguishes between 2 and 3 by comparing them alone.
                This is kind of messy tbh.
            """
            iris_y_second_pass = []
            iris_x_second_pass = []
            for i in range(len(iris.y)):
                if iris.y[i] == 1:
                    continue        
                elif iris.y[i] == 2:
                    iris.y[i] = -1
                    iris_x_second_pass.append(iris.x[i])
                    iris_y_second_pass.append(-1)
                elif iris.y[i] == 3:
                    iris.y[i] = -1
                    iris_x_second_pass.append(iris.x[i])
                    iris_y_second_pass.append(1)

            p2 = Perceptron(iris_testing.x[0], init_w_iris, bias=1)
            p2.train_weights(iris.x, iris.y, lrate=lrate, epochs=epochs, verbose=args.verbose)

            p3 = Perceptron(iris_testing.x[0], init_w_iris, bias=1)
            p3.train_weights(iris_x_second_pass, iris_y_second_pass, lrate=lrate, epochs=epochs, verbose=args.verbose)

            iris_error = 0

            iris_start_time = time.time()

            for j in range(len(iris_testing.x)):
                p2.set_x(iris_testing.x[j])
                prediction = p2.predict()
                if args.verbose:
                    print(f"Prediction for {iris_testing.x[j]}: {prediction}. Recorded classification is {iris_testing.y[j]}")

                if prediction == 1 and iris_testing.y[j] == 1:
                    iris_error += 1
                elif prediction == -1 and (iris_testing.y[j] == 2 or iris_testing.y[j] == 3):
                    iris_error += 1 

            for k in range(len(iris_testing.x)):
                if iris_testing.y[k] != 1:
                    p3.set_x(iris_testing.x[k])
                    prediction = p3.predict()
                    if args.verbose:
                        print(f"Prediction for {iris_testing.x[k]}: {prediction}. Recorded classification is {iris_testing.y[k]}")

                    if iris_testing.y[k] == 2 and prediction == -1:
                        iris_error += 1
                    elif iris_testing.y[k] == 3 and prediction == 1:
                        iris_error += 1

            iris_error = iris_error / ( len(iris_testing.y) + 10 )


            iris_total_time = time.time() - iris_start_time

            # --------------------------------------------------------------------- A4A ------------------------------------------------------------------ #
            init_w_a4a = [1 for _ in range(A4A_FEATURES)]

            if args.verbose:
                print("\nA4A:\n")

            p = Perceptron(a4a_testing.x[0], init_w_a4a, bias=1)
            p.train_weights(a4a.x, a4a.y, lrate=lrate, epochs=epochs, verbose=args.verbose)    
            
            a4a_error = 0

            a4a_start_time = time.time()

            for i in range(len(a4a_testing.x)):
                p.set_x(a4a_testing.x[i])
                prediction = p.predict()
                #if args.verbose:
                    #print(f"Prediction for {a4a_testing.x[i]}: {prediction}. Recorded classification is {a4a_testing.y[i]}")
                
                if prediction == a4a_testing.y[i]:
                    a4a_error += 1

            a4a_error = a4a_error / len(a4a_testing.y)
            a4a_total_time = time.time() - a4a_start_time

            if args.verbose:
                print(f"Iris misclassification error: {iris_error}\na4a misclassification error: {a4a_error}\n")
                print(f"Iris classification time: {iris_total_time}\na4a classification time: {a4a_total_time}")

    elif args.algorithm == 'kNN' or args.algorithm == 'knn':

        if args.verbose:
            print("Beginning k-Nearest Neighbors categorization... \n")

        # can specify k and distance with -k and -d
        if not args.k:
            max_k=25
        else:
            max_k = args.k

        if not args.distance:
            distance_metric = 'euclidean'
        else:
            distance_metric = args.distance

        for k in range(1, max_k):

            # --------------------------------------------------------------------- IRIS ----------------------------------------------------------------- #
            iris_knn = kNN(iris.x, iris.y)
            iris_error = 0

            iris_start_time = time.time()

            for i in range(len(iris_testing.x)):
                y = iris_knn.classify(new_x=iris_testing.x[i], k=k, distance_metric=distance_metric, verbose=args.verbose)
                if args.verbose:
                    print(f"Prediction for {iris_testing.x[i]}: {y}. Recorded classification is {iris_testing.y[i]}")

                if y == iris_testing.y[i]:
                    iris_error += 1

            iris_error = iris_error / len(iris_testing.y)

            iris_total_time = time.time() - iris_start_time

            # --------------------------------------------------------------------- A4A ------------------------------------------------------------------ #
            
            a4a_knn = kNN(a4a.x, a4a.y)
            a4a_error = 0

            a4a_start_time = time.time()

            for j in range(len(a4a_testing.x)):
                y = a4a_knn.classify(new_x=a4a_testing.x[j], k=k, distance_metric=distance_metric, verbose=args.verbose)
                if args.verbose:
                    print(f"Prediction for {a4a_testing.x[j]}: {y}. Recorded classification is {a4a_testing.y[j]}")

                if y == a4a_testing.y[j]:
                    a4a_error += 1

            a4a_error = a4a_error / len(a4a_testing.y)

            a4a_total_time = time.time() - a4a_start_time

            if args.verbose:
                print(f"Iris misclassification error: {iris_error}\na4a misclassification error: {a4a_error}\n")
                print(f"Iris classification time: {iris_total_time}\na4a classification time: {a4a_total_time}")

    elif args.algorithm == 'decision' or args.algorithm == 'tree' or args.algorithm == 'decision_tree':
        
        # --------------------------------------------------------------------- IRIS ----------------------------------------------------------------- #
        iris_dt = DecisionTree(iris.x, iris.y)

        #left_x, left_y, right_x, right_y = iris_dt.split(0, 0)
    

        # --------------------------------------------------------------------- A4A ------------------------------------------------------------------ #
        a4a_dt = DecisionTree(a4a.x, a4a.y)
        

        print("This part made optional, therefore not implemented for the sake of time. ")

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a4a and iris datasets using an introductory ML algorithm.")
    parser.add_argument('--algorithm', '-a', type=str, help='select algorithm name for processing (str)')
    parser.add_argument('--lrate', '-l', type=float, help='select max learning rate for perceptron (float)')
    parser.add_argument('--epochs', '-e', type=int, help='select max number of epochs for perceptron (int)')
    parser.add_argument('--k', '-k', type=int, help='select k for k nearest neighbors (int)')
    parser.add_argument('--distance', '-d', type=str, help='select distance metric for knn (str)')
    parser.add_argument('--verbose', '-v', help='show processing information', action='store_true')

    args = parser.parse_args()
    main(args)
