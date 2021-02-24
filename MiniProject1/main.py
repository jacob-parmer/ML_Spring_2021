from load_data import Data
from kNN import kNN
from perceptron import Perceptron
import time
import sys
import random as rd
import argparse
from pudb import set_trace

IRIS_FEATURES = 4
A4A_FEATURES = 14

def main(args):

    a4a = Data("a4a", A4A_FEATURES)
    a4a_testing = Data("a4a.t", A4A_FEATURES)
    iris = Data("iris.scale", IRIS_FEATURES)
    iris_testing = Data("iris.t", IRIS_FEATURES)


    if not args.algorithm:
        print("Please specify which ML Algorithm you would like to use with -a. Exiting...")
        return

    if args.algorithm == 'perceptron' or args.algorithm == 'Perceptron':

        # Can specify lrate and epochs with -l and -e
        if not args.lrate:
            lrate = 0.001
        else:
            lrate = args.lrate

        if not args.epochs:
            epochs = 1000
        else:
            epochs = args.epochs
        
        # --------------------------------------------------------------------- IRIS ----------------------------------------------------------------- #
        init_w_iris = [rd.randrange(1,3) for _ in range(IRIS_FEATURES)]
        
        p2 = Perceptron(iris_testing.x[0], init_w_iris, bias=1)
        p2.train_weights(iris.x, iris.y, lrate=lrate, epochs=epochs, verbose=args.verbose)

        iris_error = 0

        iris_start_time = time.time()

        for j in range(len(iris_testing.x)):
            p2.set_x(iris_testing.x[j])
            prediction = p2.predict()
            print(f"Prediction for {iris_testing.x[j]}: {prediction}. Recorded classification is {iris_testing.y[j]}")
    
        iris_total_time = time.time() - iris_start_time

        # --------------------------------------------------------------------- A4A ------------------------------------------------------------------ #
        init_w_a4a = [1 for _ in range(A4A_FEATURES)]

        p = Perceptron(a4a_testing.x[0], init_w_a4a, bias=1)
        p.train_weights(a4a.x, a4a.y, lrate=lrate, epochs=epochs, verbose=args.verbose)    
        
        a4a_error = 0

        a4a_start_time = time.time()

        for i in range(len(a4a_testing.x)):
            p.set_x(a4a_testing.x[i])
            prediction = p.predict()
            #print(f"Prediction for {a4a_testing.x[i]}: {prediction}. Recorded classification is {a4a_testing.y[i]}")
            if prediction == a4a_testing.y[i]:
                a4a_error += 1

        a4a_error = a4a_error / len(a4a_testing.y)
        a4a_total_time = time.time() - a4a_start_time

        print(f"Iris misclassification error: TEMP\na4a misclassification error: {a4a_error}\n")
        print(f"Iris classification time: {iris_total_time}\na4a classification time: {a4a_total_time}")


    elif args.algorithm == 'kNN' or args.algorithm == 'knn':

        if not args.k:
            k=5
        else:
            k = args.k

        if not args.distance:
            distance_metric = 'euclidean'
        else:
            distance_metric = args.distance
        
        # --------------------------------------------------------------------- IRIS ----------------------------------------------------------------- #
        iris_knn = kNN(iris.x, iris.y)
        iris_error = 0

        iris_start_time = time.time()

        for i in range(len(iris_testing.x)):
            y = iris_knn.classify(new_x=iris_testing.x[i], k=k, distance_metric=distance_metric, verbose=args.verbose)
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
            if y == a4a_testing.y[j]:
                a4a_error += 1

        a4a_error = a4a_error / len(a4a_testing.y)

        a4a_total_time = time.time() - a4a_start_time

        print(f"Iris misclassification error: {iris_error}\na4a misclassification error: {a4a_error}\n")
        print(f"Iris classification time: {iris_total_time}\na4a classification time: {a4a_total_time}")


    elif args.algorithm == 'decision' or args.algorithm == 'tree':
        print("temp")

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a4a and iris datasets using an introductory ML algorithm.")
    parser.add_argument('--algorithm', '-a', type=str, help='select algorithm name for processing (str)')
    parser.add_argument('--lrate', '-l', type=float, help='select learning rate for perceptron (float)')
    parser.add_argument('--epochs', '-e', type=int, help='select number of epochs for perceptron (int)')
    parser.add_argument('--k', '-k', type=int, help='select k for k nearest neighbors (int)')
    parser.add_argument('--distance', '-d', type=str, help='select distance metric for knn (str)')
    parser.add_argument('--verbose', '-v', help='show processing information', action='store_true')

    args = parser.parse_args()
    main(args)
