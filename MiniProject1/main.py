from load_data import Data
from kNN import kNN
from perceptron import Perceptron
import time
import sys

IRIS_FEATURES = 4
A4A_FEATURES = 14

def main():
    
    if not sys.argv[1:]:
        print("Please specify which ML Algorithm you would like to use. Exiting...")
        return

    if sys.argv[1:][0] == 'perceptron' or sys.argv[1:][0] == 'Perceptron':

        data = [[-2, 5, -4, 4], [-4, 3, 1, 2], [1, 5, -5, 3], [-3, 2, -2, 1],
                [3, -2, 2, -4], [-1, 1, 5, -5], [5, -3, 2, -4], [5, -4, -1, 1]]

        classifications = [1, 1, 1, 1, 0, 0, 0, 0]
        init_w = [1, 1, 1, 1]

        p = Perceptron(data[1], init_w, bias=1)

        p.train_weights(data, classifications, lrate=0.001, epochs=1000)

        print(p.predict())


    elif sys.argv[1:][0] == 'kNN' or sys.argv[1:][0] == 'knn':

        a4a = Data("a4a", A4A_FEATURES)
        a4a_testing = Data("a4a.t", A4A_FEATURES)
        iris = Data("iris.scale", IRIS_FEATURES)
        iris_testing = Data("iris.t", IRIS_FEATURES)

        a4a_knn = kNN(a4a.x, a4a.y)
        iris_knn = kNN(iris.x, iris.y)

        iris_error = 0
        a4a_error = 0

        k=5
        distance_metric = 'euclidean'
        verbose = False

        iris_start_time = time.time()

        for i in range(len(iris_testing.x)):
            y = iris_knn.classify(new_x=iris_testing.x[i], k=k, distance_metric=distance_metric, verbose=verbose)
            if y == iris_testing.y[i]:
                iris_error += 1

        iris_error = iris_error / len(iris_testing.y)

        iris_total_time = time.time() - iris_start_time
        a4a_start_time = time.time()

        for j in range(len(a4a_testing.x)):
            y = a4a_knn.classify(new_x=a4a_testing.x[j], k=k, distance_metric=distance_metric, verbose=verbose)
            if y == a4a_testing.y[j]:
                a4a_error += 1

        a4a_error = a4a_error / len(a4a_testing.y)

        a4a_total_time = time.time() - a4a_start_time

        print(f"Iris error: {iris_error}\na4a misclassification error: {a4a_error}\n")
        print(f"Iris classification time: {iris_total_time}\na4a classification time: {a4a_total_time}")

    return


if __name__ == "__main__":
    main()
