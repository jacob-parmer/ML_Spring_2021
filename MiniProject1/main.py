from load_data import Data
from kNN import kNN
import time

IRIS_FEATURES = 4
A4A_FEATURES = 14

def main():
    
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
