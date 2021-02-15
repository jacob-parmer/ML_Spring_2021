import math
import collections

class kNN:

    def __init__(self, x, y):
        
        self.x = x
        self.y = y
        self.y_set = set(y) # Set of all possible y classifiers, e.g. {'+1', '-1'} for a4a

    def euclidean_distance(self, p, q):
        distance = 0.0

        for i in range(len(p)):
            distance += (float(p[i]) - float(q[i]))**2

        return math.sqrt(distance)        

    def manhattan_distance(self, p, q):
        distance = 0.0

        for i in range(len(p)):
            distance += abs(float(p[i]) - float(q[i]))

        return distance

    def classify(self, new_x, k, distance_metric, verbose=False):

        shortest_distances = [] # List of k shortest distances
        shortest_distances_index = collections.deque(maxlen=k) # Queue of the indexes of k closest neighbors
        
        for i, x in enumerate(self.x):

            if distance_metric == 'euclidean':
                distance = self.euclidean_distance(new_x, x)
            
            elif distance_metric == 'manhattan':
                distance = self.manhattan_distance(new_x, x)

            else:
                raise ValueError("Error: Invalid distance metric selected. Only euclidean and manhattan supported.")

            if i < k: # Initializes arrays
                shortest_distances.append(distance)
                shortest_distances_index.append(i)
            
            else: # Grabs only k shortest distances, and the indexes in self.x where they occur
                shortest_distances = sorted(shortest_distances, reverse=True)
                if distance < shortest_distances[0]:
                    shortest_distances[0] = distance
                    shortest_distances_index.append(i)
        if verbose:
            print("New x:\n", new_x)
            print("Closest points to this new x:")
            for j in shortest_distances_index:
                print(self.x[j], self.y[j])

        closest_classifiers = [self.y[index] for index in shortest_distances_index]

        max_classifier_count = 0
        most_likely_classifier = None
        for classifier in self.y_set:
            num_of_classifier = closest_classifiers.count(classifier)
            if num_of_classifier > max_classifier_count:
                max_classifier_count = num_of_classifier
                most_likely_classifier = classifier

        if verbose:
            print(f"Most likely classifier was {most_likely_classifier} with {max_classifier_count} out of {k} as closest neighbors")

        return most_likely_classifier

        return []
