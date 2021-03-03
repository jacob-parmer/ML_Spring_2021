class DecisionTree:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def split(self, index, value):
        left_x, left_y, right_x, right_y = list(), list(), list(), list()

        for i in range(len(self.x)):
            if self.x[i][index] < value:
                left_x.append(self.x[i])
                left_y.append(self.y[i])
            else:
                right_x.append(self.x[i])
                right_y.append(self.y[i])

        return left_x, left_y, right_x, right_y

    def get_best_split(self):
        return