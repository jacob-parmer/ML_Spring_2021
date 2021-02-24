

class Data:

    def __init__(self, filename, num_features):

        if filename == "a4a" or filename == "a4a.t":
            data_index = 0
        elif filename == "iris.scale" or filename== "iris.t":
            data_index = 1
        else:
            print("Other datasets not supported because the data format is inconsistent for this project. >:(")

        data_file = open(f"data/{filename}", 'r')
        classifiers = []
        data = []
        for line in data_file:
            split_data = line.split()
            line_data = []
            for i in range(1, len(split_data)):
                temp = split_data[i].split(":")
                line_data.append(float(temp[data_index]))

            # Removes data that sometimes doesn't contain the same number of features as the rest?
            if len(line_data) == num_features:
                data.append(line_data)
                classifiers.append(float(split_data[0]))

        data_file.close()

        self.x = data
        self.y = classifiers

        return

