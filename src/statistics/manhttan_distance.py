

def manhattan_distance(x, y):
    """
    Calculate the Manhattan distance between a single sample `x` and multiple samples in `y`.

    :param x: A single sample, represented as a list or tuple of numbers.
    :param y: Multiple samples, represented as a list of lists or tuples of numbers.
    :return: An array containing the distances between `x` and the various samples in `y`.
    """
    distances = []

    for sample in y:
        #Ensure the dimensions of `x` and `sample` are the same
        if len(x) != len(sample):
            raise ValueError("Both samples must have the same number of dimensions.")

        #Calculate Manhattan distance for the current sample
        distance = sum(abs(x_i - y_i) for x_i, y_i in zip(x, sample))
        distances.append(distance)

    return distances

if __name__ == '__main__':
    from SIB.src.statistics.manhattan_distance import manhattan_distance
# Example usage:
x = [1, 2, 3]
y = [[4, 5, 6], [7, 8, 9], [10, 11, 12]]

distances = manhattan_distance(x, y)
print(distances)  # Output: [9, 27, 45]
