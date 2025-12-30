"""
Day 03 — K-Nearest Neighbors (KNN)
From scratch implementation (Classification + Regression)
"""

import math
from collections import Counter


def min_max_scale(X):
    """
    Normalize features to range [0, 1]
    """
    cols = list(zip(*X))
    min_vals = [min(col) for col in cols]
    max_vals = [max(col) for col in cols]

    scaled_X = []

    for row in X:
        scaled_row = []
        for i in range(len(row)):
            if max_vals[i] - min_vals[i] == 0:
                scaled_row.append(0)
            else:
                scaled_row.append(
                    (row[i] - min_vals[i]) / (max_vals[i] - min_vals[i])
                )
        scaled_X.append(scaled_row)

    return scaled_X, min_vals, max_vals


def scale_new_point(point, min_vals, max_vals):
    """
    Scale a new test point using training min/max
    """
    scaled = []

    for i in range(len(point)):
        if max_vals[i] - min_vals[i] == 0:
            scaled.append(0)
        else:
            scaled.append(
                (point[i] - min_vals[i]) / (max_vals[i] - min_vals[i])
            )

    return scaled


def euclidean_distance(p1, p2):
    """
    Euclidean distance between two points
    """
    return math.sqrt(
        sum((p1[i] - p2[i]) ** 2 for i in range(len(p1)))
    )


def knn_classify(X_train, y_train, x_test, k):
    """
    KNN classification using majority voting
    """
    distances = []

    for i in range(len(X_train)):
        d = euclidean_distance(X_train[i], x_test)
        distances.append((d, y_train[i]))

    distances.sort(key=lambda x: x[0])

    k_neighbors = distances[:k]
    labels = [label for _, label in k_neighbors]

    return Counter(labels).most_common(1)[0][0]


def knn_regression(X_train, y_train, x_test, k):
    """
    KNN regression using mean of neighbors
    """
    distances = []

    for i in range(len(X_train)):
        d = euclidean_distance(X_train[i], x_test)
        distances.append((d, y_train[i]))

    distances.sort(key=lambda x: x[0])
    k_neighbors = distances[:k]

    return sum(value for _, value in k_neighbors) / k


X = [
    [96, 214],
    [79, 195],
    [79, 99],
    [71, 96],
    [80, 113],
    [85, 104],
    [98, 100],
    [98, 155],
    [91, 107],
    [93, 137]
]

y = [
    "Action",
    "Action",
    "Action",
    "Kids",
    "Thriller",
    "Horror",
    "Horror",
    "Fantasy",
    "Kids",
    "Kids"
]

X_scaled, min_vals, max_vals = min_max_scale(X)

animal = [81, 204]
animal_scaled = scale_new_point(animal, min_vals, max_vals)

k = 3
prediction = knn_classify(X_scaled, y, animal_scaled, k)

print("Predicted Genre:", prediction)


X_reg = [[1], [2], [3], [4], [5]]
y_reg = [2, 4, 6, 8, 10]

X_reg_scaled, min_r, max_r = min_max_scale(X_reg)
test_point = scale_new_point([3.5], min_r, max_r)

print("Regression Prediction:", knn_regression(X_reg_scaled, y_reg, test_point, k=2))
