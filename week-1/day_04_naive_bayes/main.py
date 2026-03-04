"""
Day 04 — Naive Bayes Classifier
From-scratch implementation for categorical data (Play Tennis example)

This script demonstrates:
- Prior probability computation
- Likelihood estimation
- Posterior probability calculation
- Final class prediction using Naive Bayes
"""

from collections import Counter

# Play Tennis dataset
# Each row represents:
# (Outlook, Temperature, Humidity, Wind, Class Label)
data = [
    ("Sunny", "Hot", "High", "Weak", "No"),
    ("Sunny", "Hot", "High", "Strong", "No"),
    ("Overcast", "Hot", "High", "Weak", "Yes"),
    ("Rain", "Mild", "High", "Weak", "Yes"),
    ("Rain", "Cool", "Normal", "Weak", "Yes"),
    ("Rain", "Cool", "Normal", "Strong", "No"),
    ("Overcast", "Cool", "Normal", "Strong", "Yes"),
    ("Sunny", "Mild", "High", "Weak", "No"),
    ("Sunny", "Cool", "Normal", "Weak", "Yes"),
    ("Rain", "Mild", "Normal", "Weak", "Yes"),
    ("Sunny", "Mild", "Normal", "Strong", "Yes"),
    ("Overcast", "Mild", "High", "Strong", "Yes"),
    ("Overcast", "Hot", "Normal", "Weak", "Yes"),
    ("Rain", "Mild", "High", "Strong", "No")
]

# Feature names (order matters and must match dataset tuples)
features = ["Outlook", "Temperature", "Humidity", "Wind"]


def compute_priors(data):
    """
    Computes prior probabilities for each class label.

    Prior Probability:
        P(Class) = (Number of samples in class) / (Total samples)

    :param data: Dataset containing feature values and class labels
    :return: Dictionary mapping class labels to their prior probabilities
    """
    total = len(data)
    class_counts = Counter(row[-1] for row in data)

    priors = {}
    for cls in class_counts:
        priors[cls] = class_counts[cls] / total

    return priors


# Calculate prior probabilities
priors = compute_priors(data)
print("Priors:", priors)


def compute_likelihoods(data):
    """
    Computes likelihood probabilities for each feature value given a class.

    Likelihood:
        P(Feature = value | Class)

    These probabilities are calculated using frequency counts
    from the training dataset.

    :param data: Dataset containing feature values and class labels
    :return: Nested dictionary of likelihood probabilities
    """
    likelihoods = {}
    class_counts = Counter(row[-1] for row in data)

    for cls in class_counts:
        likelihoods[cls] = {}
        class_rows = [row for row in data if row[-1] == cls]

        for i, feature in enumerate(features):
            likelihoods[cls][feature] = {}
            values = [row[i] for row in class_rows]
            value_counts = Counter(values)

            for value in value_counts:
                likelihoods[cls][feature][value] = (
                    value_counts[value] / class_counts[cls]
                )

    return likelihoods


# Calculate likelihood probabilities
likelihoods = compute_likelihoods(data)


def predict(sample, priors, likelihoods):
    """
    Predicts the class label for a given input sample using Naive Bayes.

    Posterior Probability (unnormalized):
        P(Class | X) ∝ P(Class) × Π P(feature_i | Class)

    Note:
    - No Laplace smoothing is applied
    - Zero probability may occur for unseen feature values

    :param sample: Dictionary of feature-value pairs for prediction
    :param priors: Prior probabilities of classes
    :param likelihoods: Likelihood probabilities of features given classes
    :return: Dictionary containing posterior probabilities for each class
    """
    probabilities = {}

    for cls in priors:
        prob = priors[cls]

        for feature, value in sample.items():
            if value in likelihoods[cls][feature]:
                prob *= likelihoods[cls][feature][value]
            else:
                prob *= 0  # unseen feature value (no smoothing)

        probabilities[cls] = prob

    return probabilities


# Sample to classify
sample = {
    "Outlook": "Sunny",
    "Temperature": "Cool",
    "Humidity": "High",
    "Wind": "Strong"
}

# Perform prediction
result = predict(sample, priors, likelihoods)
print("Posterior probabilities:", result)

# Final predicted class
prediction = max(result, key=result.get)
print("Prediction:", prediction)
