# Day 06 - Random Forest Feature Importance Explainer

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Step 1: Load Dataset

# Breast cancer dataset is commonly used for classification examples
data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print("Dataset Shape:", X.shape)
print("Target Classes:", data.target_names)


# Step 2: Train-Test Split

# Splitting dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Step 3: Train Random Forest Model

# n_estimators = number of decision trees
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)


# Step 4: Make Predictions

y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")


# Step 5: Feature Importance

# Random Forest calculates importance based on
# how much each feature decreases impurity across trees
feature_importance = rf_model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": feature_importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Features:")
print(importance_df.head(10))


# Step 6: Visualization

# Plot feature importance

plt.figure(figsize=(10,6))

plt.barh(
    importance_df["Feature"][:10],
    importance_df["Importance"][:10]
)

print("Number of trees in forest:", len(rf_model.estimators_))
plt.xlabel("Feature Importance")
plt.title("Top 10 Feature Importances (Random Forest)")

plt.gca().invert_yaxis()

plt.show()