# Day 05 – Decision Trees: Entropy & Gini (Notes)

## Decision Tree Overview
A **Decision Tree** is a supervised learning algorithm used for **classification** and **prediction**.  
It works by **recursively splitting data** based on feature values to reduce uncertainty.

### Key Components
- **Root Node**: Represents the entire dataset
- **Internal Nodes**: Decision points based on feature tests
- **Branches**: Outcomes of decisions
- **Leaf Nodes**: Final prediction or class label

---

## Impurity Measures
Decision Trees decide splits using **impurity metrics**.  
Lower impurity ⇒ more homogeneous node.

The two most common measures are:
- **Entropy**
- **Gini Impurity**

---

## Entropy
Entropy measures **uncertainty or randomness** in class distribution.

- High entropy → mixed classes (impure)
- Low entropy → single dominant class (pure)

### Formula
$Entropy = - \sum_{i=1}^{n} p_i \log_2(p_i)$

For binary classification:
$Entropy = -p_1 \log_2 p_1 - p_2 \log_2 p_2$

### Calculator Trick (Casio fx-82MS)
Casio does not support $\log_2$ directly.

Use change of base:

$\log_2 n = \frac{\log n}{\log 2}$

So entropy becomes:

$-p_1 \times \frac{\log p_1}{\log 2} - p_2 \times \frac{\log p_2}{\log 2}$

---

## Gini Impurity
Gini Impurity measures the **probability of misclassification** if labels are assigned randomly according to class probabilities.

### Formula
$Gini = 1 - \sum_{i=1}^{n} p_i^2$

For binary classification:

$Gini = 1 - p_1^2 - p_2^2$

- Gini = 0 → perfectly pure node
- Higher Gini → more impurity

---

## Split Selection Strategy
Decision Trees follow a **greedy approach**:

1. Evaluate all possible splits at a node
2. Compute impurity for child nodes
3. Compute **weighted impurity**
4. Choose split with **lowest impurity** (or highest information gain)

### Weighted Gini
$G_{total} = \sum_{k} \frac{N_k}{N} \times G_k$

---

## Continuous Attributes
For numerical features (e.g., Age):

1. Sort values in ascending order
2. Compute midpoints between adjacent values
3. Treat each midpoint as a potential split
4. Compute impurity for each split
5. Choose split with lowest impurity

---

## Algorithms That Use These Concepts
These impurity calculations are core to many tree algorithms:

- **ID3** – Entropy & Information Gain
- **C4.5** – Gain Ratio, supports continuous features
- **CART** – Gini Impurity, binary splits
- **Hunt’s Algorithm** – Conceptual foundation of tree induction

> This project focuses on **node-level decision logic**, not full tree construction.

---

## Why Gini vs Entropy?
- Entropy is more sensitive to changes in probabilities
- Gini is computationally simpler
- In practice, both often choose the same split

---

## Limitations of Decision Trees
- Prone to **overfitting**
- Sensitive to small changes in data
- Greedy (not globally optimal)

---

## Key Takeaways
- Decision Trees choose splits mathematically, not intuitively
- Impurity reduction drives tree growth
- Continuous features require threshold testing
- Interpretability is a major strength of trees

---
