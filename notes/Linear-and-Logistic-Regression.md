# Notes — Linear & Logistic Regression (From Scratch)

This notes section documents the **core theory, mathematics, intuition, and differences** between **Linear Regression** and **Logistic Regression**, written alongside hands-on implementations from scratch.
The goal is **conceptual clarity first**, then code.

---

## 1. Linear Regression

### 1.1 What is Linear Regression?

Linear Regression is a **supervised learning algorithm** used to model the relationship between one or more independent variables and a **continuous dependent variable**.

It assumes a **linear relationship** between inputs and output.

Examples:

- Predicting house prices from area
- Predicting salary from years of experience
- Predicting temperature from time

---

### 1.2 Linear Regression Model

For a single feature:

$$
y = mx + c
$$

In machine learning notation:

$$
\hat{y} = w x + b
$$

Where:

- $ w $ → weight (slope)
- $ b $ → bias (intercept)
- $ x $ → input feature
- $ \hat{y} $ → predicted value

For multiple features:

$$
\hat{y} = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b
$$

---

### 1.3 Loss Function (Mean Squared Error)

Linear regression minimizes **Mean Squared Error (MSE)**:

$$
J(w, b) = \frac{1}{n} \sum (y - \hat{y})^2
$$

Why MSE?

- Penalizes large errors heavily
- Convex → guarantees a global minimum

---

### 1.4 Learning the Parameters

Parameters $ w $ and $ b $ are learned using:

- Gradient Descent
- Normal Equation (closed-form, when feasible)

Gradient descent updates:

$$
w := w - \alpha \frac{\partial J}{\partial w}
$$

$$
b := b - \alpha \frac{\partial J}{\partial b}
$$

---

### 1.5 Key Properties of Linear Regression

- Predicts **continuous values**
- Output is **unbounded**
- Decision boundary is **not meaningful**
- Sensitive to outliers
- Assumes linearity and homoscedasticity

---

## 2. Logistic Regression

### 2.1 What is Logistic Regression?

Logistic Regression is a **supervised classification algorithm** used to predict the **probability of a categorical outcome**.

Despite its name, logistic regression is **not a regression algorithm** in the traditional sense.

Examples:

- Pass / Fail
- Yes / No
- Fraud / Not Fraud

---

### 2.2 Logistic Regression Model

Logistic regression uses a **linear model** followed by a **sigmoid activation**:

$$
z = w x + b
$$

$$
\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

Where:

- $ \hat{y} $ represents **probability**
- $ 0 < \hat{y} < 1 $

---

### 2.3 Sigmoid Function — Why It Exists

The sigmoid function:

- Converts any real number into a probability
- Is smooth and differentiable
- Allows gradient-based optimization

Properties:

- $ \sigma(z) \to 1 $ as $ z \to +\infty $
- $ \sigma(z) \to 0 $ as $ z \to -\infty $

---

### 2.4 Odds and Log-Odds (Logit)

Logistic regression models the **log-odds**:

$$
\text{Odds} = \frac{P(y=1)}{1 - P(y=1)}
$$

$$
\log(\text{odds}) = w x + b
$$

Why log-odds?

- Probability is bounded
- Log-odds range from $ -\infty $ to $ +\infty $
- Makes classification compatible with a linear model

---

### 2.5 Loss Function (Binary Cross-Entropy)

Logistic regression minimizes **Binary Cross-Entropy Loss**:

$$
J = -\frac{1}{n} \sum \Big[y \log(\hat{y}) + (1-y)\log(1-\hat{y})\Big]
$$

Why not MSE?

- MSE performs poorly for classification
- Cross-entropy strongly penalizes confident wrong predictions

---

### 2.6 Learning the Parameters

There is **no closed-form solution**.

Parameters are learned using:

- Maximum Likelihood Estimation (MLE)
- Implemented via Gradient Descent

Gradients:

$$
\frac{\partial J}{\partial w} = \frac{1}{n} \sum (\hat{y} - y)x
$$

$$
\frac{\partial J}{\partial b} = \frac{1}{n} \sum (\hat{y} - y)
$$

---

### 2.7 Decision Boundary

The **decision boundary** is where the model is uncertain:

$$
P(y=1) = 0.5
$$

This occurs when:

$$
w x + b = 0
$$

For 1D:

$$
x = -\frac{b}{w}
$$

For higher dimensions:

- Line (2D)
- Plane (3D)
- Hyperplane (nD)

---

## 3. Linear vs Logistic Regression (Comparison)

| Aspect         | Linear Regression  | Logistic Regression |
| -------------- | ------------------ | ------------------- |
| Task           | Regression         | Classification      |
| Output         | Continuous         | Probability (0–1)  |
| Activation     | None               | Sigmoid             |
| Loss           | MSE                | Cross-Entropy       |
| Boundary       | Not applicable     | Decision boundary   |
| Output range   | (-∞, +∞)         | (0, 1)              |
| Interpretation | Numeric prediction | Probability         |

---

## 4. Core Intuition (Most Important)

- **Linear regression** fits a line to predict numbers
- **Logistic regression** fits a line to separate classes
- Sigmoid does **not** make the model nonlinear
- Logistic regression is a **linear classifier with probabilistic output**

---

## 5. Key Takeaways

- Regression vs Classification is about **output interpretation**
- Loss function choice defines learning behavior
- Decision boundary comes from the linear equation, not sigmoid
- Sigmoid only converts distance → probability
- Understanding math first makes libraries trivial

---

## 6. Next Steps

- Polynomial Regression
- Regularized Logistic Regression
- Multiclass Logistic Regression (Softmax)
- KNN vs Logistic Regression
- Visualizing loss surfaces

---

## Quick Notes (At a Glance)

| Linear Regression                                       | Logistic Regression                                       |
| ------------------------------------------------------- | --------------------------------------------------------- |
| Predicts**continuous values**                     | Predicts**categorical classes** (via probabilities) |
| Fits a**best-fit straight line** to the data      | Uses the**sigmoid (S-shaped) curve**                |
| Used to solve**regression problems**              | Used to solve**classification problems**            |
| Output values are**unbounded**                    | Output is a**probability between 0 and 1**          |
| Common loss function:**Mean Squared Error (MSE)** | Common loss function:**Binary Cross-Entropy**       |
| -                                                       | Decision is made using a**threshold** (usually 0.5) |

---

*These notes accompany hands-on implementations built entirely from scratch as part of a structured learning challenge.*
