# Day 04 — Naive Bayes

Naive Bayes is a **probabilistic machine learning algorithm** based on **Bayes’ Theorem**.  
It is widely used for **classification tasks**, especially in applications like **spam filtering**, **sentiment analysis**, and **text classification**.

---

## What is Naive Bayes?

Naive Bayes is a probabilistic classification algorithm that predicts a class by choosing the one with the **highest posterior probability** given the input features, while assuming that the features are **independent of each other**.

This independence assumption is usually not true in real-world data, which is why the algorithm is called *naive*.  
Despite this, it performs surprisingly well in many practical scenarios.

---

## Bayes’ Theorem

The conditional probability of event $A$ given event $B$ is defined as:

$P(A \mid B) = \frac{P(A \cap B)}{P(B)}$

Using Bayes’ Theorem, this can be rewritten as:

$P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}$

In Naive Bayes classification:
- $P(A)$ is the **prior probability**
- $P(B \mid A)$ is the **likelihood**
- $P(A \mid B)$ is the **posterior probability**

---

## Example 1 — Play Tennis Dataset (Naive Bayes)

This is the classic **Play Tennis** dataset used to demonstrate **Naive Bayes classification**.  
The objective is to predict whether we should **play tennis** based on weather conditions.


### Dataset Description

**Features:**
- Outlook: Sunny, Overcast, Rain
- Temperature: Hot, Mild, Cool
- Humidity: High, Normal
- Wind: Weak, Strong

**Target Class:**
- Play Tennis: Yes / No


### Dataset

| Outlook   | Temperature | Humidity | Wind   | Play Tennis |
|----------|-------------|----------|--------|-------------|
| Sunny    | Hot         | High     | Weak   | No  |
| Sunny    | Hot         | High     | Strong | No  |
| Overcast | Hot         | High     | Weak   | Yes |
| Rain     | Mild        | High     | Weak   | Yes |
| Rain     | Cool        | Normal   | Weak   | Yes |
| Rain     | Cool        | Normal   | Strong | No  |
| Overcast | Cool        | Normal   | Strong | Yes |
| Sunny    | Mild        | High     | Weak   | No  |
| Sunny    | Cool        | Normal   | Weak   | Yes |
| Rain     | Mild        | Normal   | Weak   | Yes |
| Sunny    | Mild        | Normal   | Strong | Yes |
| Overcast | Mild        | High     | Strong | Yes |
| Overcast | Hot         | Normal   | Weak   | Yes |
| Rain     | Mild        | High     | Strong | No  |

Total instances = **14**

| Class | Count |
|------|------|
| Yes  | 9 |
| No   | 5 |


### Step 1 — Prior Probabilities

$P(Yes) = \frac{9}{14}$

$P(No) = \frac{5}{14}$


### Step 2 — Likelihood Probabilities

### Outlook

| Outlook | Yes | No |
|-------|-----|----|
| Sunny | 2 | 3 |
| Overcast | 4 | 0 |
| Rain | 3 | 2 |

$P(Sunny \mid Yes) = \frac{2}{9}$ \
$P(Sunny \mid No) = \frac{3}{5}$


### Temperature

| Temperature | Yes | No |
|------------|-----|----|
| Hot | 2 | 2 |
| Mild | 4 | 2 |
| Cool | 3 | 1 |

$P(Cool \mid Yes) = \frac{3}{9}$  \
$P(Cool \mid No) = \frac{1}{5}$


### Humidity

| Humidity | Yes | No |
|----------|-----|----|
| High | 3 | 4 |
| Normal | 6 | 1 |

$P(High \mid Yes) = \frac{3}{9}$ \
$P(High \mid No) = \frac{4}{5}$


### Wind

| Wind | Yes | No |
|------|-----|----|
| Weak | 6 | 2 |
| Strong | 3 | 3 |

$P(Strong \mid Yes) = \frac{3}{9}$ \
$P(Strong \mid No) = \frac{3}{5}$


### Step 3 — Classifying a New Instance

**Given conditions:**

- Outlook = Sunny  
- Temperature = Cool  
- Humidity = High  
- Wind = Strong  


### Compute Posterior for **Yes**

$P(Yes \mid X) \propto P(Yes) \times P(Sunny \mid Yes) \times P(Cool \mid Yes) \times P(High \mid Yes) \times P(Strong \mid Yes)$

Substitute values:

$= \frac{9}{14} \times \frac{2}{9} \times \frac{3}{9} \times \frac{3}{9} \times \frac{3}{9}$

$= \frac{9}{14} \times \frac{54}{6561}$

$= \frac{486}{91854}$

$= 0.00529$


### Compute Posterior for **No**

$P(No \mid X) \propto P(No) \times  P(Sunny \mid No) \times P(Cool \mid No) \times P(High \mid No) \times P(Strong \mid No)$

Substitute values:

$= \frac{5}{14} \times \frac{3}{5} \times \frac{1}{5} \times \frac{4}{5} \times \frac{3}{5}$

$= \frac{5}{14} \times \frac{36}{625}$

$= \frac{180}{8750}$

$= 0.02057$


### Step 4 — Final Decision

$P(No \mid X) > P(Yes \mid X)$

$\boxed{\text{Prediction: Play Tennis = No}}$


---

## Example 2 — Fruits Dataset

The task is to calculate the probability of each feature given a fruit class, such as:
- Yellow Mango
- Sweet Banana
- Long Mango, etc.

### Dataset

| Fruit  | Yellow | Sweet | Long | Total |
|--------|--------|-------|------|-------|
| **Mango**  | 350 | 450 | 0   | **650** |
| **Banana** | 400 | 300 | 350 | **400** |
| **Others** | 50  | 100 | 50  | **150** |
| **Total**  | **800** | **850** | **400** | **1200** |

---

### Example Calculation

We calculate the probability that a fruit is **yellow given that it is a mango**.

$P(\text{Yellow} \mid \text{Mango})$

Using Bayes’ Theorem:

$P(Yellow \mid Mango) =\frac{P(Mango \mid Yellow) \cdot P(Yellow)}{P(Mango)}$

Substituting values:

$= \frac{(350/800) \times (800/1200)}{650/1200}$

$= \frac{350}{650}$

$= 0.54$

So,

$P(Yellow \mid Mango) = 0.54$

This same method can be applied to compute:
- $P(Sweet \mid Mango)$
- $P(Long \mid Banana)$
- $P(Yellow \mid Others)$, etc.



---

## Limitations of Naive Bayes

- Assumes **feature independence**, which is rarely true in real-world data
- Zero probabilities can break predictions if **Laplace smoothing** is not applied
- Not suitable for problems with highly correlated features
- Performance drops when decision boundaries are highly non-linear

---

## What I Learned

- **Probability** and **likelihood** are often used interchangeably in Naive Bayes
- Technically:
  - *Probability* refers to the chance of an event occurring
  - *Likelihood* refers to how probable observed data is given a model
- In **Naive Bayes**, likelihoods are computed for **discrete values**
- In **Gaussian Naive Bayes**, likelihoods are computed using **continuous probability distributions**

---

## References

1. [Gate Smashers – Lec-8: Naive Bayes Classification Full Explanation with examples | Supervised Learning](https://www.youtube.com/watch?v=GBMMtXRiQX0)
2. [StatQuest with Josh Starmer - Naive Bayes, Clearly Explained!!!](https://www.youtube.com/watch?v=O2L2Uv9pdDA&list=PLJ07VAG7bJEqbhbxYm79EOP4jBHdtJ7lN&index=12)
3. [Mahesh Huddar - 1. Solved Example Naive Bayes Classifier to classify New Instance PlayTennis Example Mahesh Huddar](https://www.youtube.com/watch?v=XzSlEA4ck2I&list=PLJ07VAG7bJEqbhbxYm79EOP4jBHdtJ7lN&index=14)
4. ChatGPT
