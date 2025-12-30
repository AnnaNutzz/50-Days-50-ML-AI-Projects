# Day 03 — K-Nearest Neighbors (KNN)

KNN is used to classify a new data entry in a dataset by finding the *k nearest neighbors* and determining which category the data point most likely belongs to.

---

## What is KNN?

K-Nearest Neighbors (KNN) is a supervised machine learning algorithm primarily used for **classification**, but it can also be applied to **regression** problems.  
It works by identifying the *k* closest data points (neighbors) to a given input and making predictions based on:

- **Majority voting** (for classification)
- **Average value** (for regression)

KNN is a **lazy learning** algorithm, meaning it does not learn an explicit model during training and instead relies on the entire dataset at prediction time.

---

## Steps

1. Data Collection  
2. Distance Calculation (commonly Euclidean distance)  
   <img src="euclidean.png" width="80%">  
3. Finding the *K-Nearest Neighbors*  
4. Majority Voting (classification) / Averaging (regression)

> ### Feature Scaling (Important)
>
> Before applying KNN, features must be normalized or standardized.  
> Otherwise, features with larger numeric ranges dominate distance calculations.
>
> Common methods:
> - Min-Max Scaling  
> - Standardization (Z-score)

---

## Example 1 — Movie Genre Classification

The data for recently released movies on Rotten Tomatoes is shown below:

| Name | Rating (%) | Duration (min) | Genre |
|---|---|---|---|
| Dhurandar | 96 | 214 | Action |
| Avatar: Fire and Ash | 79 | 195 | Action |
| Anaconda | 79 | 99 | Action |
| The Spongebob Movie | 71 | 96 | Kids |
| Now You See Me: Now You Don't | 80 | 113 | Thriller |
| Five Nights at Freddy's 2 | 85 | 104 | Horror |
| Chainsaw Man: The Movie – Reze Arc | 98 | 100 | Horror |
| Kimetsu no Yaiba: Infinity Castle | 98 | 155 | Fantasy |
| Zootopia 2 | 91 | 107 | Kids |
| Wicked: For Good | 93 | 137 | Kids |

Suppose we want to predict the genre of the movie **Animal**, which has:
- Rating = **81%**
- Duration = **204 minutes**

(Although the real genre is known, this is used purely for demonstration.)

---

### 1. Calculating Distance

The Euclidean distance formula is:

$d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$

Where:
- Rating = $x$
- Duration = $y$
- $x_1 = 81$, $y_1 = 204$

Distances from *Animal* to other movies:

- $Dhurandar = \sqrt{(81-96)^2 + (204-214)^2} = 18.02$
- $Avatar: Fire and Ash = \sqrt{(81-79)^2 + (204-195)^2} = 9.21$
- $Anaconda = 105.01$
- $The\ Spongebob\ Movie = 108.46$
- $Now\ You\ See\ Me:\ Now\ You\ Don't = 91$
- $Five\ Nights\ at\ Freddy's\ 2 = 100.07$
- $Chainsaw\ Man:\ Reze\ Arc = 105.38$
- $Kimetsu\ no\ Yaiba:\ Infinity\ Castle = 51.86$
- $Zootopia\ 2 = 97.51$
- $Wicked:\ For\ Good = 68.06$

---

### 2. Finding K-Nearest Neighbors

- If **k = 1**, the closest movie is:
  - *Avatar: Fire and Ash* (distance = 9.21)

- If **k = 3**, the nearest neighbors are:
  - Avatar: Fire and Ash (9.21)
  - Dhurandar (18.02)
  - Kimetsu no Yaiba: Infinity Castle (51.86)

---

### 3. Majority Voting (Classification)

- Avatar: Fire and Ash → Action  
- Dhurandar → Action  
- Kimetsu no Yaiba: Infinity Castle → Fantasy  

Since **Action** appears most frequently, the movie **Animal** is classified as:

**Action**

> **Note:** Multi-label classification would require an explicitly designed multi-label KNN approach.

---

## Example 2 — Sports Prediction (Statistical Machine Learning – Masters)

Data encoding:
- **1 = Female**
- **0 = Male**

> **Note:** Gender is encoded numerically here for simplicity.  
> In real-world ML systems, categorical variables should be encoded carefully (e.g., one-hot encoding).

| Name | Age (y) | Gender (x) | Sport |
|---|---|---|---|
| Sara | 16 | 1 | Cricket |
| Zaira | 34 | 1 | Cricket |
| Sachin | 55 | 0 | × |
| Rahul | 40 | 0 | Cricket |
| Pooja | 20 | 1 | × |
| Smith | 15 | 0 | Cricket |
| Laxmi | 55 | 1 | Football |
| Michael | 15 | 0 | Football |

**Problem:**  
Predict which sport **Angelina** would play if:
- Age = **5**
- Gender = **Female**
- $k = 3$

> **Note:** The test point (age = 5) lies outside the training age range.  
> KNN predictions are less reliable for such extrapolated inputs.

---

### Distance Calculation

Using:
$d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$

Where $(x_1 = 1), (y_1 = 5)$:

- Sara = 11  
- Zaira = 29  
- Sachin = 50.01  
- Rahul = 35.01  
- Pooja = 15  
- Smith = 10.05  
- Laxmi = 50  
- Michael = 10.05  

---

### Nearest Neighbors (k = 3)

- Smith (10.05) → Cricket  
- Michael (10.05) → Football  
- Sara (11) → Cricket  

**Majority Vote:** Cricket

> **Prediction:** Angelina would most likely play **Cricket**.

---

## Limitations of KNN

- No explicit training phase; prediction is computationally expensive
- Highly sensitive to feature scaling
- Memory intensive
- Performs poorly in high-dimensional spaces (curse of dimensionality)

---

## What I Learned

- KNN does not learn a model; it relies entirely on stored data at inference time
- Feature scaling is **critical** for distance-based algorithms
- Choice of **k** directly affects bias–variance tradeoff
- KNN struggles with extrapolation beyond the training data range
- Numerical encoding of categorical variables can introduce unintended bias
- KNN is intuitive and powerful for small datasets but does not scale well

---

## References

1. [Gate Smashers – Lec-7: KNN Classification with Real Life Example](https://www.youtube.com/watch?v=O1nWXTXcCwI)
2. [GeeksforGeeks – K Nearest Neighbors](https://www.geeksforgeeks.org/machine-learning/k-nearest-neighbours/)  
3. [Rotten Tomatoes](https://www.rottentomatoes.com/)  
4. ChatGPT  
5. [Visually Explained – KNN in 3 Minutes](https://www.youtube.com/watch?v=gs9E7E0qOIc)
6. [Gate Smashers – Lec-19: KNN for Classification & Regression](https://www.youtube.com/watch?v=zqQ_pi6j2jE)
