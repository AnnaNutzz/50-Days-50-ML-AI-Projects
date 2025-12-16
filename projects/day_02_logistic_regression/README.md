# Day 02 — Logistic Regression

Logistic regression predicts a probability of categorical data like for some amount of rain, is there a posibility of flood and how much or trying to predict passing or failing of a student depending on hours spent to study.

---

# What is Logistic Regression?

Logistic Regression helps predict probability using a number of independent data that is of any numeric type but the output variable will be binary - indicating a True or False value (possibilities range from 0 to 1). It is used in supervised classification model.

$ \sigma(z) = \frac{1}{1+e^{-1}} $

where,

- $\sigma(z)$ tends towards 1 as $ z→ \infin $
- $\sigma(z)$ tends towards 0 as $ z→ -\infin $
- $\sigma(z)$ is always bounded between 0 and 1

where the probability of class can be measured as:
$P(y=1)= \sigma(z)$
$P(y=0)= 1-\sigma(z)$

It can be used on:

1. Binomial Data - 2 categories (Yes/No, True/False)
2. Multinomial Data - more than 2 categories (Yes/No/Maybe, Pass/Fail/Backlog)
3. Ordinal Data - data which is in order (High/Medium/Low, Stop/Wait/Go, Hot/Neutral/Cold)

Logistic Regression models the Log of the odds of an event,
basically,

$ \text{odds =}\frac{\text{something that can occur}}{\text{something that cant occur}} $

$$
\boxed{odds = \frac{P(x)}{1-p(x)}}
$$

$ \text{1 being the entirety of possibilities; everything} $


## Logits:

this is similar to linear regression:

$ \text{Log(Odds) = } \beta_0 + \beta_{1 \times i1} + \beta_{2 \times i2} + ... + \beta_{p \times ip} + error $

here Probability ranges from 0 to 1, but Log Odds ranges from $-\infin$ to $+\infin$
as

$$
\begin{aligned}
P = 0  \\
&= log [\frac{0}{1-0}] \\
&= log [0] \\
&= -\infin
\end{aligned}
$$

and

$$
\begin{aligned}
P = 1  \\
&= log [\frac{1}{1-1}] \\
&= log [1] \\
&= +\infin
\end{aligned}
$$


---
# The Mathematics behind Sigmoid

$ y = \frac{1}{1+e^{-(a_0+a_1x)}} $

where,  
- $ a_0 $ is intercept 
- $ a_1 $ is the co-efficient
- $x$ is independent variable

$ a_0\ and\ a_1 $ are found after using different methods like Maximum Likelihood Estimation Function and Gradient Decent Method or Specializer Solved Method on top of it.

## Simple Example:

Lets assume that $a_0$ is calculated to be $-1.5$ and $a_1$ is $0.6$ on the given dataset of results of hours studied by students,

|Hours studied|Pass/Fail|
|---|---|
|2  |0  |
|3  |0  |
|4  |0  |
|5  |1  |
|6  |1  |
|7  |1  |

Lets calculate for the student who studied 5 hours, so $x$ will be 5  
So,  

$$ 
\text{Power of e}=-(a_0 + a_1 x)  \\
    = -1.5 + 0.6 \times 5 \\
    = -1.5 + 3 \\
    = 1.5
$$

$e \approx 2.718$

$$
\therefore y= \frac{1}{1+e^{-1.5}} \\
\approx \frac{1}{1+2.718^{(-1.5)}} \\
\approx \frac{1}{1+0.223} \\
\approx \frac{1}{1.233} \\
\approx 0.817
$$

$y \approx 0.817 $  
So the chances of student to pass is $\approx$ 81.7%, they come under **PASS**

Lets say if the student studied 1.5 hours, so $x$ will be 1.5  
So,

$$ 
\text{Power of e}=-(a_0 + a_1 x)  \\
    = -1.5 + 0.6 \times 1.5 \\
    = -1.5 + 0.9 \\
    = -(-0.6) = 0.6
$$

$e \approx 2.718$

$$
\therefore y= \frac{1}{1+e^{0.6}} \\
\approx \frac{1}{1+2.718^{(0.6)}} \\
\approx \frac{1}{1+1.822} \\
\approx \frac{1}{2.822} \\
\approx 0.354
$$

$y \approx 0.354 $  
So the Chances of student to pass is $\approx$ 35.4%, they come under **FAIL**



# References

- [GeekForGeeks - machine learning - Logistic Regression](https://www.geeksforgeeks.org/machine-learning/understanding-logistic-regression/)
- [Learn Analytics - 2. Logistic Regression – Introduction](https://www.youtube.com/watch?v=JuxX-hfRg1o&list=PLJ07VAG7bJEqbhbxYm79EOP4jBHdtJ7lN&index=3)
- [Visually Explained - Logistic Regression (and why it's different from Linear Regression)](https://www.youtube.com/watch?v=3bvM3NyMiE0&list=PLJ07VAG7bJEqbhbxYm79EOP4jBHdtJ7lN&index=4)
- [Gate Smashers - Lec-5: Logistic Regression with Simplest & Easiest Example | Machine Learning](https://www.youtube.com/watch?v=r8OjlgWpAI0&list=PLJ07VAG7bJEqbhbxYm79EOP4jBHdtJ7lN&index=6)