# Day 01 — Linear Regression (Analytical + Gradient Descent Visualization)

Linear Regression is one of the simplest and most fundamental algorithms in Machine Learning.  
It tries to model the relationship between an input variable `x` and an output variable `y` using a straight line.

---

# What is Linear Regression?

Linear Regression assumes a linear relationship between input and output:

$y = mx + b $

Where:  
- $x $ = input (independent variable)  
- $y $ = output (dependent variable)  
- $m $ = slope (how much y changes for each unit change in x)  
- $b $ = intercept (value of y when x = 0)

Example:  
Predicting exam score using hours studied →  
$y = \text{score}, \ x = \text{hours studied} $

---

# Steps  
1. Data Collection  
2. Calculations (Manual Linear Regression)  
3. Predictions  
4. MSE + Gradient Descent  
5. Visualization  
6. Learning Summary  

---

# Example — Predict Pizza Price Using Diameter

We have the following dataset:

| Diameter (x) | Price (y) | Mean(x) | Mean(y) | Deviation(x) | Deviation(y) | Product of Deviation | Square Deviation(x) |
|---|---|---|---|---|---|---|---|
| 8     | 10    | 10      | 13      | -2     | -3     | 6       | 4     |
| 10    | 13    | -       | -       | 0      | 0      | 0       | 0     |
| 12    | 16    | -       | -       | 2      | 3      | 6       | 4     |

---

# 1. Manual Calculations (Analytical Linear Regression)

### Mean of x  
$\text{Mean}(x) = \frac{8+10+12}{3} = 10 $

### Mean of y  
$\text{Mean}(y) = \frac{10+13+16}{3} = 13 $

### Deviations  
- Dev(x) = x – Mean(x)  
- Dev(y) = y – Mean(y)  

### Product of Deviations  
$\text{Dev}(x) \times \text{Dev}(y) $

### Slope (m)

$m = \frac{\sum (x - \bar{x})(y - \bar{y})}{\sum (x - \bar{x})^2} $

$m = \frac{12}{8} = 1.5 $

### Intercept (b)

$b = \bar{y} - m\bar{x} $

$b = 13 - (1.5)(10) = -2 $

### Final Equation

$y = 1.5x - 2$

This line perfectly fits the data.

---

# 2. What is MSE?

MSE (Mean Squared Error) tells us **how wrong the model is**.

For each point:

- **Error** = predicted – actual  
- **Squared Error** = $(\text{predicted} - \text{actual})^2 $

We square errors because:

- Negative errors should not cancel positive errors  
- Large errors should be punished more  

Example:
- Error = 5 → SE = 25  
- Error = 10 → SE = 100 (punished heavily)  

MSE is simply the **average of squared errors**.

---

# 3. What is Gradient Descent?

Gradient Descent is an optimization algorithm.  
It helps us **find the best values of slope (w) and intercept (b)** by minimizing the error.

Conceptually:

- Imagine standing on a hill.  
- The hill’s height = MSE (error).  
- You want to reach the lowest point.  
- The slope tells you which direction is downhill.  
- You take small steps down → these are updates to w and b.

### Update Rules

$w = w - \alpha \cdot \frac{\partial MSE}{\partial w} $  

$b = b - \alpha \cdot \frac{\partial MSE}{\partial b} $

Where:
- $\alpha$ = learning rate (step size)  
- Derivatives = gradient (slope of error curve)

---

# 4. Graph

![Loss Curve](Figure_1.png)


---

# 5. Why the Gradient Descent Curve Looks Like This

My curve:

- Starts **very high** → model is initially wrong  
- Drops **very fast** → big errors = big gradients = big corrections  
- Slowly **flattens** → model approaches the minimum  
- Never hits exactly zero → tiny rounding + tiny gradients  

This shape is **correct** and expected.

---

# 6. Learned

1. **What does MSE measure?**  
    The average wrongness of the model, punishing big mistakes heavily.

2. **Why square the errors?**  
    To remove signs and punish large errors more.

3. **What does the gradient tell us?**  
   The direction of steepest increase in error.

4. **Why subtract the gradient?**  
   Because gradient points uphill; subtracting moves us downhill.

5. **What is the learning rate?**  
   It controls step size during updates (too big = explode, too small = slow).

---

# References
- Chatgpt
- [Gate Smashers - (Lec-4: Linear Regression📈 with Real life examples & Calculations | Easiest Explanation)](https://www.youtube.com/watch?v=zUQr6HAAKp4)
- [GeekForGeeks - data science - gradient decent](https://www.geeksforgeeks.org/data-science/what-is-gradient-descent/)
- [StatQuest with Josh Starmer - Gradient Descent, Step-by-Step](https://www.youtube.com/watch?v=sDv4f4s2SB8)
- [Google for Developers - Machine Learning Crash Course: Gradient Descent](https://www.youtube.com/watch?v=QoK1nNAURw4)

