# Day 02 — Logistic Regression 

Logistic regression predicts a probability of categorical data like for some amount of rain, is there a posibility of flood and how much.

---

## What is Logistic Regression?

Logistic Regression helps predict probability using a number of independent data that is of any numeric type but the output variable will be binary - indicating a True or False value.


$ P (y = 1∣x)= \sigma( wTx + b) $

where,  
- $ wTx+b $ is a linear line/plane  
- $ \sigma(z) $ squashes it into [0,1]


It can be used on:  
1. Binomial Data - 2 categories (Yes/No, True/False)  
2. Multinomial Data - more than 2 categories (Yes/No/Maybe, Pass/Fail/Backlog)  
3. Ordinal Data - data which is in order (High/Medium/Low, Stop/Wait/Go, Hot/Neutral/Cold)

basically,  

$ \text{odds =}\frac{\text{something that can occur}}{\text{something that cant occur}} $

$$ \boxed{odds = \frac{p(x)}{1-p(x)}} $$

