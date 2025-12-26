# Day 01 — K- Nearest Neighbor (KNN)

KNN is used to help in classifying the new data entry in a dataset by finding the nearest neighbor and discerning which category it belongs to.


---

# What is KNN?

K-Nearest Neighbors (KNN) is a supervised machine learning algorithm generally used for classification but can also be used for regression tasks. It works by finding the "k" closest data points (neighbors) to a given input and makes a predictions based on the majority class (for classification) or the average value (for regression). 


---

## Steps:

1. Data Collection
2. Calculate distance - Euclidean distance ![euclidean](euclidean.png)
3. 



---

## Example

The data for the recent released movies on Rotten Tomatoes are as follows:

|Name   |Rating (%) |Duration (min)|Genre|
|---|---|---|---|
|Dhurandar| 96 | 214 | Action |
|Avatar: Fire and Ash | 79 | 195 | Action |
|Anaconda | 79 | 99 | Action |
|The Spongebob Movie | 71 | 96 | Kids |
|Now You See Me: Now You Don't| 80 | 113 | Thriller |
|Five Nights at Freddy's 2 | 85 | 104 | Horror |
|Chainsaw Man: The Movie - Reze Arc | 98 | 100 | Horror |
|Kimetsu no Yaiba: Infinity Castle | 98 | 155 | Fantasy |
|Zootopia 2 | 91 | 107 | Kids |
|Wicked: For Good | 93 | 137 | Kids |

Lets say that for now i want to know which category would Animal movie be with Rotten Tomatoes rating of 81% and 204 minutes run time (we already know its in Action but for the sake of example ...)

as,  
$d= \sqrt{(x_2-x_1)^2+(y_2-y_1)^2}$

and rating is $x$, duration is $y$,  
$x_1$ is 81 and $y_1$ is 204,  
the distance between Animal's rating and duration with other movies are:  

$Dhurandar = \sqrt{(81-96)^2+(204-214)^2} = 18.02$  

$Avatar: Fire and Ash = \sqrt{(81-79)^2+(204-195)^2} = 9.21$  

$Anaconda = \sqrt{(81-79)^2+(204-99)^2} = 105.01$ 

$The Spongebob Movie = \sqrt{(81-71)^2+(204-96)^2} = 108.46$  

$Now You See Me: Now You Don't = \sqrt{(81-80)^2+(204-113)^2} = 91$  

$Five Nights at Freddy's 2 = \sqrt{(81-85)^2+(204-104)^2} = 100.07$  

$Chainsaw Man: The Movie - Reze Arc = \sqrt{(81-98)^2+(204-100)^2} = 105.38$  

$Kimetsu no Yaiba: Infinity Castle = \sqrt{(81-98)^2+(204-155)^2} = 51.86$  

$Zootopia 2 = \sqrt{(81-91)^2+(204-107)^2} = 97.51$  

$Wicked: For Good = \sqrt{(81-93)^2+(204-137)^2} = 68.06$  






---
# References

1. [Gate Smashers - Lec-7: kNN Classification with Real Life Example | Movie Imdb Example | Supervised Learning](https://www.youtube.com/watch?v=O1nWXTXcCwI)
2. [GeekForGeeks - machine learning - K Nearest Neighbor](https://www.geeksforgeeks.org/machine-learning/k-nearest-neighbours/)
3. [Rotten Tomatoes](https://www.rottentomatoes.com/)
4. ChatGPT
5. 


