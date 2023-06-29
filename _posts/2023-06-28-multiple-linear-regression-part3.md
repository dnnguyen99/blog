---
layout: post
title: "Multiple Linear Regression (Part 3)"
author: "Diep Nguyen"
tags: [linear regression, optimization]
categories: journal
image: lr3_bg.jpg
---
> In this article, we will look at another technique to calculate the regression coefficients $\beta$ called Gradient Descent. This is an optimization technique that involves taking the gradient to find the optimal $\beta$ values that minimize the loss function. 

 
## Optimization and Linear Regression
In the [previous post]({{ site.github.url }}{% post_url 2023-05-30-multiple-linear-regression-part1 %}), we looked at how the normal equation can be used to solve for the coefficient estimates $\beta$. Our loss function from the OLS method remains the same. Now, we will discuss a different approach that uses an optimization technique called Gradient Descent (GD) to find $\beta$ estimates that minimize the loss function. GD is an iterative method that starts with a random initiation of the $\beta$ estimates and iteratively updates the estimates until convergence. Let us introduce a few notations for GD:

1. $\beta^{(i)}$: the current $\beta$ estimate
2. $\beta^{(i+1)}$: the new estimate obtained from $\beta$ after one iteration of GD
3. $\lambda$: the learning rate or step size
4. $\nabla L(y,\hat{y})$ : the gradient of the loss function. This is a vector of derivatives of the loss function w.r.t. $\beta$.
 
 We update $\beta$ in the opposite direction of the gradient of the loss function:
 
  $$\beta^{(i+1)} = \beta^{(i)} - \lambda \nabla L(y, \hat{y}) $$
 
## Numerical Example  
Let's use the same dataset used in [Multiple Linear Regression (Part 1)]({{ site.github.url }}{% post_url 2023-05-30-multiple-linear-regression-part1 %}):

Person | Age  | Height | IQ
:-----:|:----:|:------:|:-------:
1      | 20   | 170    | 99
2      | 6    | 45     | 20
3      | 40   | 160    | 101
4      | 12   | 150    | 45

After normalizing the dataset and get the design matrix $$X = \begin{bmatrix}
  1 & 0.7704 & 1.1832\\
  1 & -1.7148 & -1.5213\\
  1 & 0.5716 & 0.5071\\
  1 & 0.3728 & -0.1690\\
  \end{bmatrix}$$, we can predict $\beta = [ \beta_0  \quad \beta_1 \quad \beta_2]^T$ using gradient descent. 

### GD Step 1: Initialize the estimates
First, let us start with a random initiation of $\beta$. This can be anything, but let's initialize it to be a vector of zeros, $\beta^{(0)} = [ 0 \quad 0 \quad 0]^T$.

### GD Step 2: Calculate the gradient of the loss function using the current estimates
Now, we need to find the gradient of the loss function, $\nabla L(y,\hat{y})$. This is the same as taking the derivative of the loss function w.r.t. $\beta$. Luckily, we know from the derivation of the Normal Equation in Part 1 that 

$$\nabla L(y,\hat{y}) = - 2X^T y + 2X^TX \beta= 2X^T(X \beta -y)$$

where

$$X = \begin{bmatrix}
  1 & 0.7704 & 1.1832\\
  1 & -1.7148 & -1.5213\\
  1 & 0.5716 & 0.5071\\
  1 & 0.3728 & -0.1690\\
  \end{bmatrix}, y = \begin{bmatrix}
  20\\
  6\\
  40\\
  12
  \end{bmatrix}, \beta = \begin{bmatrix}
  0\\
  0\\
  0
  \end{bmatrix}$$
 
So, 

$$\nabla L(y,\hat{y}) = \begin{bmatrix}-156\\
-64.9157\\
-86.688
\end{bmatrix}$$

### GD Step 3: Update the estimates
Let the step size $\lambda = 0.01$. We can adjust the step size to make the coefficients converge faster. However, we should be careful about using a step size that is too large as we might miss the optimal solution. We will update the new estimate using our update rule:

$$\begin{align} 
\beta^{(1)} &= \beta^{(0)} - \lambda \nabla L(y, \hat{y})\\
\beta^{(1)} &= \begin{bmatrix}
  0\\
  0\\
  0
  \end{bmatrix} - 0.01  \begin{bmatrix}-156\\
-64.9157\\
-86.688
\end{bmatrix}\\
\beta^{(1)} &= \begin{bmatrix} 1.56\\
0.6492\\
0.8669
\end{bmatrix} 
\end{align}$$
 
### GD Step 4: Repeat until convergence
We repeat steps 2 and 3 to update our estimates until $\beta^{(i+1)} = \beta^{(i)}$. Note that we need to use the current $\beta^{(i)}$ to compute the gradient for each iteration. 

After 1000 iterations, we get the estimates 
$$\beta = \begin{bmatrix} 19.5 \\
-3.059\\
13.389
\end{bmatrix}$$, which is the same as the estimates we get by using the normal equation.

## Understanding Gradient Descent

Think of gradient descent as finding the lowest point on a hill. Imagine that the shape of the hill represents our loss function, which measures how far off the predicted is from the actual response. Our goal is to minimize the loss. So, we want to reach the bottom of the hill, which corresponds to the optimal solution.

To start, we pick a random position on the hill (randomly initialize our $\beta$. Now, in order to descend to the bottom, we need to move in the direction of the steepest descent. The gradient of the loss function tells us the direction of the steepest ascent, so we actually want to go in the opposite direction of the gradient.

We take steps down the hill, and the size of each step is determined by $\lambda$, the learning rate or the step size. We want to reach the bottom as quickly as possible, but we need to keep in mind the value for $\lambda$. If $\lambda$. is too large, we might overshoot the bottom of the hill and miss the optimal solution. On the other hand, if the learning rate is too small, it will take us a very long time to reach the bottom because our steps are too tiny and we converge very slowly.

![alt text](https://github.com/dnnguyen99/dnnguyen99.github.io/blob/gh-pages/assets/img/gd.jpg?raw=true){:width="600px"}


In summary, gradient descent is like descending a hill to find the lowest point. We start at a random position, move in the opposite direction of the gradient (which points uphill), and adjust the size of our steps using the learning rate. The aim is to find the optimal solution efficiently without overshooting or converging too slowly.

## Difference Between Closed-form Solution and GD in Linear Regression
Unlike the mathematically derived OLS solution, gradient descent does not give us a closed-form equation for our $\beta$ coefficients. In fact, GD is an iterative method and is more general in the sense that it can be applied to solve other (optimization) problems that involve minimizing an objective function. In the case of linear regression, the objective function is the loss function in OLS. 

It should be noted that the difference we talk about here is the difference in the *approach* of solving OLS: using matrix algebra to get a closed-form solution vs using an iterative method. So, when should we use one instead of the other? GD can be useful if we have a large dataset making it computationally costly to find the inverse of the design matrix. If we have a small/moderate dataset, using the closed-form solution to compute the estimates might be beneficial as GD can take a long time to converge, depending on the random initialization in the beginning and the step-size chosen.

