---
layout: post
title: "Multiple Linear Regression"
author: "Diep Nguyen"
categories: journal
---

## What is Multiple Linear Regression?
Linear regression is a machine learning technique used to model the relationship between a quantitative response variable and one or more predictors. The goal is to find a linear model that captures the relationship between the independent (predictor) variables and the dependent (response) variable. This model can then be used to predict new, unseen data. When there are multiple predictors involved, we use the term "multiple linear regression" to denote the extension of simple linear regression to accommodate more than one predictor variable. 

For example, there might be a linear relationship between a person's age and their height and weight. We can fit a linear regression model using the available age, height, and weight data to capture this relationship. Then, the response (denoted as $y$ will be the age we try to predict. 
There are two predictors: height (denoted as $X^{(1)}$) and weight (denoted as $X^{(2)}$). Linear regression uses a linear equation with coefficients $\beta$ to capture the relationship between the predictors 
and the response:

  $$age = \beta_0 + \beta_1 (Height) + \beta_2 (Weight)$$
  
In general, if there are $p$ predictors
  $$y = \beta_0 + \beta_1X^{(1)} + \beta_2X^{(2)} + \cdots + \beta_pX^{(p)}, \text{ where }  \beta_0 \text{ is the intercept term. }$$
 
The phrase "fitting a linear model" means we want to find coefficients $\beta$ such that the model $y = \beta_0 + \beta_1X^{(1)} +  \beta_2X^{(2)} +\cdots + \beta_pX^{(p)}$ accurately represents the relationship between the predictors and the response. 

## Linear Regression in Matrix Form

In many cases, the dataset we work with contains multiple observations, such as data on N individuals including their age, height, and weight. Writing out the linear regression model for each individual can become cumbersome and notation-heavy. Therefore, it is customary to represent the data in matrix form. We let $y$ be an $N \times 1$ vector that contains all the ages of person $1$ upto person $N$. Usually, we say $y \in \mathbb{R}^{(N \times 1)}$. We will create a matrix $X \in \mathbb{R}^{N \times (p+1)} $, where $p$ is the number of predictor variables. Usually, this matrix is called the design matrix that contains $1$'s in its
first column to account for the intercept term $\beta_0$, and the predictor variables in its remaining columns. Each row of this matrix represents one observation. In our example, the design matrix contains $1$'s 
in its first column and the height and weight of person 1 up to person N in its second and third columns,
respectively. Lastly, we will have a vector containing the coefficients called $\beta \in \mathbb{R}^{(p+1) \times 1}$. The dimension of this vector depends on the number of predictors. 
Since our example has two predictors (height and weight), we will have  $\beta = [ \beta_0  \beta_1  \beta_2]^T$, where $\beta_0$ is the intercept term. 

  $$y = \begin{bmatrix}
  y_0\\
  y_1\\
  \cdot\\
  \cdot\\
  \cdot\\
  y_N
  \end{bmatrix}, X = \begin{bmatrix}
  1 & X_1^{(1)} & X_1^{(2)} & \cdots & X_1^{(p)}\\
  1 & X_2^{(1)} & X_2^{(2)} & \cdots & X_2^{(p)}\\
  \cdot & \cdot & \cdot & \cdot & \cdot \\
  \cdot & \cdot & \cdot & \cdot & \cdot \\
  \cdot & \cdot & \cdot & \cdot & \cdot \\
  1 & X_N^{(1)} & X_N^{(2)} & \cdots & X_N^{(p)}\\
  \end{bmatrix}, \beta = \begin{bmatrix} \beta_0 \\
  \cdot \\
  \cdot \\
  \cdot \\
  \beta_p \end{bmatrix}$$
 
The regression model aims to find coefficients $\beta$ that, when multiplied by the design matrix X, give a good estimate for $y$. 

  $$\hat{y} =\beta_0 + \sum_{j=1}^{p} X^{(j)} \beta_j = X \beta $$
 
Note that we use $\hat{y}$ instead of $y$ because there are almost always some unexplained variations or errors present in the data. So, we will not have a perfect linear relationship between the predictors and the response variable. Thus, $\hat{y}$ is an estimate of the true $y$.

## The Loss Function

To reiterate, we do not expect the predictors and the response to have a perfect linear relationship. So, $\hat{y}$ will not be the same as $y$. However, we can still fit a model that gives $\hat{y}$ as close to $y$ as possible. Intuitively, the perfect model will give $\beta_0, \beta_1, \cdots, \beta_p$ estimates such that, when using these estimates to calculate $\hat{y}$, the overall distance/difference between $\hat{y}$ and $y$ is minimized.

We use the Ordinary Least Squares (OLS) method to estimate the coefficients of the regression model. OLS aims to find the best-fit line that minimizes the overall difference between the predicted values and the actual values (also known that the residual sum of squares or RSS). We define the loss function $L(y,\hat{y})$ that measures the overall difference between the actual $y$ and the predicted $\hat{y}. For a particular observation n, the loss function is:

  $$L(y_n,\hat{y_n}) = (y_n - \hat{y_n}) = (y_n - X_n \beta)^2$$
 
 Then, the overall difference is:

  $$L(y,\hat{y}) = \sum_{n=1}^N(y_n - X \beta)^2$$

 
One can show that $\sum_{n=1}^N(y_n - X \beta)^2$ is equivalent to $\lVert y-X \beta \rVert_2^2$. Usually, $\lVert.\rVert_2$ is referred to as the L2 norm or the Euclidean norm. We want to find coefficients $\beta$ that will minimize this loss:

  $$\hat{\beta} = argmin_{\beta} L(y,\hat{y})$$
 
this means that $\hat{\beta}$ should be the values that minimize the loss function (the minimizer of $L(y,\hat{y})$.

## Deriving the coefficient estimates

Recall from Calculus, for a convex function $f(x)$, the derivative of $f(x)$ gives us the slope of the line tangent to the point $x$ on the curve. Since the function is convex, at the minimum, the slope of the tangent line is equal to 0. To find the minimum point of the function, we find $x$ such that the derivative of the function at $x$ is equal to $0$. In other words, we take the derivative of $f(x)$, set it equal to $0$, and solve for $x$. 

We can apply the same concept to find $\hat{\beta}$ that minimizes the loss function. Note that the loss function will have to be convex. We will omit the proof for this, but one can show that any lp norm is convex using the definition of a convex function and the triangle inequality. To find $\hat{\beta}$, we take the (partial) derivative of $L(y,\hat{y})$ with respect to $\beta$, set it equal to $0$, and solve for $\hat{\beta}$. 

Let us first compute the derivative of the loss function: 

  $$\frac{\partial}{\partial \beta} L(y,\hat{y}) =\frac{\partial}{\partial \beta} \lVert y-X \beta \rVert_2^2 $$

Recall from linear algebra that $\lVert x \rVert_2^2 = x^T x$ and $(AB)^T = B^TA^T$. Then,

  $$\frac{\partial}{\partial \beta} \lVert y-X \beta \rVert_2^2  = \frac{\partial}{\partial \beta} (y-X \beta)^T(y-X \beta)$$ 
  $$ = \frac{\partial}{\partial \beta} (y^T - \beta^T X^T) (y - X \beta)$$
  $$ = \frac{\partial}{\partial \beta} y^Ty  -y^TX \beta - \beta^T X^Ty + \beta^T X^T X \beta$$
  $$ =\frac{\partial}{\partial \beta} y^Ty - 2 \beta^T X^T y + \beta^T X^T X \beta$$
  $$ = -2X^Ty + 2X^T X \beta $$

Note that since $y^TX \beta \text{ is a scalar value, } y^TX \beta = (y^TX \beta)^T = \beta^TX^Ty$. So, $-y^TX \beta - \beta^T X^Ty = - 2 \beta^T X^T y$.

Setting this derivative equal to 0 gives us the Normal Equation. This equation is used to find the closed-form solution for $\hat{\beta}$ that minimize the loss function. 

  $$-2X^Ty + 2X^TX = 0$$
  $$2X^TX = 2X^Ty$$
  $$X^TX \beta = X^T y$$

 
If $X^TX$ is invertible, we can solve for $\hat{\beta}$:
  $$\hat{\beta} = (X^TX)^{-1}X^T y$$

## Interpretation
We have derived a closed-form expression for $\hat{\beta}$, but how can we apply this to fit a regression model? Let's go back to our 
example. Let's say we have data on the age, height, and weight of four people:

Person | Age  | Height  | Weight
------ |:----:|:--------| --------:
1      | 20   | 170     | 60
2      | 6    | 45      | 20
3      | 40   | 160     | 50
4      | 12   | 150     | 40

Now, given a fifth person with a height of $135$ cm and
weight $40$ kg, how do we use the available data to predict this new person's age? We can use the closed form equation $\hat{\beta}= (X^TX)^{-1}X^T y$ to find the coefficient estimates. First, let's put our data into matrix form:

  $$y = \begin{bmatrix}
  y_1\\
  y_2\\
  y_3\\
  y_4
  \end{bmatrix} = \begin{bmatrix}
  20\\
  6\\
  40\\
  12
  \end{bmatrix}, X = \begin{bmatrix}
  1 & X_1^{(1)} & X_1^{(2)}\\
  1 & X_2^{(1)} & X_2^{(2)} \\
  1 & X_3^{(1)} & X_3^{(2)} \\
  1 & X_4^{(1)} & X_4^{(2)}\\
  \end{bmatrix} = \begin{bmatrix}
  1 & 170 & 60\\
  1 & 45 & 20\\
  1 & 160 & 50\\
  1 & 150 & 40\\
  \end{bmatrix} , \beta = \begin{bmatrix} \beta_0 \\
  \beta_1 \\
  \beta_2 \end{bmatrix}$$

Before we proceed, we should normalize the design matrix. A person's height is measured in cm and a person's weight is measured in kg. The two variables have different magnitudes (e.g., height ranging from 45 cm to 170 cm and weight ranging from 20 kg to 60 kg). Normalization ensures that the variables are on a similar scale, preventing one variable from dominating the regression model's results simply due to its larger magnitude. The new, normalized design matrix $X$ is

{% raw %}
  $$X = \begin{bmatrix}
  1 & 0.7704 & 1.1832\\
  1 & -1.7148 & -1.5213\\
  1 & 0.5716 & 0.5071\\
  1 & 0.3728 & -0.1690\\
  \end{bmatrix}$$
 {% endraw %}

We will also need to normalize the new person's height and weight using the mean and sd height and weight from the dataset. For the fifth person, their normalized height is $0.0746$ and their normalized weight is $-0.1690$.

Now, we can use the equation $\hat{\beta} = (X^TX)^{-1}X^T y$ to find the coefficients of the regression model. First, we need to find the inverse of $X^TX$. Recall from our derivation above that $X^TX$ has to be non-singular (i.e., $X^TX$ has an inverse).  

$$(X^TX)^{-1} = 
  \begin{bmatrix} 
  2.5e-01 & -1.1485e-17 & 1.5963e-17\\
  -1.1485e-17 & 2.0424 & -1.9133\\
  1.5963e-17 & -1.9133 & 2.0424
  \end{bmatrix}$$

Then, 
  $$\hat{\beta} = \begin{bmatrix} 
  2.5e-01 & -1.1485e-17 & 1.5963e-17\\
  -1.1485e-17 & 2.0424 & -1.9133\\
  1.5963e-17 & -1.9133 & 2.0424
  \end{bmatrix} \begin{bmatrix} 
  1 & 1& 1& 1\\
  0.7704 & -1.7148 & 0.5716 & 0.3728\\
  1.1832 & -1.5213 & 0.5071 &-0.1690
  \end{bmatrix} \begin{bmatrix}
  20\\
  6\\
  40\\
  12
  \end{bmatrix}
  $$\\
  $$\hat{\beta} = \begin{bmatrix} 19.5\\
  3.5503\\
  4.8721
  \end{bmatrix}$$
  
Lastly, given the fifth person's height $(0.0746)$ and weight $(-0.1690)$, we can predict their age using the estimates we just calculated:

  $$\hat{y} = \hat{\beta_0} + 0.0746 \hat{\beta_1} + (-0.1690) \hat{\beta_2} $$
  $$\hat{y} =  19.5 + 0.0746 * 3.5503  -0.1690 * 4.8721 $$
  $$\hat{y} = 22.811 = 23$$

Given the dataset containing the age, height, and weight of four observations, we fit a linear regression model and used the coefficients to predict the age of a new person using (unseen) data on their height and weight. 

## Limitations
Linear regression is a widely used statistical technique for modeling the relationship between a dependent variable and one or more independent variables. However, it has certain limitations that should be considered. Recall when deriving the closed form solution of $\hat{\beta}$, we need $X^TX)$ to be invertible. This condition does not always hold. It is proven that

  $$(X^TX) \text{ is singular} \iff rank(X) < \text{ number of columns of } X $$
 
For the design matrix $X$, $rank(X)$ represents the number of columns that are "unique". So, when the number of unique columns is less than the total number of columns, $X^TX$ will not have an inverse. A trivial way this can happen is when we accidentally enter the same predictor more than once (the design matrix has columns that are linearly dependent). Another way $X^TX$ is not invertible is when  $N \ll p$ (when we have more predictors than observations). Having a singular $X^TX$ means that there's no unique best solution OLS can find.

Even when $X^TX$ has an inverse, we should not always fit a linear regression model using all the available predictors. Overfitting occurs when the model follows the training dataset too closely, leading to poor generalization to new, unseen data. Including all predictors in the model increases the complexity and flexibility of the regression equation, allowing it to capture even small variations in the training data. However, this can result in an overly complex model that fails to generalize well to new observations.

To solve these problems, we might want to consider using a regularized linear regression technique. Ridge and Lasso are two well-known regularized regression methods that add a penalization term to the loss fucntion. Because of this added term, the solution to Ridge and Lasso will always be unique. Further, we can use dimensionality reduction techniques to ensure $p \ll N$. Principal Component Analysis (PCA) is a well-known technique. PCA takes the original predictor variables and transforms them into new variables called principal components. These components are created by combining the original variables in a smart way to capture the most important patterns and variations in the data. By choosing the principal components that explain the most variation, we can reduce the number of predictors while preserving important information. 

<!-- PCA transforms the original predictor variables into uncorrelated variables known as principal components. These components capture the maximum variance in the data and are obtained by linearly combining the original variables. By selecting the principal components that explain the most variance, we can effectively reduce the dimensionality of the data while preserving crucial information. -->

## Optimization and Linear Regression
We looked at how the normal equation can be used to solve for the coefficient estimates above. Now, we will discuss a different method that uses an optimization technique called Gradient Descent (GD). GD is an iterative method that starts with a random initiation of the $\beta$ estimates and iteratively updates the estimates until convergence. Let us introduce a few notations for GD:

1. $\beta^{(i)}$: the current $\beta$ estimate
2. $\beta^{(i+1)}$: the new estimate obtained from $\beta$ after one iteration of GD
3. $\lambda$: the learning rate or step size
4. $\nabla L(y,\hat{y})$ : the gradient of the loss function. This is a vector of derivatives of the loss function w.r.t. $\beta$.
 
 We update $\beta$ in the opposite direction of the gradient of the loss function:
 
  $$\beta^{(i+1)} = \beta^{(i)} - \lambda \nabla L(y, \hat{y}) $$
 
Let's use the same example above. Recall our dataset

Person     | Age        | Height     | Weight
---------- | :---------:| :----------| ----------:
1          | 20         | 170        | 60
2          | 6          | 45         | 20
3          | 40         | 160        | 50
4          | 12         | 150        | 40

We want to predict $\beta = [ \beta_0  \quad \beta_1 \quad \beta_2]^T$ using this data and gradient descent.

### GD Step 1: Initialize the estimates
First, let us start with a random initiation of $\beta$. This can be anything, but let's initialize it to be a vector of zeros, $\beta^{(0)} = [ 0 \quad 0 \quad 0]^T$.

### GD Step 2: Calculate gradient of the loss function using the current estimates
Now, we need to find the gradient of the loss function, $\nabla L(y,\hat{y})$ . This is the same as taking the derivative of the loss function w.r.t. $\beta$. Luckily, we know from the derivation of the Normal Equation above that 

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
-65.5840
\end{bmatrix}$$

### GD Step 3: Update the estimates
Let the step size $\lambda = 0.01$. We can adjust the step size to make the coefficients converge faster. However, we should be careful about using step size that is too large as we might miss the optimal solution. We will update the new estimate using our update rule:

$$\beta^{(1)} = \beta^{(0)} - \lambda \nabla L(y, \hat{y}) $$
$$\beta^{(1)} = \begin{bmatrix}
  0\\
  0\\
  0
  \end{bmatrix} - 0.01  \begin{bmatrix}-156\\
-64.9157\\
-65.5840
\end{bmatrix}$$\\
$$\beta^{(1)} = \begin{bmatrix} 1.56\\
0.6492\\
0.6558
\end{bmatrix} $$

### GD Step 4: Repeat until convergence
We repeat steps 2 and 3 to update our estimates until $\beta^{(i+1)} = \beta^{(i)}$. Note that we need to use the current $\beta^{(i)}$ to compute the gradient for each iteration. After 1000 iterations, we get the estimates $\beta = \begin{bmatrix} 19.5\\
3.5544\\
4.8679\end{bmatrix}$.

## Understanding Gradient Descent

Think of gradient descent as trying to find the lowest point on a hill. Imagine that the shape of the hill represents our loss function, which measures how far off the predicted is from the actual response. Our goal is to minimize the loss. So, we want to reach the bottom of the hill, which corresponds to the optimal solution.

To start, we pick a random position on the hill (randomly initialize our $\beta$. Now, in order to descend to the bottom, we need to move in the direction of the steepest descent. The gradient of the loss function tells us the direction of the steepest ascent, so we actually want to go in the opposite direction of the gradient.

We take steps down the hill, and the size of each step is determined by $\lambda$, the learning rate or the step size. We want to reach the bottom as quickly as possible, but we need to keep in mind the value for $\lambda$. If $\lambda$. is too large, we might overshoot the bottom of the hill and miss the optimal solution. On the other hand, if the learning rate is too small, it will take us a very long time to reach the bottom because our steps are too tiny and we converge very slowly.

![alt text](https://github.com/dnnguyen99/dnnguyen99.github.io/blob/gh-pages/assets/img/gd.jpg?raw=true){:width="600px"}


In summary, gradient descent is like descending a hill to find the lowest point. We start at a random position, move in the opposite direction of the gradient (which points uphill), and adjust the size of our steps using the learning rate. The aim is to find the optimal solution efficiently without overshooting or converging too slowly.

