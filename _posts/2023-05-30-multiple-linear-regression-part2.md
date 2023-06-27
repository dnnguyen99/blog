---
layout: post
title: "Multiple Linear Regression (Part 2)"
author: "Diep Nguyen"
tags: [linear regression, code]
categories: journal
image: lr2_bg.jpg
---
> In this article, we will explore the common metrics that help us assess the performance of our linear regression model. We will also look at an implementation of linear regression using the `scikit-learn` library in Python. Additionally, we will discuss the assumptions underlying linear regression and discuss when linear regression is useful in real-world scenarios.


## Common Regression Evaluation Metrics: 


Linear regression models are used to predict quantitative values, such as house prices or a person's height. After building a model, it is important to evaluate its performance before deploying it. Various performance metrics can be used for this purpose. However, before discussing these metrics, let's first address the process of splitting data for model training and testing.

During training, the model's performance on the training set is not a reliable measure of its generalization ability. This is because the model has already seen and learned from this data, making it prone to overfitting. To understand this, let's consider a scenario where a professor provides a set of practice problems to students a week before an exam, and then uses the exact same problems for the actual exam. In such a case, students who studied those specific problems would perform well because they had seen and practiced them before. To truly assess the student's knowledge, the professor should offer a set of different but related problems on the exam. Similarly, in model evaluation, we aim to test the model's performance on similar but unseen data, which simulates real-world scenarios. 

In supervised learning, dividing the data into train, validation, and test sets is essential for proper model evaluation and generalization. The training set is used to train the model by learning patterns and relationships between predictors (columns of the design matrix, denoted as $X$) and the corresponding response (denoted as $y$). The validation set is used to fine-tune the model and its hyperparameters, if applicable. By evaluating the model's performance on the validation set, adjustments can be made, and parameters can be optimized. It is important to note that the model does not directly learn from the validation set but the evaluation still impacts the model's performance. Note that if the model does not have any hyperparameters, as in the case of linear regression, we can omit the validation set. 

Finally, the test set is used to assess the model's performance on unseen data. By evaluating the model on this new dataset, we gain insights into how well the model generalizes to real-world examples. Now that we have covered the train/validation/test split, we can look at some common evaluation metrics used in the model assessment process. Note that these metrics should be used on the test set and not on the train set.

## Mean Squared Error (MSE) 
Mean squared error (`sklearn.metrics.mean_squared_error(y_true, y_pred)`), is one of the most common metrics for evaluating a regression model. The MSE measures how far off our regression line is from the true response values. The way to calculate the MSE is in its name, we need to first square all the ‘errors’ (the difference between the true and predicted values), and then take the mean. The formula is given below.

$$MSE = \frac{1}{n} \sum{i = 1}^{n} (y_i - \hat{y_i})^2$$ 
If the predicted values are the same as the true values, the MSE is equal to 0. The larger the distance between the predicted and the true values, the larger the MSE. Having a low MSE means the model’s predictions are close to the actual values, and having a large MSE means that our model gives predictions that are very different from the true values. Because the error or the difference is squared, MSE gives higher weightage to larger errors. So, this metric is more sensitive to outliers or large errors in the data. This metric is helpful when we want to place emphasis on larger errors. 

## Mean Absolute Error (MAE)
Mean Absolute Error (`sklearn.metrics.mean_absolute_error(y_true, y_pred)`) looks at the average errors (in absolute value) between the predicted and the actual values. Unlike MSE, the Mean Absolute Error metric treats all errors equally. Since MAE takes the absolute value of the difference instead of the square, it is not influenced by outliers or large errors like MSE. Thus, this metric is good when we expect to have significant outliers in our dataset, or when we want to use a metric that is less affected by extreme error values. 

$$MSE = \frac{1}{n} \sum{i = 1}^{n} \lvert y_i - \hat{y_i}\rvert$$ 

## R-squared 
R-squared (`sklearn.metrics.r2_score(y_true, y_pred)`) is also known as the coefficient of determination or $R^2$. This measurement tells us the proportion of the variance in the response variable ($y$) that can be explained by the predictors ($X$). This metric ranges between 0 and 1 and helps us assess how well the model fits the test data. Having a high R-squared value means that the model gives a good fit since it explains a larger proportion of variability in the response. Conversely, having a low R-squared value means that the model explains less of the variability and does not give good predictions. To compute R-squared, we first need to find the sum of squared errors (SSE) as well as the total sum of squares (SST) between the actual and the predicted values.

$$R^2 = 1 - \frac{SSE}{SST} = 1 -  \frac{\sum (y - \hat{y})^2}{\sum (y - \bar{y})^2}$$

Where $\bar{y}$ is the average of all the response variables. 

## Linear Regression Assumptions
Before fitting a model, we need to make sure that our data meets all of the assumptions to perform linear regression.

$$ \begin{itemize}
\item Since we are looking at a regression problem, we need to ensure that our response $y$ is a continuous value. 
\item We want to use a straight line to model the relationship between the response and the predictors in linear regression. So, we need to make sure that there is a linear relationship between each predictor and the response. We can quickly check for this assumption using a scatterplot. 
\item All observations should be independent. In other words, the value of one observation should not be influenced by the value of another observation.  ​​
\item There should be no significant outliers.
\item There should be mild or no correlation between the predictors. For example, we cannot predict a person’s age using both height and weight since the 2 predictors are highly correlated. This phenomenon is known as multi-collinearity. 
\item The variance of the errors (the difference between $\hat{y}$ and $y$) should be constant across all levels of the independent variables. This is known as homoscedasticity and we should check for this assumption after we fit the model by plotting the errors against the predicted values. If the spread of the errors appears to follow a particular pattern (e.g., a funnel shape), it means that this assumption is violated. 
\item The errors should be normally distributed. To check for this assumption, we can look at the histogram of the errors after fitting the model. The histogram should resemble a bell-shaped curve. 
\end{itemize}$$

Note that we used a lot of visualization techniques to check for the assumptions above. In reality, visual assessment might not be conclusive and we should perform other statistical tests. 

## Multiple Linear Regression with `scikit-learn`

For this demonstration, we will use one of `sckikit-learn` toy datasets called `diabetes`. This dataset has a total of 442 observations. The first 10 columns of the data are predictors such as BMI value, average blood pressure, and blood serum measurements. We want to use these features to predict the 11th column, the target, which measures the disease progression one year after baseline. 

We will check for all the assumptions, split the data into a train and a test set, perform multiple linear regression, and evaluate the model’s performance using the above metrics. Lastly, we will link back to our previous post (link) and look at how we can obtain the $\beta$ estimates (commonly known as the regression coefficients) using linear algebra. 

The detailed codes and analysis can be found in this notebook 

