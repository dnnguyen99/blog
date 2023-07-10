---
layout: post
title: "The Complex Beauty of the Mandelbrot Set"
author: "Diep Nguyen"
tags: [math, tableau]
categories: journal
image: lr3_bg.jpg
---
> In this article, we will explore the artistic component of mathematics. Specifically, we will look at the Mandelbrot set-- a set of complex numbers that exhibits artistic beauty.

 
## What Constitutes the Mandelbrot Set?

The Mandelbrot set is a mathematical set of complex numbers. When we plot the points inside and outside this set we can observe mesmerizing fractal patterns. The Mandelbrot set is defined by the behavior of the recursive equation $z = z^2 + c$, where $z$ and $c$ are complex numbers. If $c$ is in the Mandelbrot set, then 

## Generating the Mandelbrot Set
To determine whether a complex number $c$ is in the Mandelbrot set, the iteration starts with $z = 0$ and repeatedly computes the equation $z = z^2 + c$ until one of the two conditions is met:
When the magnitude of z exceeds a certain threshold (usually 2)
When the number of iterations (`num_iter`) reaches a specified maximum number of iterations (`max_iter`)

If the magnitude of $z$ is greater than $2$ before we reach the maximum number of iterations, it is safe to say that $z$ will probably go to infinity. So, given a complex number $c$, we will iteratively compute $z = z^2 + c$. If $\lvert z \rvert > 2$ while `num_iter` $<$ `max_iter`, then $c$ is not in the Mandelbrot set. Conversely, if we keep iteratively calculating  $z = z^2 + c$ and observe that the magnitude of $z$ is less than $2$ even after we reach the maximum number of iterations, then $c$ is in the Mandelbrot set. 

Below is the pseudo-code and a diagram illustrating the process of generating the Mandelbrot set.

`initialize z = 0 
initialize num_iter = 0
while |z| <= 2 AND num_iter < max_iter:
z = z * z + c
	num_iter += 1
return num_iter`

All the points with `num_iter == max_iter` are the points in the Mandelbrot set. Note that in theory, we will need to test every point $c$ in the complex plane. However, due to the infinite nature of the complex plane, we cannot test every point. Practically, we will refine our window to a finite region in the complex plane and compute the Mandelbrot set for a discrete set of points in that region. Typically, we bound a region with real and imaginary axes. In this article, the region is defined by the boundaries $-2$ to $1$ for the real (horizontal) axis, and $-1$ to $1$ for the imaginary (vertical) axis.

We will generate a $600 \times 400$ pixels plot. So, we will need to convert each pixel coordinate $(x,y)$ to its corresponding complex value $c$. I consulted this website (https://www.codingame.com/playgrounds/2358/how-to-plot-the-mandelbrot-set/mandelbrot-set) for the conversion and the codes. 

Insert diagram here

## Visualizing the Mandelbrot Set in Tableau

I will use Tableau to visualize the Mandelbrot set. You can directly plot the set in Python. However, I prefer Tableau since it is a useful tool for data analysis (in my opinion).
 
As we can see, all the points in the Mandelbrot set (all the points with `num_iter` == `max_iter`) are represented by one color (blue). The other points whose `num_iter` $<$ `max_iter` are not in the set. Depending on the actual value of `num_iter`, we color-code using different shades of green. 

The boundaries between the points inside and outside the Mandelbrot set exhibit very intriguing fractal shapes. 

<iframe src="https://public.tableau.com/views/MandelbrotSet_16889171113990/Sheet1?:showVizHome=no&:embed=true"
 width="645" height="955"></iframe>

https://public.tableau.com/views/MandelbrotSet_16889171113990/Sheet1?:language=en-GB&:display_count=n&:origin=viz_share_link

## Exploring Mandelbrot Set: The Multibrot
The recursive equation $z = z^2 + c$ above can be modified and might yield different 
