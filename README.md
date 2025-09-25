# DeepLearningBishopFigures

This repository contains my self-driven project to study and reproduce key examples from **Christopher Bishop’s *Deep Learning Foundations and Concepts***. The goal is to deepen understanding of fundamental deep learning concepts by implementing them in Python/Jupyter notebooks and visualizing results.

## Overview

During this project, I focused on reproducing several core examples from Bishop’s book:

* **Curve fitting**: Linear and nonlinear regression examples, exploring model complexity and overfitting.
* **Sinusoidal interpolation**: Reproducing interpolation examples to understand basis functions and function approximation.
* **Density transformations**: Visualizing probability density transformations to gain intuition on probabilistic modeling.

The project emphasizes **reproducibility**, **clarity**, and **visual intuition**, rather than creating production-level code. Each notebook includes explanations, plots, and step-by-step derivations of the underlying mathematics.

## Notebooks

- `Chap2-Transformation-Densities.ipynb` - Transformation of densities have interesting property where they're not quite the same as just regular functions (multiplied by the Jacobian).

![image](./chap2-transform-density.png)

- `sin_func_interpolation.jl` - A [Pluto.jl](https://plutojl.org/) notebook attempting to reproduce a **Tutorial Example** Section `1.2` from the book.
![image](./sin-func-interpolation_demo.gif)

- `probabilities-chap2.jl` - Also a Pluto notebook demonstrating how bias occurs when you calculate sample variance using the sample mean. It's discussed in Chapter 2, Section `2.3.3` **Bias of maximum likelihood**. As shown in the gif below, if the number of data points is low (equals 2) then the variance is off by a huge amount. The blue bell shaped curve is quite narrow. But as number of data points increase, this effect is less pronounced. Mathematically:

$$ E[\sigma^2_{ML}] = (\frac{N - 1}{N})\sigma^2 $$

As you can see from the equation above, when N=2, then maximum likelihood variance is off from the real variance by a factor of 1/2! But as $N -> \infty$ then this effect is not noticable. And to make the variance unbiased we use `1/(N-1)` instead of `1/N` and it's called **Bessel correction**.
![image](./bessel_correction_demo.gif)

## License

MIT License — free to use for educational purposes.
