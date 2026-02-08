# Linear_Regression_Scratch
This repository contains a first-principles implementation of a Multiple Linear Regression model. Instead of utilizing high-level machine learning libraries like Scikit-Learn, I developed this engine using Linear Algebra and Vectorized Matrix Operations via NumPy.

The project demonstrates the ability to bridge the gap between statistical theory and computational implementation.
## Mathematics
To find the optimal coefficients ($\hat{\beta}$) that minimize the sum of squared residuals, I implemented the Normal Equation:
$$\hat{\beta} = (X^T X)^{-1} X^T y$$
### Key Mathematical Steps:
Feature Augmentation
Matrix Transposition
Matrix Inversion
Vectorized Prediction
## Technical Features
Object-Oriented Programming: Encapsulated the logic within a LinearModel class.

Vectorization: Optimized for performance using NumPy's C-based array operations rather than Python loops.

Data Simulation: Includes a built-in data generator using Gaussian noise to simulate real-world data variance.

## How to Run
Ensure you have NumPy installed: pip install numpy

Run the script: 
python linear_regression_scratch.py
