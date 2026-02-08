import numpy as np
import random

class LinearModel:
    def __init__(self, n_samples, n_features):
        self.n_samples = n_samples
        self.n_features = n_features
        self.X = None
        self.Y = None
        self.beta = None
                
    def Matrix_Setup(self):
        self.X = np.random.rand(self.n_samples, self.n_features)
        self.X_new = np.random.rand(self.n_samples, self.n_features)
        true_betas = np.array([3,5])
        intercept = 10
        noise = np.random.normal(0,1, self.n_samples)
        self.y = (self.X @ true_betas) + intercept + noise
        print(f"Data generated: {self.n_samples} samples, {self.n_features} features.")

    def add_intercept(self, X):
        ones = np.ones((X.shape[0],1))
        return np.hstack([ones, X])

    def Fit(self):
        X_biased = self.add_intercept(self.X)
        X_biased_transposed = X_biased.T
        XTX_biased = X_biased_transposed@ X_biased
        XTX_inv = np.linalg.inv(XTX_biased)
        self.beta = XTX_inv @ X_biased_transposed @ self.y
        print(f"Model Training Complete")
        print(f"Calculated Intercept & Betas: {self.beta}")

    def Prediction_Logic(self, X_new):
        X_new_bias = self.add_intercept(X_new)
        y_hat = X_new_bias @ self.beta
        return y_hat

    def Evaluate(self):
        y_hat = self.Prediction_Logic(self.X)
        residuals = self.y - y_hat
        rss = np.sum(residuals**2)
        tss = np.sum((self.y - np.mean(self.y))**2)
        r_squared = 1-(rss/tss)
        return r_squared

if __name__ == "__main__":
    model = LinearModel(n_samples=500,n_features =2)
    model.Matrix_Setup()
    model.Fit()
    accuracy = model.Evaluate()
    print(f"R-Squared Score: {accuracy: 4f}")
    test_point = np.array([[0.5, 0.8]])
    prediction = model.Prediction_Logic(test_point)
    print(f"Prediction for test point: {prediction}")
