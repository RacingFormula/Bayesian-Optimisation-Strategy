import numpy as np
from scipy.optimize import minimize

class BayesianOptimisationStrategy:
    def __init__(self, objective_function, bounds, kernel=None):
        self.objective_function = objective_function
        self.bounds = bounds
        self.kernel = kernel
        self.samples = []
        self.values = []

    def suggest_next_point(self):
        if not self.samples:
            return np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])

        def acquisition_function(x):
            mean, variance = self._predict(x)
            return -1 * (mean + 1.96 * np.sqrt(variance))

        result = minimize(
            acquisition_function,
            x0=np.random.uniform(self.bounds[:, 0], self.bounds[:, 1]),
            bounds=self.bounds,
            method='L-BFGS-B'
        )
        return result.x

    def _predict(self, x):
        if self.kernel is None:
            raise ValueError("A kernel function must be provided for predictions.")
        x = np.atleast_2d(x)
        
        # Compute kernel matrices
        K = self.kernel(self.samples, self.samples) + 1e-6 * np.eye(len(self.samples))
        K_s = self.kernel(self.samples, x)
        K_ss = self.kernel(x, x) + 1e-6

        K_inv = np.linalg.inv(K)

        # Mean and variance of the predictive distribution
        mean = K_s.T.dot(K_inv).dot(self.values)
        variance = K_ss - K_s.T.dot(K_inv).dot(K_s)

        return mean.flatten(), np.diag(variance)

    def update(self, x, y):
        self.samples.append(x)
        self.values.append(y)

if __name__ == "__main__":
    # Example usage

    def objective_function(x):
        return -1 * (np.sin(3 * x) + x**2 - 0.7 * x)

    def rbf_kernel(x1, x2, length_scale=1.0):
        x1 = np.atleast_2d(x1)
        x2 = np.atleast_2d(x2)
        dist = np.sum((x1[:, None] - x2[None, :])**2, axis=2)
        return np.exp(-0.5 / length_scale**2 * dist)

    bounds = np.array([[0, 2]])
    strategy = BayesianOptimisationStrategy(objective_function, bounds, kernel=rbf_kernel)

    for _ in range(10):
        x_next = strategy.suggest_next_point()
        y_next = objective_function(x_next)
        strategy.update(x_next, y_next)
        print(f"Sampled at {x_next}, value: {y_next}")
