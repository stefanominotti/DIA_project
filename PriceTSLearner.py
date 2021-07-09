import numpy as np
from PriceLearner import PriceLearner


class PriceTSLearner(PriceLearner):
    def __init__(self, arms, returns_horizon):
        super().__init__(arms, returns_horizon)
        self.beta_distribution_per_arm = np.ones((len(self.arms), 2))

    def pull_arm(self):
        returns_samples = np.array([estimator.rvs() for estimator in self.returns_estimators])
        priced_samples = np.random.beta(self.beta_distribution_per_arm[:, 0], self.beta_distribution_per_arm[:, 1]) * self.arms * (1 + returns_samples)
        return self.arms[np.random.choice(np.argwhere(priced_samples == priced_samples.max()).reshape(-1))]
 
    def update(self, customers, returns=[]):
        for customer in customers:
            self.update_observations(customer)
            arm_idx = self.arms.index(customer.conversion_price)
            self.beta_distribution_per_arm[arm_idx, 0] += customer.conversion
            self.beta_distribution_per_arm[arm_idx, 1] += (1.0 - customer.conversion)
        self.update_returns(returns)

    def get_optimal_arm(self):
        alpha = self.beta_distribution_per_arm[:, 0] 
        beta = self.beta_distribution_per_arm[:, 1]
        returns_means = np.array([estimator.mean() for estimator in self.returns_estimators])
        return self.arms[np.argmax(alpha / (alpha + beta) * self.arms * (1 + returns_means))]