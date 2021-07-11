import numpy as np
from PriceLearner import PriceLearner


class PriceTSLearner(PriceLearner):
    def __init__(self, arms, returns_horizon):
        super().__init__(arms, returns_horizon)
        self.beta_distribution_per_arm = np.ones((len(self.arms), 2))

    def pull_arm(self):
        returns_samples = np.array([estimator.sample() for estimator in self.returns_estimators])
        reward_samples = np.random.beta(self.beta_distribution_per_arm[:, 0], self.beta_distribution_per_arm[:, 1]) * np.array(self.arms) * (1 + returns_samples)
        return self.arms[np.random.choice(np.argwhere(reward_samples == reward_samples.max()).reshape(-1))]
 
    def update(self, customers, returns=[]):
        for customer in customers:
            self.update_observations(customer)
            arm_idx = self.arms.index(customer.conversion_price)
            self.beta_distribution_per_arm[arm_idx, 0] += customer.conversion
            self.beta_distribution_per_arm[arm_idx, 1] += (1.0 - customer.conversion)
        self.update_returns(returns)

    def get_optimal_arm(self):
        returns_means = np.array([estimator.mean() for estimator in self.returns_estimators])
        return self.arms[np.argmax(self.get_expected_conversion_per_arm() * np.array(self.arms) * (1 + returns_means))]

    def get_expected_conversion_per_arm(self):
        alpha = self.beta_distribution_per_arm[:, 0]
        beta = self.beta_distribution_per_arm[:, 1]
        return alpha / (alpha + beta)