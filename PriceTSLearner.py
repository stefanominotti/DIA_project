import numpy as np
from PriceLearner import PriceLearner


class PriceTSLearner(PriceLearner):
    def __init__(self, arms, returns_horizon):
        super().__init__(arms, returns_horizon)
        self.beta_distribution_per_arm = np.ones((len(self.arms), 2))

    def pull_arm(self):
        priced_samples = np.random.beta(self.beta_distribution_per_arm[:, 0], self.beta_distribution_per_arm[:, 1]) * (1 + self.returns_estimator.weihgted_sum()) * self.arms
        return self.arms[np.random.choice(np.argwhere(priced_samples == priced_samples.max()).reshape(-1))]
 
    def update(self, customers, returns):
        self.update_returns(customers, returns)
        for customer in customers:
            self.update_observations(customer)
            arm_idx = self.arms.index(customer.conversion_price)
            self.beta_distribution_per_arm[arm_idx, 0] += customer.conversion
            self.beta_distribution_per_arm[arm_idx, 1] += (1.0 - customer.conversion)
        self.update_rewards(customers)

    def get_optimal_arm(self):
        alpha = self.beta_distribution_per_arm[:, 0] 
        beta = self.beta_distribution_per_arm[:, 1] 
        return self.arms[np.argmax(alpha / (alpha + beta) * (1 + self.returns_estimator.weihgted_sum()) * self.arms)]