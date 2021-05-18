import numpy as np
from PriceLearner import PriceLearner


class PriceTSLearner(PriceLearner):
    def __init__(self, arms):
        super().__init__(arms)
        self.beta_distribution_per_arm = np.ones((len(self.arms), 2))
        self.mean_returns = 0
        self.sigma_returns = 1
        self.returns_counts = []

    def pull_arm(self):
        priced_samples = np.random.beta(self.beta_distribution_per_arm[:, 0], self.beta_distribution_per_arm[:, 1]) * (1 + np.random.normal(self.mean_returns, self.sigma_returns)) * self.arms
        return self.arms[np.random.choice(np.argwhere(priced_samples == priced_samples.max()).reshape(-1))]
 
    def update(self, customers):
        for customer in customers:
            self.update_observations(customer)
            arm_idx = self.arms.index(customer.conversion_price)
            self.rewards_per_arm[arm_idx].append(customer.conversion * customer.conversion_price * (1 + customer.returns_count))
            self.beta_distribution_per_arm[arm_idx, 0] += customer.conversion
            self.beta_distribution_per_arm[arm_idx, 1] += (1.0 - customer.conversion)
            if customer.conversion:
                self.returns_counts.append(customer.returns_count)
        self.mean_returns = np.mean(self.returns_counts)
        n_samples = self.samples_per_arm[arm_idx]
        if n_samples > 1:
            self.sigma_returns = np.std(self.returns_counts) / n_samples

    def get_optimal_arm(self):
        alpha = self.beta_distribution_per_arm[:, 0] 
        beta = self.beta_distribution_per_arm[:, 1] 
        return self.arms[np.argmax(alpha / (alpha + beta) * (1 + self.mean_returns) * self.arms)]