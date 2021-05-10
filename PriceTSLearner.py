import numpy as np
from PriceLearner import PriceLearner


class PriceTSLearner(PriceLearner):
    def __init__(self, arms):
        super().__init__(arms)
        self.beta_distribution_per_arm = np.ones((len(self.arms), 2))

    def pull_arm(self):
        priced_beta_samples = np.random.beta(self.beta_distribution_per_arm[:, 0], self.beta_distribution_per_arm[:, 1]) * self.arms

        return self.arms[np.random.choice(np.argwhere(priced_beta_samples == priced_beta_samples.max()).reshape(-1))]

    def update(self, pulled_arm, reward):
        arm_idx = self.arms.index(pulled_arm)
        self.update_observations(arm_idx, reward)

        for customer in reward:
            self.beta_distribution_per_arm[arm_idx, 0] += customer.conversion
            self.beta_distribution_per_arm[arm_idx, 1] += (1 - customer.conversion)
