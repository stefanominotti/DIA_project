import numpy as np
from PriceLearner import PriceLearner


class PriceUCBLearner(PriceLearner):
    def __init__(self, arms):
        super().__init__(arms)
        self.expected_reward_per_arm = np.zeros(len(self.arms))
        self.upper_bound_per_arm = np.ones(len(self.arms))

    def pull_arm(self):
        if self.t < len(self.arms):
            return self.arms[np.random.choice(np.argwhere(self.rounds_per_arm == 0).reshape(-1))]

        priced_upper_bound = self.upper_bound_per_arm * self.arms
        return self.arms[np.random.choice(np.argwhere(priced_upper_bound == priced_upper_bound.max()).reshape(-1))]

    def update(self, pulled_arm, reward):
        arm_idx = self.arms.index(pulled_arm)
        self.update_observations(arm_idx, reward)

        for customer in reward:
            self.expected_reward_per_arm[arm_idx] = (self.expected_reward_per_arm[arm_idx] * (self.observations_per_arm[arm_idx] - 1) + customer.conversion) / self.observations_per_arm[arm_idx]

        self.upper_bound_per_arm = self.expected_reward_per_arm + np.sqrt((2 * np.log(self.total_observations) / self.observations_per_arm))
