import numpy as np
from PriceLearner import PriceLearner


class PriceUCBLearner(PriceLearner):
    def __init__(self, arms, context):
        super().__init__(arms, context)
        self.expected_reward_per_arm = np.zeros(len(self.arms))
        self.upper_bound_per_arm = np.ones(len(self.arms))

    def pull_arm(self):
        if self.t < len(self.arms):
            return self.arms[np.random.choice(np.argwhere(self.rounds_per_arm == 0).reshape(-1))]

        priced_upper_bound = self.upper_bound_per_arm * self.arms
        return self.arms[np.random.choice(np.argwhere(priced_upper_bound == priced_upper_bound.max()).reshape(-1))]

    def update(self, customers):
        for customer in customers:
            self.update_observations(customer)
            arm_idx = self.arms.index(customer.conversion_price)
            self.expected_reward_per_arm[arm_idx] = (self.expected_reward_per_arm[arm_idx] * (self.rounds_per_arm[arm_idx] - 1) + customer.conversion) / self.rounds_per_arm[arm_idx]
            self.upper_bound_per_arm = self.expected_reward_per_arm + np.sqrt((2 * np.log(self.t) / self.rounds_per_arm))