import numpy as np
from PriceLearner import PriceLearner


class PriceUCBLearner(PriceLearner):
    def __init__(self, arms):
        super().__init__(arms)
        self.expected_conversion_per_arm = np.zeros(len(self.arms))
        self.upper_bound_per_arm = np.ones(len(self.arms))

    def pull_arm(self):
        if len(np.argwhere(self.rounds_per_arm == 0)) != 0:
            return self.arms[np.random.choice(np.argwhere(self.rounds_per_arm == 0).reshape(-1))]

        priced_upper_bound = self.upper_bound_per_arm
        return self.arms[np.random.choice(np.argwhere(priced_upper_bound == priced_upper_bound.max()).reshape(-1))]

    def update(self, customers):
        for customer in customers:
            self.update_observations(customer)
            arm_idx = self.arms.index(customer.conversion_price)
            self.rewards_per_arm[arm_idx].append(customer.conversion * customer.conversion_price * (1 + customer.returns_count))
            self.expected_conversion_per_arm[arm_idx] = (self.expected_conversion_per_arm[arm_idx] * (self.rounds_per_arm[arm_idx] - 1) + self.rewards_per_arm[arm_idx][-1]) / self.rounds_per_arm[arm_idx]
            self.upper_bound_per_arm = self.expected_conversion_per_arm + np.sqrt((2 * np.log(self.t) / self.rounds_per_arm))

    def get_optimal_arm(self):
        return self.arms[np.argmax(self.expected_conversion_per_arm)]