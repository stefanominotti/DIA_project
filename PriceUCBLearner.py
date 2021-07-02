import numpy as np
from PriceLearner import PriceLearner


class PriceUCBLearner(PriceLearner):
    def __init__(self, arms, returns_horizon):
        super().__init__(arms, returns_horizon)
        self.expected_conversion_per_arm = np.zeros(len(self.arms))
        self.conversion_upper_bound_per_arm = np.ones(len(self.arms))

    def pull_arm(self):
        if len(np.argwhere(self.samples_per_arm == 0)) != 0:
            return self.arms[np.random.choice(np.argwhere(self.samples_per_arm == 0).reshape(-1))]

        priced_upper_bound_per_arm = self.conversion_upper_bound_per_arm * (1 + self.returns_estimator.weihgted_sum()) * self.arms
        return self.arms[np.random.choice(np.argwhere(priced_upper_bound_per_arm == priced_upper_bound_per_arm.max()).reshape(-1))]

    def update(self, customers, returns):
        self.update_returns(customers, returns)
        for customer in customers:
            self.update_observations(customer)
            arm_idx = self.arms.index(customer.conversion_price)
            self.expected_conversion_per_arm[arm_idx] = (self.expected_conversion_per_arm[arm_idx] * (self.samples_per_arm[arm_idx] - 1) + self.expected_conversion_per_arm[arm_idx][-1]) / self.samples_per_arm[arm_idx]
            self.conversion_upper_bound_per_arm = self.expected_conversion_per_arm + np.sqrt((2 * np.log(self.t) / self.samples_per_arm))
        self.update_rewards(customers)

    def get_optimal_arm(self):
        return self.arms[np.argmax(self.expected_conversion_per_arm * (1 + self.returns_estimator.weihgted_sum()) * self.arms)]