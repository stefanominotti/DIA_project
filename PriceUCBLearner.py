import numpy as np
from PriceLearner import PriceLearner


class PriceUCBLearner(PriceLearner):
    def __init__(self, arms, returns_horizon):
        super().__init__(arms, returns_horizon)
        self.expected_conversion_per_arm = np.zeros(len(self.arms))
        self.conversion_upper_bound_per_arm = np.ones(len(self.arms))
        self.pulled_once = []

    def pull_arm(self):
        not_pulled_arms = list(set(self.arms) - set(self.pulled_once))
        if len(not_pulled_arms) > 0:
            arm = np.random.choice(not_pulled_arms)
            self.pulled_once.append(arm)
            return arm

        if len(np.argwhere(self.samples_per_arm == 0)) != 0:
            return self.arms[np.random.choice(np.argwhere(self.samples_per_arm == 0).reshape(-1))]

        returns_means = np.array([estimator.mean() for estimator in self.returns_estimators])
        returns_sample_per_arm = np.array([estimator.total_customers for estimator in self.returns_estimators])
        returns_upper_bound_per_arm = returns_means + self.returns_horizon * np.sqrt((2 * np.log(returns_sample_per_arm.sum()) / returns_sample_per_arm))

        priced_upper_bound_per_arm = self.conversion_upper_bound_per_arm * self.arms * (1 + returns_upper_bound_per_arm)
        return self.arms[np.random.choice(np.argwhere(priced_upper_bound_per_arm == priced_upper_bound_per_arm.max()).reshape(-1))]

    def update(self, customers, returns=[]):
        for customer in customers:
            self.update_observations(customer)
            arm_idx = self.arms.index(customer.conversion_price)
            self.expected_conversion_per_arm[arm_idx] = (self.expected_conversion_per_arm[arm_idx] * (self.samples_per_arm[arm_idx] - 1) + customer.conversion) / self.samples_per_arm[arm_idx]
            self.conversion_upper_bound_per_arm = self.expected_conversion_per_arm + np.sqrt((2 * np.log(self.total_customers) / self.samples_per_arm))
        self.update_returns(returns)

    def get_optimal_arm(self):
        returns_means = np.array([estimator.mean() for estimator in self.returns_estimators])
        return self.arms[np.argmax(self.expected_conversion_per_arm * self.arms * (1 + returns_means))]