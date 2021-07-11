import numpy as np
from PriceLearner import PriceLearner


class PriceUCBLearner(PriceLearner):
    def __init__(self, arms, returns_horizon):
        super().__init__(arms, returns_horizon)
        self.expected_conversion_per_arm = np.zeros(len(self.arms))
        self.conversion_upper_bound_per_arm = np.ones(len(self.arms))

    def pull_arm(self):
        returns_sample_per_arm = np.array([estimator.total_customers for estimator in self.returns_estimators])

        if len(np.argwhere(self.samples_per_arm == 0)) != 0:
            return self.arms[np.random.choice(np.argwhere(self.samples_per_arm == 0).reshape(-1))]

        returns_means_per_arm = np.array([estimator.mean() for estimator in self.returns_estimators])        
        returns_upper_bound_per_arm = returns_means_per_arm + self.returns_horizon * np.sqrt(2 * np.log(returns_sample_per_arm) / returns_sample_per_arm.sum())
        reward_upper_bound_per_arm = self.conversion_upper_bound_per_arm * np.array(self.arms) * (1 + returns_upper_bound_per_arm)
        return self.arms[np.random.choice(np.argwhere(reward_upper_bound_per_arm == reward_upper_bound_per_arm.max()).reshape(-1))]

    def update(self, customers, returns=[]):
        for customer in customers:
            self.update_observations(customer)

        self.expected_conversion_per_arm = [np.mean(collected_conversions) for collected_conversions in self.collected_conversions_per_arm]
        self.conversion_upper_bound_per_arm = self.expected_conversion_per_arm + np.sqrt(2 * np.log(self.total_customers) / self.samples_per_arm)
        self.update_returns(returns)

    def get_optimal_arm(self):
        returns_means = np.array([estimator.mean() for estimator in self.returns_estimators])
        return self.arms[np.argmax(self.expected_conversion_per_arm * np.array(self.arms) * (1 + returns_means))]

    def get_expected_conversion_per_arm(self):
        return self.expected_conversion_per_arm.copy()