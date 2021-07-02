import numpy as np
from abc import ABC, abstractmethod
from ReturnsEstimator import ReturnsEstimator


class PriceLearner(ABC):
    def __init__(self, arms, returns_horizon):
        self.t = 0
        self.arms = arms
        self.pulled_arms = []
        self.samples_per_arm = np.zeros(len(self.arms))
        self.rewards_per_arm = [[] for _ in range(len(self.arms))]
        self.returns_estimator = ReturnsEstimator(returns_horizon)
        self.returns_horizon = returns_horizon
        self.returns_horizon_customers = []

    def update_observations(self, customer):
        arm_idx = self.arms.index(customer.conversion_price)
        self.t += 1
        self.pulled_arms.append(self.arms[arm_idx])
        self.samples_per_arm[arm_idx] += 1

    def update_returns(self, customers, returns):
        self.returns_estimator.update(list(filter(lambda customer: customer.conversion == 1, customers)), returns)

    def update_rewards(self, customers):
        self.returns_horizon_customers.append(customers)
        if len(self.returns_horizon_customers) > self.returns_horizon:
            delayed_customers = self.returns_horizon_customers.pop(0)
            for customer in delayed_customers:
                arm_idx = self.arms.index(customer.conversion_price)
                self.rewards_per_arm[arm_idx].append(customer.conversion * customer.conversion_price * (1 + customer.returns_count))

    @abstractmethod
    def pull_arm(self):
        pass

    @abstractmethod
    def update(self, customers, returns):
        pass
    