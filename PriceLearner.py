import numpy as np
from abc import ABC, abstractmethod
from ReturnsEstimator import ReturnsEstimator


class PriceLearner(ABC):
    def __init__(self, arms, returns_horizon):
        self.total_customers = 0
        self.arms = arms
        self.returns_horizon = returns_horizon
        self.samples_per_arm = np.zeros(len(self.arms))
        self.collected_conversions_per_arm = [[] for _ in range(len(self.arms))]
        self.returns_estimators = [ReturnsEstimator(returns_horizon) for _ in range(len(self.arms))]

    def update_observations(self, customer):
        arm_idx = self.arms.index(customer.conversion_price)
        self.total_customers += 1
        self.samples_per_arm += 1
        self.collected_conversions_per_arm[arm_idx].append(customer.conversion)
        self.returns_estimators[arm_idx].new_customer(customer)

    def update_returns(self, returns):
        for customer in returns:
            arm_idx = self.arms.index(customer.conversion_price)
            self.returns_estimators[arm_idx].new_return(customer)

    @abstractmethod
    def pull_arm(self):
        pass

    @abstractmethod
    def update(self, customers, returns=[]):
        pass
    