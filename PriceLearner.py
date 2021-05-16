import numpy as np
from abc import ABC, abstractmethod


class PriceLearner(ABC):
    def __init__(self, arms):
        self.t = 0
        self.arms = arms
        self.pulled_arms = []
        self.samples_per_arm = np.zeros(len(self.arms))
        self.rewards_per_arm = [[] for _ in range(len(self.arms))]

    def update_observations(self, customer):
        arm_idx = self.arms.index(customer.conversion_price)
        self.t += 1
        self.pulled_arms.append(self.arms[arm_idx])
        self.samples_per_arm[arm_idx] += 1

    @abstractmethod
    def pull_arm(self):
        pass

    @abstractmethod
    def update(self, customer):
        pass
    