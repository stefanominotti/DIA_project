import numpy as np
from abc import ABC, abstractmethod


class PriceLearner(ABC):
    def __init__(self, arms):
        self.t = 0
        self.arms = arms
        self.pulled_arms = []
        self.collected_rewards = []
        self.rewards_per_arm = [[] for _ in range(len(self.arms))]
        self.rounds_per_arm = np.zeros(len(self.arms))
        self.total_observations = 0
        self.observations_per_arm = np.zeros(len(self.arms))

    def update_observations(self, arm_idx, reward):
        self.t += 1

        self.pulled_arms.append(self.arms[arm_idx])
        self.collected_rewards.append(reward)
        self.rewards_per_arm[arm_idx].append(reward)

        self.rounds_per_arm[arm_idx] += 1
        self.total_observations += len(reward)
        self.observations_per_arm[arm_idx] += len(reward)

    @abstractmethod
    def pull_arm(self):
        pass

    @abstractmethod
    def update(self, pulled_arm, reward):
        pass
    