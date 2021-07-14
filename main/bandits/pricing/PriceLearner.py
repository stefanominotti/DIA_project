import numpy as np
from abc import ABC, abstractmethod

from main.utils.ReturnsEstimator import ReturnsEstimator


class PriceLearner(ABC):
    """
    Abstract class for a price bandit learner
    """

    def __init__(self, arms, returns_horizon):
        """Class constructor

        Args:
            arms (list): set of arms to pull
            returns_horizon (integer): days horizon during which a customer can return to purchase after the first time
        """

        self.total_customers = 0
        self.arms = arms
        self.returns_horizon = returns_horizon
        self.samples_per_arm = np.zeros(len(self.arms))
        self.collected_conversions_per_arm = [[] for _ in range(len(self.arms))]
        self.returns_estimators = [ReturnsEstimator(returns_horizon) for _ in range(len(self.arms))]
        self.rewards_per_arm = [[] for _ in range(len(self.arms))]

    def update_observations(self, customer):
        """Update the estimations given a customer

        Args:
            customer (Customer): a collected customer sample
        """

        arm_idx = self.arms.index(customer.conversion_price)
        self.total_customers += 1
        self.samples_per_arm[arm_idx] += 1
        self.collected_conversions_per_arm[arm_idx].append(customer.conversion)
        self.returns_estimators[arm_idx].new_customer(customer)
        self.rewards_per_arm[arm_idx].append(customer.conversion * customer.conversion_price * (1 + customer.returns_count))
    
    @abstractmethod
    def pull_arm(self):
        """Pull an arm

        Returns:
            float: the pulled arm
        """

        pass

    @abstractmethod
    def update(self, customers):
        """Update the estimations given a set of daily customers

        Args:
            customers (list): the daily customers
        """

        pass

    @abstractmethod
    def get_optimal_arm(self):
        """Get the actual optimal arm

        Returns:
            float: the optimal arm
        """

        pass

    @abstractmethod
    def get_expected_conversion_per_arm(self, arm):
        """Get the expected conversion rate for a specific arm

        Args:
            arm ([type]): the arm for which we want the conversion rate

        Returns:
            np.floating: expected conversion rate
        """
        
        pass
    