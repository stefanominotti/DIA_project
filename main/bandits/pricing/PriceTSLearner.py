import numpy as np

from main.bandits.pricing.PriceLearner import PriceLearner


class PriceTSLearner(PriceLearner):
    """
    A price learner based on Thompson Sampling
    """

    def __init__(self, arms, returns_horizon):
        """Class constructor

        Args:
            arms (list): set of arms to pull
            returns_horizon (integer): days horizon during which a customer can return to purchase after the first time
        """

        super().__init__(arms, returns_horizon)
        self.beta_distribution_per_arm = np.ones((len(self.arms), 2))

    def pull_arm(self):
        """Pull an arm

        Returns:
            float: the pulled arm
        """

        returns_samples = np.array([estimator.sample() for estimator in self.returns_estimators])
        reward_samples = np.random.beta(self.beta_distribution_per_arm[:, 0], self.beta_distribution_per_arm[:, 1]) * np.array(self.arms) * (1 + returns_samples)
        return self.arms[np.random.choice(np.argwhere(reward_samples == reward_samples.max()).reshape(-1))]
 
    def update(self, customers):
        """Update the estimations given a set of daily customers

        Args:
            customers (list): the daily customers
        """

        for customer in customers:
            self.update_observations(customer)
            arm_idx = self.arms.index(customer.conversion_price)
            self.beta_distribution_per_arm[arm_idx, 0] += customer.conversion
            self.beta_distribution_per_arm[arm_idx, 1] += (1.0 - customer.conversion)

    def get_optimal_arm(self):
        """Get the actual optimal arm

        Returns:
            float: the optimal arm
        """

        returns_means = np.array([estimator.mean() for estimator in self.returns_estimators])
        return self.arms[np.argmax(self.get_expected_conversion_per_arm() * np.array(self.arms) * (1 + returns_means))]

    def get_expected_conversion_per_arm(self, arm=None):
        """Get the expected conversion rate for all the arms or for a specific arm

        Args:
            arm (float, optional): the arm for which we want the conversion rate. Defaults to None.

        Returns:
            np.floating/np.ndarray: expected conversion rate(s)
        """
        
        alpha = self.beta_distribution_per_arm[:, 0]
        beta = self.beta_distribution_per_arm[:, 1]
        if arm:
            return (alpha / (alpha + beta))[self.arms.index(arm)]
        return (alpha / (alpha + beta))