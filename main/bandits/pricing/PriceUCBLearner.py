import numpy as np

from main.bandits.pricing.PriceLearner import PriceLearner


class PriceUCBLearner(PriceLearner):
    """
    A price learner based on Upper Confidence Bound
    """

    def __init__(self, arms, returns_horizon):
        """Class constructor

        Args:
            arms (list): set of arms to pull
            returns_horizon (integer): days horizon during which a customer can return to purchase after the first time
        """

        super().__init__(arms, returns_horizon)
        self.expected_conversion_per_arm = np.zeros(len(self.arms))
        self.conversion_upper_bound_per_arm = np.ones(len(self.arms))

    def pull_arm(self):
        """Pull an arm

        Returns:
            float: the pulled arm
        """

        returns_sample_per_arm = np.array([estimator.total_customers for estimator in self.returns_estimators])

        if len(np.argwhere(self.samples_per_arm == 0)) != 0:
            return self.arms[np.random.choice(np.argwhere(self.samples_per_arm == 0).reshape(-1))]

        returns_means_per_arm = np.array([estimator.mean() for estimator in self.returns_estimators])        
        returns_upper_bound_per_arm = returns_means_per_arm + self.returns_horizon * np.sqrt(2 * np.log(returns_sample_per_arm) / returns_sample_per_arm.sum())
        reward_upper_bound_per_arm = self.conversion_upper_bound_per_arm * np.array(self.arms) * (1 + returns_upper_bound_per_arm)
        return self.arms[np.random.choice(np.argwhere(reward_upper_bound_per_arm == reward_upper_bound_per_arm.max()).reshape(-1))]

    def update(self, customers):
        """Update the estimations given a set of daily customers

        Args:
            customers (list): the daily customers
        """

        for customer in customers:
            self.update_observations(customer)

        self.expected_conversion_per_arm = [np.mean(collected_conversions) for collected_conversions in self.collected_conversions_per_arm]
        self.conversion_upper_bound_per_arm = self.expected_conversion_per_arm + np.sqrt(2 * np.log(self.total_customers) / self.samples_per_arm)

    def get_optimal_arm(self):
        """Get the actual optimal arm

        Returns:
            float: the optimal arm
        """

        returns_means = np.array([estimator.mean() for estimator in self.returns_estimators])
        return self.arms[np.argmax(self.expected_conversion_per_arm * np.array(self.arms) * (1 + returns_means))]

    def get_expected_conversion_per_arm(self, arm):
        """Get the expected conversion rate for a specific arm

        Args:
            arm ([type]): the arm for which we want the conversion rate

        Returns:
            np.floating: expected conversion rate
        """

        return self.expected_conversion_per_arm[self.arms.index(arm)]