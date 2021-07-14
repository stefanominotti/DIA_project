from matplotlib import pyplot as plt
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.preprocessing import StandardScaler

from main.bandits.joint.PriceBidLearner import PriceBidLearner
from main.bandits.joint.ContextPriceBidLearner import ContextPriceBidLearner


class PriceBidGPTSLearner(object):
    """
    Learner for joint pricing and bidding using gaussian process regressor to estimate daily clicks
    """

    def __init__(self, bid_arms, price_arms, negative_probability_threshold, returns_horizon, price_discrimination=False, features=None, customer_classes=None, context_generator_class=None, context_generation_rate=None, confidence=None, incremental_generation=None, approximate=None):
        """Class constructor

        Args:
            bid_arms (list): list of possible bids
            price_arms (list): list of possible prices
            negative_probability_threshold (float): reward negative probability threshold under which an arm can't be pulled
            returns_horizon (integer): days horizon during which a customer can return to purchase after the first time
            price_discrimination (bool, optional): choose wether performing price discrimination. Defaults to False.
            features (list, optional): list of features for splitting contexts. Defaults to None.
            customer_classes (list, optional): list of customer classes. Defaults to None.
            context_generator_class (PriceContextGenerator, optional): type of context generator used. Defaults to None.
            context_generation_rate (integer, optional): rate in days for context generation. Defaults to None.
            confidence (float, optional): Hoeffding confidence. Defaults to None.
            incremental_generation (bool, optional): choose wether generation is incremental or from scratch. Defaults to None.
            approximate (bool, optional): choose wether considering pricing problem as disjoint from bidding problem. Defaults to None.
        """

        self.learner = self.get_learner(bid_arms, price_arms, negative_probability_threshold, returns_horizon, price_discrimination, features, customer_classes, context_generator_class, context_generation_rate, confidence, incremental_generation, approximate)
        
    def get_learner(self, bid_arms, price_arms, negative_probability_threshold, returns_horizon, price_discrimination=False, features=None, customer_classes=None, context_generator_class=None, context_generation_rate=None, confidence=None, incremental_generation=None, approximate=None):
        """If price_discrimination is True return a learner based on ContextPriceBidLearner else returns a learner based on PriceBidLearner

        Args:
            bid_arms (list): list of possible bids
            price_arms (list): list of possible prices
            negative_probability_threshold (float): reward negative probability threshold under which an arm can't be pulled
            returns_horizon (integer): days horizon during which a customer can return to purchase after the first time
            price_discrimination (bool, optional): choose wether performing price discrimination. Defaults to False.
            features (list, optional): list of features for splitting contexts. Defaults to None.
            customer_classes (list, optional): list of customer classes. Defaults to None.
            context_generator_class (PriceContextGenerator, optional): type of context generator used. Defaults to None.
            context_generation_rate (integer, optional): rate in days for context generation. Defaults to None.
            confidence (float, optional): Hoeffding confidence. Defaults to None.
            incremental_generation (bool, optional): choose wether generation is incremental or from scratch. Defaults to None.
            approximate (bool, optional): choose wether considering pricing problem as disjoint from bidding problem. Defaults to None.

        Returns:
            Learner: the learner
        """

        class Learner(ContextPriceBidLearner if price_discrimination else PriceBidLearner):

            def __init__(self, bid_arms, price_arms, negative_probability_threshold, returns_horizon, price_discrimination=False, features=None, customer_classes=None, context_generator_class=None, context_generation_rate=None, confidence=None, incremental_generation=None, approximate=None):
                if price_discrimination:
                    super().__init__(bid_arms, price_arms, negative_probability_threshold, returns_horizon, features, customer_classes, context_generator_class, context_generation_rate, confidence, incremental_generation, approximate)
                else:
                    super().__init__(bid_arms, price_arms, negative_probability_threshold, returns_horizon)

            def update(self, pulled_arm, customers):
                super().update(pulled_arm, customers)

                if len(self.pulled_arms) > 1:
                    self.daily_clicks_means, self.daily_clicks_sigmas = self.gp_regression(self.collected_daily_clicks_per_arm)
                
            def gp_regression(self, samples_per_arm):
                alpha = 0.1
                kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
                gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=10)
                x = []
                y = []
                for idx, arm in enumerate(samples_per_arm):
                    x.extend([self.arms[idx] for _ in range(len(arm))])
                    y.extend(arm)
                x = np.atleast_2d(x).T
                gp.fit(x, y)
                means, sigmas = gp.predict(np.atleast_2d(self.arms).T, return_std=True)
                sigmas = np.maximum(sigmas, 1e-2) 
                samples_per_arm = np.array(samples_per_arm)

                return means, sigmas
        
        return Learner(bid_arms, price_arms, negative_probability_threshold, returns_horizon, price_discrimination, features, customer_classes, context_generator_class, context_generation_rate, confidence, incremental_generation, approximate)
            
    def pull_arm(self):
        """Pull an arm

        Returns:
            tuple: tuple of bid and price if price_discrimination is False else tuple containing a bid, the context structure, prices for each context, prices for each class
        """

        return self.learner.pull_arm()

    def update(self, pulled_arm, customers):
        """Update the estimations given a set of daily customers

        Args:
            pulled_arm (float): the daily pulled bid
            customers (list): the daily customers
        """

        self.learner.update(pulled_arm, customers)
    
    def get_optimal_arm(self):
        """Get the actual optimal arm

        Returns:
            tuple: tuple of optimal price and bid if price_discrimination is False else tuple containing the optimal price for each context in the structure and the optimal bid
        """

        return self.learner.get_optimal_arm()
    