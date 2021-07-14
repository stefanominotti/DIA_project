import numpy as np

from main.bandits.joint.PriceBidLearner import PriceBidLearner
from main.bandits.joint.ContextPriceBidLearner import ContextPriceBidLearner


class PriceBidGTSLearner(object):
    """
    Learner for joint pricing and bidding using gaussian estimations for daily clicks and cost per click
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
                    super().__init__(bid_arms, price_arms, negative_probability_threshold, returns_horizon, approximate)

            def update(self, pulled_arm, customers):
                super().update(pulled_arm, customers)

                arm_idx = self.arms.index(pulled_arm)

                self.daily_clicks_means[arm_idx] = np.mean(self.collected_daily_clicks_per_arm[arm_idx])
                n_daily_clicks_samples = len(self.collected_daily_clicks_per_arm[arm_idx])
                if n_daily_clicks_samples > 0:
                    self.daily_clicks_sigmas[arm_idx] = np.std(self.collected_daily_clicks_per_arm[arm_idx]) / n_daily_clicks_samples
        
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
    
    
