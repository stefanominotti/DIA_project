import numpy as np

from PriceBidLearner import PriceBidLearner
from ContextPriceBidLearner import ContextPriceBidLearner


class PriceBidGTSLearner(object):

    def __init__(self, bid_arms, price_arms, negative_probability_threshold, returns_horizon, price_discrimination=False, features=None, customer_classes=None, context_generator_class=None, context_generation_rate=None, confidence=None):
        self.learner = self.get_learner(bid_arms, price_arms, negative_probability_threshold, returns_horizon, price_discrimination, features, customer_classes, context_generator_class, context_generation_rate, confidence)

    def get_learner(self, bid_arms, price_arms, negative_probability_threshold, returns_horizon, price_discrimination=False, features=None, customer_classes=None, context_generator_class=None, context_generation_rate=None, confidence=None):
        class Learner(ContextPriceBidLearner if price_discrimination else PriceBidLearner):

            def __init__(self, bid_arms, price_arms, negative_probability_threshold, returns_horizon, price_discrimination=False, features=None, customer_classes=None, context_generator_class=None, context_generation_rate=None, confidence=None):
                if price_discrimination:
                    super().__init__(bid_arms, price_arms, negative_probability_threshold, returns_horizon, features, customer_classes, context_generator_class, context_generation_rate, confidence)
                else:
                    super().__init__(bid_arms, price_arms, negative_probability_threshold, returns_horizon)

            def update(self, pulled_arm, customers, returns=[]):
                super().update(pulled_arm, customers, returns=returns)

                arm_idx = self.arms.index(pulled_arm)

                self.daily_clicks_means[arm_idx] = np.mean(self.collected_daily_clicks_per_arm[arm_idx])
                n_daily_clicks_samples = len(self.collected_daily_clicks_per_arm[arm_idx])
                if n_daily_clicks_samples > 0:
                    self.daily_clicks_sigmas[arm_idx] = np.std(self.collected_daily_clicks_per_arm[arm_idx]) / n_daily_clicks_samples
        
        return Learner(bid_arms, price_arms, negative_probability_threshold, returns_horizon, price_discrimination, features, customer_classes, context_generator_class, context_generation_rate, confidence)
            
    def pull_arm(self):
        return self.learner.pull_arm()

    def update(self, pulled_arm, customers, returns=[]):
        self.learner.update(pulled_arm, customers, returns)
    
    
