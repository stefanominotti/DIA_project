import numpy as np
from PriceBidLearner import PriceBidLearner


class BidGTSLearner(PriceBidLearner):

    def update(self, pulled_arm, customers, returns=[]):
        super.update(pulled_arm, customers, returns=returns)

        arm_idx = self.arms.index(pulled_arm)

        self.daily_clicks_means[arm_idx] = np.mean(self.collected_daily_clicks_per_arm[arm_idx])
        n_daily_clicks_samples = len(self.collected_daily_clicks_per_arm[arm_idx])
        if n_daily_clicks_samples > 0:
            self.daily_clicks_sigmas[arm_idx] = np.std(self.collected_daily_clicks_per_arm[arm_idx]) / n_daily_clicks_samples
