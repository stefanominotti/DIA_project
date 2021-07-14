from matplotlib import pyplot as plt
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.preprocessing import StandardScaler

from main.bandits.joint.PriceBidLearner import PriceBidLearner
from main.bandits.joint.ContextPriceBidLearner import ContextPriceBidLearner


class PriceBidGPTSLearner(object):
    def __init__(self, bid_arms, price_arms, negative_probability_threshold, returns_horizon, price_discrimination=False, features=None, customer_classes=None, context_generator_class=None, context_generation_rate=None, confidence=None, incremental_generation=None, approximate=None):
        self.learner = self.get_learner(bid_arms, price_arms, negative_probability_threshold, returns_horizon, price_discrimination, features, customer_classes, context_generator_class, context_generation_rate, confidence, incremental_generation, approximate)
        
    def get_learner(self, bid_arms, price_arms, negative_probability_threshold, returns_horizon, price_discrimination=False, features=None, customer_classes=None, context_generator_class=None, context_generation_rate=None, confidence=None, incremental_generation=None, approximate=None):
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
                #plt.figure()
                #print(list(map(lambda x: len(x), samples_per_arm)))
                ##plt.plot(self.arms, list(map(lambda x: len(x), samples_per_arm)), 'r:', label=r'$n(x)$')
                #plt.plot(x, y, 'ro', label=u'Observed clicks')
                #plt.plot(self.arms, means, 'b-', label=u'Predicted clicks')
                #plt.fill_between(self.arms, means-sigmas, means+sigmas, 
                #                 alpha=.5, fc='b', ec='None', label='95% conf interval')
                #plt.xlabel('$x$')
                #plt.ylabel('$y$')
                #plt.legend(loc='lower right')
                #plt.show() 
                return means, sigmas
        
        return Learner(bid_arms, price_arms, negative_probability_threshold, returns_horizon, price_discrimination, features, customer_classes, context_generator_class, context_generation_rate, confidence, incremental_generation, approximate)
            
    def pull_arm(self):
        return self.learner.pull_arm()

    def update(self, pulled_arm, customers):
        self.learner.update(pulled_arm, customers)
    
    def get_optimal_arm(self):
        return self.learner.get_optimal_arm()
    