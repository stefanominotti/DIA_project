from ReturnsEstimator import ReturnsEstimator
import numpy as np
from scipy.stats import norm


class BidGTSLearner(object):
    def __init__(self, arms, negative_probability_threshold, returns_horizon):
        self.t = 0
        self.arms = arms
        self.negative_probability_threshold = negative_probability_threshold
        self.pulled_arms = []
        self.collected_rewards = []
        self.rounds_per_arm = np.zeros(len(self.arms))
        self.daily_clicks_per_arm = [[] for _ in range(len(self.arms))]
        self.cost_per_click_per_arm = [[] for _ in range(len(self.arms))]
        self.daily_clicks_means = np.zeros(len(self.arms))
        self.daily_clicks_sigmas = np.ones(len(self.arms)) * 1e3
        self.cost_per_click_means = np.zeros(len(self.arms))
        self.cost_per_click_sigmas = np.ones(len(self.arms)) * 1e3
        self.collected_daily_clicks = []
        self.collected_cost_per_click = []
        self.returns_estimator = ReturnsEstimator(returns_horizon)

    def update_observations(self, arm_idx, daily_clicks, costs_per_click):
        self.t += 1
        self.pulled_arms.append(self.arms[arm_idx])
        self.collected_daily_clicks.append(daily_clicks)
        self.collected_cost_per_click.extend(costs_per_click)
        self.daily_clicks_per_arm[arm_idx].append(daily_clicks)
        self.cost_per_click_per_arm[arm_idx].extend(costs_per_click)
        self.rounds_per_arm[arm_idx] += 1

    def update_returns(self, customers, returns):
        self.returns_estimator.update(list(filter(lambda customer: customer.conversion == 1, customers)), returns)

    def pull_arm(self, conversion_rates, price):
        if len(np.argwhere(self.rounds_per_arm == 0)) != 0:
            return self.arms[np.random.choice(np.argwhere(self.rounds_per_arm == 0).reshape(-1))]

        daily_clicks_samples = np.random.normal(self.daily_clicks_means, self.daily_clicks_sigmas)
        cost_per_click_samples = np.random.normal(self.cost_per_click_means, self.cost_per_click_sigmas)

        reward_samples = daily_clicks_samples * (conversion_rates * price * (1 + self.returns_estimator.weihgted_sum()) - cost_per_click_samples)
        single_customer_reward_means = conversion_rates * price * (1 + self.returns_estimator.weihgted_sum()) - self.cost_per_click_means
        single_customer_reward_sigmas = self.cost_per_click_sigmas
        threshold_probs = np.array([norm(loc=loc, scale=scale).cdf(0) for (loc, scale) in zip(single_customer_reward_means, single_customer_reward_sigmas)])
        valid_arms_idx = np.argwhere(threshold_probs < self.negative_probability_threshold).reshape(-1)
        valid_arms = np.take(self.arms, valid_arms_idx).reshape(-1)
        valid_reward_samples = np.take(reward_samples, valid_arms_idx).reshape(-1)

        if len(valid_reward_samples) == 0:
            raise Exception("All arms exceed negative probability threshold")
        
        return valid_arms[np.random.choice(np.argwhere(valid_reward_samples == valid_reward_samples.max()).reshape(-1))]

    def update(self, pulled_arm, customers, returns):
        self.update_returns(customers, returns)
        arm_idx = self.arms.index(pulled_arm)
        daily_clicks = len(customers)
        costs_per_click = list(map(lambda x: x.cost_per_click, customers))
        self.update_observations(arm_idx, daily_clicks, costs_per_click)
        self.daily_clicks_means[arm_idx] = np.mean(self.daily_clicks_per_arm[arm_idx])
        self.cost_per_click_means[arm_idx] = np.mean(self.cost_per_click_per_arm[arm_idx])
        n_samples = len(self.daily_clicks_per_arm[arm_idx])
        if n_samples > 0:
            self.daily_clicks_sigmas[arm_idx] = np.std(self.daily_clicks_per_arm[arm_idx]) / n_samples
            self.cost_per_click_sigmas[arm_idx] = np.std(self.cost_per_click_per_arm[arm_idx]) / len(self.cost_per_click_per_arm[arm_idx])

    def get_optimal_arm(self, conversion_rate, price, returns_estimate):
        rewards = self.daily_clicks_means * (conversion_rate * price * (1 + returns_estimate) - self.cost_per_click_means)
        return self.arms[np.argmax(rewards)]