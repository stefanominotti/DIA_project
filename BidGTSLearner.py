import numpy as np
from scipy.stats import binom
from PriceTSLearner import PriceTSLearner


class BidGTSLearner(object):
    def __init__(self, bid_arms, price_arms, negative_probability_threshold, returns_horizon):
        self.arms = bid_arms
        self.negative_probability_threshold = negative_probability_threshold
        self.pulled_arms = []
        self.rounds_per_arm = np.zeros(len(self.arms))
        self.collected_daily_clicks_per_arm = [[] for _ in range(len(self.arms))]
        self.collected_cost_per_click_per_arm = [[] for _ in range(len(self.arms))]
        self.daily_clicks_means = np.zeros(len(self.arms))
        self.daily_clicks_sigmas = np.ones(len(self.arms)) * 1e3
        self.cost_per_click_means = np.zeros(len(self.arms))
        self.cost_per_click_sigmas = np.ones(len(self.arms)) * 1e3
        self.price_learner_per_arm = [PriceTSLearner(price_arms, returns_horizon) for _ in range(len(self.arms))]

    def pull_arm(self):
        if len(np.argwhere(self.rounds_per_arm == 0)) != 0:
            idx = np.random.choice(np.argwhere(self.rounds_per_arm == 0).reshape(-1))
            return self.arms[idx], self.price_learner_per_arm[idx].pull_arm()

        daily_clicks_samples = np.random.normal(self.daily_clicks_means, self.daily_clicks_sigmas)
        cost_per_click_samples = np.random.normal(self.cost_per_click_means, self.cost_per_click_sigmas)

        price_per_arm = np.array([learner.pull_arm() for learner in self.price_learner_per_arm])
        price_idx_per_arm = [self.price_learner_per_arm[idx].arms.index(price_per_arm[idx]) for idx in range(len(price_per_arm))]
        conversion_rates_per_arm = np.array([self.price_learner_per_arm[idx].get_expected_conversion_per_arm()[price_idx_per_arm[idx]] for idx in range(len(price_idx_per_arm))])
        returns_mean_per_arm = np.array([self.price_learner_per_arm[idx].returns_estimators[price_idx_per_arm[idx]].mean() for idx in range(len(price_idx_per_arm))])

        reward_samples = daily_clicks_samples * (conversion_rates_per_arm * price_per_arm * (1 + returns_mean_per_arm) - cost_per_click_samples)
        converted_user_reward_means = price_per_arm * (1 + returns_mean_per_arm) - self.cost_per_click_means

        negative_reward_conversions_threshold = self.daily_clicks_means * self.cost_per_click_means / (converted_user_reward_means + self.cost_per_click_means)
        negative_reward_conversions_threshold = negative_reward_conversions_threshold.astype(int)
        negative_reward_probs = np.array([binom.cdf(negative_reward_conversions_threshold[idx], self.daily_clicks_means[idx], conversion_rates_per_arm[idx]) for idx in range(len(self.arms))])

        valid_arms_idx = np.argwhere(negative_reward_probs < self.negative_probability_threshold).reshape(-1)
        valid_arms = np.take(self.arms, valid_arms_idx).reshape(-1)
        valid_prices = np.take(price_per_arm, valid_arms_idx).reshape(-1)

        valid_reward_samples = np.take(reward_samples, valid_arms_idx).reshape(-1)

        if len(valid_reward_samples) == 0:
            raise Exception("All arms exceed negative probability threshold")

        idx = np.random.choice(np.argwhere(valid_reward_samples == valid_reward_samples.max()).reshape(-1))
        
        return valid_arms[idx], valid_prices[idx]

    def update(self, pulled_arm, customers, returns=[]):
        arm_idx = self.arms.index(pulled_arm)
        self.rounds_per_arm[arm_idx] += 1

        daily_clicks = len(customers)
        costs_per_click = list(map(lambda customer: customer.cost_per_click, customers))

        self.collected_daily_clicks_per_arm[arm_idx].append(daily_clicks)
        self.collected_cost_per_click_per_arm[arm_idx].extend(costs_per_click)

        self.daily_clicks_means[arm_idx] = np.mean(self.collected_daily_clicks_per_arm[arm_idx])
        self.cost_per_click_means[arm_idx] = np.mean(self.collected_cost_per_click_per_arm[arm_idx])
        n_samples = len(self.collected_daily_clicks_per_arm[arm_idx])
        if n_samples > 0:
            self.daily_clicks_sigmas[arm_idx] = np.std(self.collected_daily_clicks_per_arm[arm_idx]) / n_samples
            self.cost_per_click_sigmas[arm_idx] = np.std(self.collected_cost_per_click_per_arm[arm_idx]) / len(self.collected_cost_per_click_per_arm[arm_idx])

        self.price_learner_per_arm[arm_idx].update(customers)

    def get_optimal_arm(self):
        price_per_arm = np.array([learner.get_optimal_arm() for learner in self.price_learner_per_arm])
        price_idx_per_arm = [self.price_learner_per_arm[idx].arms.index(price_per_arm[idx]) for idx in range(len(price_per_arm))]
        conversion_rates_per_arm = np.array([self.price_learner_per_arm[idx].get_expected_conversion_per_arm()[price_idx_per_arm[idx]] for idx in range(len(price_idx_per_arm))])
        returns_mean_per_arm = np.array([self.price_learner_per_arm[idx].returns_estimators[price_idx_per_arm[idx]].mean() for idx in range(len(price_idx_per_arm))])
        rewards = self.daily_clicks_means * (conversion_rates_per_arm * price_per_arm * (1 + returns_mean_per_arm) - self.cost_per_click_means)
        return self.arms[np.argmax(rewards)]