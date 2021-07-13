import numpy as np

from scipy.stats import binom

from main.bandits.joint.PriceBidLearner import PriceBidLearner
from main.bandits.pricing.PriceTSLearner import PriceTSLearner
from main.bandits.pricing.ContextPriceLearner import ContextPriceLearner


class ContextPriceBidLearner(PriceBidLearner):
    def __init__(self, bid_arms, price_arms, negative_probability_threshold, returns_horizon, features, customer_classes, context_generator_class, context_generation_rate, confidence, incremental_generation, approximate):
        super().__init__(bid_arms, price_arms, negative_probability_threshold, returns_horizon, approximate)
        if approximate:
            context_learner = ContextPriceLearner(price_arms, PriceTSLearner, features, customer_classes, returns_horizon, context_generator_class, context_generation_rate, confidence, incremental_generation)
            self.price_learner_per_arm = [context_learner for _ in range(len(self.arms))]
        else:
            self.price_learner_per_arm = [ContextPriceLearner(price_arms, PriceTSLearner, features, customer_classes, returns_horizon, context_generator_class, context_generation_rate, confidence, incremental_generation) for _ in range(len(self.arms))]

    def pull_arm(self):
        if len(np.argwhere(self.rounds_per_arm == 0)) != 0:
            idx = np.random.choice(np.argwhere(self.rounds_per_arm == 0).reshape(-1))
            return self.arms[idx], *self.price_learner_per_arm[idx].pull_arm()

        daily_clicks_samples = np.random.normal(self.daily_clicks_means, self.daily_clicks_sigmas)
        cost_per_click_samples = np.random.normal(self.cost_per_click_means, self.cost_per_click_sigmas)

        contexts_per_arm = []
        price_per_context_per_arm = []
        price_per_class_per_arm = []
        for learner in self.price_learner_per_arm:
            contexts, price_per_context, price_per_class = learner.pull_arm()
            contexts_per_arm.append(contexts)
            price_per_context_per_arm.append(price_per_context)
            price_per_class_per_arm.append(price_per_class)

        conversion_rates_per_context_per_arm = np.array([self.price_learner_per_arm[idx].get_expected_conversion_per_arm(price_per_context_per_arm[idx]) for idx in range(len(price_per_context_per_arm))])
        returns_mean_per_context_per_arm = np.array([self.price_learner_per_arm[idx].get_expected_return_per_arm(price_per_context_per_arm[idx]) for idx in range(len(price_per_context_per_arm))])
        context_weights_per_arm = np.array([price_learner.get_contexts_weights() for price_learner in self.price_learner_per_arm])

        customer_value_per_arm = []
        for idx in range(len(self.arms)):
            customer_value_per_arm.append((np.array(context_weights_per_arm[idx]) * np.array(price_per_context_per_arm[idx]) * np.array(conversion_rates_per_context_per_arm[idx]) * (1 + np.array(returns_mean_per_context_per_arm[idx]))).sum())

        reward_samples = daily_clicks_samples * (customer_value_per_arm - cost_per_click_samples)

        converted_user_reward_means = []
        for idx in range(len(self.arms)):
            converted_user_reward_means.append((np.array(context_weights_per_arm[idx]) * np.array(price_per_context_per_arm[idx]) * (1 + np.array(returns_mean_per_context_per_arm[idx]))).sum() - self.cost_per_click_means[idx])

        conversion_rates_per_arm = []
        for idx in range(len(self.arms)):
            conversion_rates_per_arm.append((np.array(context_weights_per_arm[idx]) * np.array(conversion_rates_per_context_per_arm[idx])).sum())

        negative_reward_conversions_threshold = self.daily_clicks_means * self.cost_per_click_means / (converted_user_reward_means + self.cost_per_click_means)
        negative_reward_conversions_threshold = negative_reward_conversions_threshold.astype(int)
        negative_reward_probs = np.array([binom.cdf(negative_reward_conversions_threshold[idx], self.daily_clicks_means[idx], conversion_rates_per_arm[idx]) for idx in range(len(self.arms))])

        valid_arms_idx = np.argwhere(negative_reward_probs < self.negative_probability_threshold).reshape(-1)
        valid_arms = np.take(self.arms, valid_arms_idx).reshape(-1)
        valid_contexts_per_arm = np.take(contexts_per_arm, valid_arms_idx, axis=0)
        valid_price_per_context_per_arm = np.take(price_per_context_per_arm, valid_arms_idx, axis=0)
        valid_price_per_class_per_arm = np.take(price_per_class_per_arm, valid_arms_idx, axis=0)

        valid_reward_samples = np.take(reward_samples, valid_arms_idx).reshape(-1)

        if len(valid_reward_samples) == 0:
            raise Exception("All arms exceed negative probability threshold")

        idx = np.random.choice(np.argwhere(valid_reward_samples == valid_reward_samples.max()).reshape(-1))
        
        return valid_arms[idx], valid_contexts_per_arm[idx], valid_price_per_context_per_arm[idx], valid_price_per_class_per_arm[idx]

    def get_optimal_arm(self):
        price_per_context_per_arm = [learner.get_optimal_arm() for learner in self.price_learner_per_arm]
        conversion_rates_per_context_per_arm = np.array([self.price_learner_per_arm[idx].get_expected_conversion_per_arm(price_per_context_per_arm[idx]) for idx in range(len(price_per_context_per_arm))])
        returns_mean_per_context_per_arm = np.array([self.price_learner_per_arm[idx].get_expected_return_per_arm(price_per_context_per_arm[idx]) for idx in range(len(price_per_context_per_arm))])
        context_weights_per_arm = np.array([price_learner.get_contexts_weights() for price_learner in self.price_learner_per_arm])
        daily_clicks_means = np.array(list(map(lambda x: [x], self.daily_clicks_means)))
        cost_per_click_means = np.array(list(map(lambda x: [x], self.cost_per_click_means)))
        rewards = daily_clicks_means * (context_weights_per_arm * conversion_rates_per_context_per_arm * price_per_context_per_arm * (1 + returns_mean_per_context_per_arm) - cost_per_click_means)
        return (self.price_learner_per_arm[np.argmax(rewards)].get_optimal_arm_per_class(), self.arms[np.argmax(np.argmax(rewards))])
