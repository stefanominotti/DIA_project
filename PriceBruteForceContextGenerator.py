import numpy as np
from ContextGenerator import ContextGenerator


class PriceBruteForceContextGenerator(ContextGenerator):
    def __init__(self, features, customer_classes, learner_class, arms, returns_horizon, confidence):
        super().__init__(features, customer_classes, learner_class, arms, returns_horizon, confidence)

    def get_best_contexts(self):
        if len(self.customers_per_day) == 0:
            base_learner = self.generate_learner(self.customer_classes)
            self.contexts.append(self.customer_classes)
            self.learners.append(base_learner)
        else:
            self.generate_contexts()
        return self.contexts, self.learners

    def generate_contexts(self):
        possible_contexts_split = []
        learners_per_split = []
        rewards_per_split = []
        for contexts_split in self.get_subsets(self.customer_classes):
            if len(contexts_split) > 0:
                possible_contexts_split.append(contexts_split)
                learners = [self.generate_learner(context) for context in contexts_split]
                learners_per_split.append(learners)
                rewards_per_split.append(np.sum([self.get_context_reward_lower_bound(learner) for learner in learners]))
        best_split_idx = np.argmax(rewards_per_split)
        self.contexts = possible_contexts_split[best_split_idx]
        self.learners = learners_per_split[best_split_idx]

    def get_context_reward_lower_bound(self, learner):
        context_customers = np.sum([arm for arm in learner.samples_per_arm])
        total_customers = np.sum([len(daily_customers) for daily_customers in self.customers_per_day])
        
        best_arm = learner.get_optimal_arm()
        best_arm_idx = learner.arms.index(best_arm)
        best_arm_rewards = learner.rewards_per_arm[best_arm_idx]
        
        mean = np.mean(best_arm_rewards)
        std = np.std(best_arm_rewards)
        cardinality = len(best_arm_rewards)

        mu_val = self.get_mean_lower_bound(mean, std, cardinality)
        p_val = self.get_hoeffding_lower_bound(context_customers / total_customers, total_customers)
        return p_val*mu_val

    def get_subsets(self, s):
        result = [[]]
        for x in s:
            result[0].append([x])
        result.append([s[:2], [s[2]]])
        result.append([s[1:], [s[0]]])
        result.append([[s[0], s[2]], [s[1]]])
        result.append([s])
        return result