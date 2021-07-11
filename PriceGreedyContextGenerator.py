import numpy as np
from PriceContextGenerator import PriceContextGenerator


class PriceGreedyContextGenerator(PriceContextGenerator):
    def __init__(self, features, customer_classes, learner_class, arms, returns_horizon, confidence):
        super().__init__(features, customer_classes, learner_class, arms, returns_horizon, confidence)

    def get_best_contexts(self):
        base_learner = self.generate_learner(self.customer_classes)
        self.contexts = [self.customer_classes]
        self.learners = [base_learner]

        if len(self.customers_per_day) != 0:
            base_learner = self.generate_learner(self.customer_classes)
            self.contexts = [self.customer_classes]
            self.learners = [base_learner]
            base_contexts = self.contexts.copy()
            base_learners = self.learners.copy()
            self.contexts = []
            self.learners = []
            for idx in range(len(base_contexts)):
                self.generate_contexts(base_contexts[idx], base_learners[idx])

        return self.contexts, self.learners

    def generate_contexts(self, base_context, base_learner):
        split = self.get_context_best_split(base_context, base_learner)
        if split:
            (left_context, left_learner), (right_context, right_learner) = split
            self.generate_contexts(left_context, left_learner)
            self.generate_contexts(right_context, right_learner)
        else:
            self.contexts.append(base_context)
            self.learners.append(base_learner)

    def get_split_values(self, context, feature_idx):

        left_context = list(filter(lambda customer_class: customer_class.feature_values[feature_idx] == 0, context))
        right_context = list(filter(lambda customer_class: customer_class.feature_values[feature_idx] == 1, context))

        customers = [customer for day in self.customers_per_day for customer in day]

        left_context_customers = self.filter_customers_by_context(left_context, customers)
        right_context_customers = self.filter_customers_by_context(right_context, customers)

        total_customers = len(left_context_customers) + len(right_context_customers)
        p_left = self.get_hoeffding_lower_bound(len(left_context_customers)/total_customers,
                                                total_customers)
        p_right = self.get_hoeffding_lower_bound(len(right_context_customers)/total_customers,
                                                 total_customers)
        
        left_learner = self.generate_learner(left_context)
        right_learner = self.generate_learner(right_context)

        mu_left = self.get_learner_reward_lower_bound(left_learner)
        mu_right = self.get_learner_reward_lower_bound(right_learner)

        return (p_left, mu_left, left_context, left_learner), (p_right, mu_right, right_context, right_learner)

    def get_context_best_split(self, context, learner):
        mu = self.get_learner_reward_lower_bound(learner)

        split = False
        learners = set()
        best_split_value = 0
        for feature_idx in range(len(self.features)):
            if self.is_feature_splittable(context, feature_idx):
                (p_left, mu_left, left_context, left_learner), (p_right, mu_right, right_context, right_learner) = self.get_split_values(context, feature_idx)
                split_value = p_left * mu_left + p_right * mu_right
                if split_value > mu and split_value >= best_split_value:           
                    split = True
                    best_split_value = split_value
                    learners = (left_context, left_learner), (right_context, right_learner)

        return learners if split else False
   
    def is_feature_splittable(self, context, feature_idx):
        return len(set(map(lambda customer_class: customer_class.feature_values[feature_idx], context))) > 1

    def get_learner_reward_lower_bound(self, learner):
        best_arm_rewards = learner.rewards_per_arm[learner.arms.index(learner.get_optimal_arm())]
        mean = np.mean(best_arm_rewards)
        std = np.std(best_arm_rewards)
        cardinality = len(best_arm_rewards)

        return self.get_mean_lower_bound(mean, std, cardinality)

    