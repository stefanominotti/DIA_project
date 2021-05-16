import numpy as np


class PriceGreedyContextGenerator(object):
    def __init__(self, features, customer_classes, learner_class, arms, hoeffding_confidence):
        self.features = features
        self.customer_classes = customer_classes
        for customer_class in self.customer_classes:
            if customer_class.feature_labels != self.features:
                raise Exception("Customer classes must have the same features of the provided list")

        self.learner_class = learner_class
        self.arms = arms
        self.hoeffding_confidence = hoeffding_confidence
        self.customers = []
        self.contexts = []
        self.learners = []

    def update(self, customers):
        self.customers.extend(customers)

    def get_best_contexts(self):
        if len(self.customers) == 0:
            base_learner = self.generate_learner(self.customer_classes)
            self.contexts.append(self.customer_classes)
            self.learners.append(base_learner)
        else:
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

        left_context_customers = self.filter_customers_by_context(left_context)
        right_context_customers = self.filter_customers_by_context(right_context)

        total_customers = len(left_context_customers) + len(right_context_customers)
        p_left = self.get_hoeffding_lower_bound(len(left_context_customers)/total_customers,
                                                total_customers)
        p_right = self.get_hoeffding_lower_bound(len(right_context_customers)/total_customers,
                                                 total_customers)
        
        left_learner = self.generate_learner(left_context)
        right_learner = self.generate_learner(right_context)

        left_best_arm_rewards = left_learner.rewards_per_arm[left_learner.arms.index(left_learner.get_optimal_arm())]
        right_best_arm_rewards = right_learner.rewards_per_arm[right_learner.arms.index(right_learner.get_optimal_arm())]

        mu_left = self.get_hoeffding_lower_bound(np.mean(left_best_arm_rewards),
                                                 len(left_best_arm_rewards))
        mu_right = self.get_hoeffding_lower_bound(np.mean(right_best_arm_rewards),
                                                 len(right_best_arm_rewards))

        return (p_left, mu_left, left_context, left_learner), (p_right, mu_right, right_context, right_learner)

    def get_context_best_split(self, context, learner):
        best_arm_rewards = learner.rewards_per_arm[learner.arms.index(learner.get_optimal_arm())]

        mu = self.get_hoeffding_lower_bound(np.mean(best_arm_rewards),
                                            len(best_arm_rewards))

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

    def generate_learner(self, context):
        learner = self.learner_class(self.arms)
        customers = self.filter_customers_by_context(context)
        if len(customers) > 0:
            learner.update(customers)
        return learner
    
    def filter_customers_by_context(self, context):
        return list(filter(lambda customer: customer.customer_class in context, self.customers)) 

    def get_hoeffding_lower_bound(self, mean, cardinality):
        if cardinality == 0:
            return -np.inf
        return mean - np.sqrt(-np.log(self.hoeffding_confidence) / (2 * cardinality))