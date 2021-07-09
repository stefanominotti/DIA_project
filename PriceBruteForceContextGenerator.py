import numpy as np


class PriceBruteForceContextGenerator(object):
    def __init__(self, features, customer_classes, learner_class, arms, hoeffding_confidence):
        self.features = features
        self.customer_classes = customer_classes
        for customer_class in self.customer_classes:
            if customer_class.feature_labels != self.features:
                raise Exception("Customer classes must have the same features of the provided list")
        self.t = 0
        self.learner_class = learner_class
        self.arms = arms
        self.hoeffding_confidence = hoeffding_confidence
        self.customers = []
        self.contexts = []
        self.learners = []

    def update(self, customers):
        self.t += 1
        self.customers.extend(customers)

    def get_best_contexts(self):
        if len(self.customers) == 0:
            base_learner = self.generate_learner(self.customer_classes)
            self.contexts.append(self.customer_classes)
            self.learners.append(base_learner)
        else:
            self.generate_contexts()
        return self.contexts, self.learners

    def generate_contexts(self):
        all_contexts = []
        all_learners = []
        arm_rewards = []
        for context_set in self.get_subsets(self.customer_classes):
            if len(context_set) > 0:
                all_contexts.append(context_set)
                learners = [self.generate_learner(context) for context in context_set]
                all_learners.append(learners)
                arm_rewards.append([self.get_lower_bound(learner) for learner in learners])
        arm_rewards = list(map(lambda rewards: np.sum(rewards), arm_rewards))
        print(arm_rewards)
        max_idx = np.argmax(arm_rewards)
        self.contexts = all_contexts[max_idx]
        self.learners = all_learners[max_idx]
        

    def get_lower_bound(self, learner):
        context_customers = np.sum([arm for arm in learner.samples_per_arm])
        total_customers = len(self.customers)
        best_arm = learner.get_optimal_arm()
        best_arm_idx = learner.arms.index(best_arm)
        best_arm_rewards = learner.rewards_per_arm[best_arm_idx]

        u_val = self.get_hoeffding_lower_bound(np.mean(best_arm_rewards),
                                               len(best_arm_rewards)) * learner.get_optimal_arm()
        p_val = self.get_hoeffding_lower_bound(context_customers/total_customers, total_customers)
        return p_val*u_val

    def generate_learner(self, context):
        learner = self.learner_class(self.arms)
        customers = self.filter_customers_by_context(context, self.customers)
        if len(customers) > 0:
            learner.update(customers)
        return learner
    
    def filter_customers_by_context(self, context, customers):
        return list(filter(lambda customer: customer.customer_class in context, customers)) 

    def get_hoeffding_lower_bound(self, mean, cardinality):
        if cardinality == 0:
            return -np.inf
        return mean - np.sqrt(-np.log(self.hoeffding_confidence) / (2 * cardinality))

    def get_subsets(self, s):
        result = [[]]
        for x in s:
            result[0].append([x])
        result.append([s[:2], [s[2]]])
        result.append([s[1:], [s[0]]])
        result.append([[s[0], s[2]], [s[1]]])
        result.append([s])
        return result