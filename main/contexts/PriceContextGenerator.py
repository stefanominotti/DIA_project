import numpy as np
from abc import ABC, abstractmethod


class PriceContextGenerator(ABC):
    def __init__(self, features, customer_classes, learner_class, arms, returns_horizon, confidence):
        self.features = features
        self.customer_classes = customer_classes
        for customer_class in self.customer_classes:
            if customer_class.feature_labels != self.features:
                raise Exception("Customer classes must have the same features of the provided list")
        self.learner_class = learner_class
        self.arms = arms
        self.returns_horizon = returns_horizon
        self.confidence = confidence
        self.customers_per_day = []
        self.contexts = []
        self.learners = []

    def update(self, customers):
        self.customers_per_day.append(customers)

    @abstractmethod
    def get_best_contexts(self, incremental=False):
        pass

    @abstractmethod
    def generate_contexts(self):
        pass
    
    def generate_learner(self, context):
        learner = self.learner_class(self.arms, self.returns_horizon)
        for daily_customers in self.customers_per_day:
            customers = self.filter_customers_by_context(context, daily_customers)
            if len(customers) > 0:
                learner.update(customers, [])
        return learner
    
    def filter_customers_by_context(self, context, customers):
        return list(filter(lambda customer: customer.customer_class in context, customers)) 

    def get_hoeffding_lower_bound(self, mean, cardinality):
        if cardinality == 0:
            return 0
        return mean - np.sqrt(-np.log(self.confidence) / (2 * cardinality))

    def get_mean_lower_bound(self, mean, std, cardinality):
        if cardinality == 0:
            return 0
        return mean - 1.96 * std / np.sqrt(cardinality)