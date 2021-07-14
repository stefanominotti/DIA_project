import numpy as np
from abc import ABC, abstractmethod


class PriceContextGenerator(ABC):
    """
    Abstract class for a context generator
    """
    def __init__(self, features, customer_classes, learner_class, arms, returns_horizon, confidence):
        """Class constructor

        Args:
            features (list): list of features for splitting contexts
            customer_classes (list): list of customer classes
            learner_class (PriceLearner): type of price learner used
            arms (list): set of arms to pull
            returns_horizon (integer): days horizon during which a customer can return to purchase after the first time
            confidence (float): Hoeffding confidence
        """

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
        """Update the collecton of customers per day

        Args:
            customers (list): the daily customers
        """
        self.customers_per_day.append(customers)

    @abstractmethod
    def get_best_contexts(self, incremental=False):
        pass

    @abstractmethod
    def generate_contexts(self):
        pass
    
    def generate_learner(self, context):
        """Generate a new learner trainded ofline with collected data

        Args:
            context (list): list of class customer belonging to a context 

        Returns:
            PriceLearner: the learner trainded ofline with collected data
        """

        learner = self.learner_class(self.arms, self.returns_horizon)
        for daily_customers in self.customers_per_day:
            customers = self.filter_customers_by_context(context, daily_customers)
            if len(customers) > 0:
                learner.update(customers)
        return learner
    
    def filter_customers_by_context(self, context, customers):
        """Filter customers by context

        Args:
            context (list): list of class customer belonging to a context
            customers (list): list of customers to filter

        Returns:
            list: filtered list of customers
        """

        return list(filter(lambda customer: customer.customer_class in context, customers)) 

    def get_hoeffding_lower_bound(self, mean, cardinality):
        """Get the Hoeffding lower bound

        Args:
            mean (np.Floating): mean of the distribution whose lower bound is to be calculated
            cardinality (Integer): cardinality of the distribution whose lower bound is to be calculated

        Returns:
            np.Floating: Hoeffding lower bound value
        """
        if cardinality == 0:
            return 0
        return mean - np.sqrt(-np.log(self.confidence) / (2 * cardinality))

    def get_mean_lower_bound(self, mean, std, cardinality):
        """Get lower bound

        Args:
            mean (np.Floating): mean of the distribution whose lower bound is to be calculated
            std (np.Floating): standard deviation of the distribution whose lower bound is to be calculated
            cardinality (Integer): cardinality of the distribution whose lower bound is to be calculated

        Returns:
            np.Floating: lower bound value
        """
        if cardinality == 0:
            return 0
        return mean - 1.96 * std / np.sqrt(cardinality)