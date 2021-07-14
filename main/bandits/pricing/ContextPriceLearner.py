import numpy as np

class ContextPriceLearner(object):
    """
    A price learner performing price discrimination 
    """

    def __init__(self, arms, learner_class, features, customer_classes, returns_horizon, context_generator_class, context_generation_rate, confidence, incremental_generation):
        """Class constructor

        Args:
            arms (list): set of arms to pull
            learner_class (PriceLearner): type of price learner used
            features (list): list of features for splitting contexts
            customer_classes (list): list of customer classes
            returns_horizon (integer): days horizon during which a customer can return to purchase after the first time
            context_generator_class (PriceContextGenerator): type of context generator used
            context_generation_rate (integer): rate in days for context generation
            confidence (float): Hoeffding confidence
            incremental_generation (bool): choose wether generation is incremental or from scratch
        """

        self.day = 0
        self.arms = arms
        self.context_generation_rate = context_generation_rate
        self.incremental_generation = incremental_generation
        self.customer_classes = customer_classes
        self.context_generator = context_generator_class(features, customer_classes, learner_class, arms, returns_horizon, confidence)
        self.contexts, self.learners = self.context_generator.get_best_contexts()

    def pull_arm(self):
        """Pull an arm

        Returns:
            tuple: tuple containing context structure, pulled prices for each context, pulled prices for each class
        """

        prices_per_context = [learner.pull_arm() for learner in self.learners]
        prices_per_class = [0 for _ in range(len(self.customer_classes))]
        for context_idx, context in enumerate(self.contexts):
            for customer_class in context:
                prices_per_class[self.customer_classes.index(customer_class)] = prices_per_context[context_idx]
        return self.contexts.copy(), prices_per_context, prices_per_class

    def update(self, customers):
        """Update the estimations given a set of daily customers and generate new contexts

        Args:
            customers (list): the daily customers
        """

        self.day += 1
        self.context_generator.update(customers)
        for context_idx, context in enumerate(self.contexts):
            self.learners[context_idx].update(list(filter(lambda customer: customer.customer_class in context, customers)))
        
        if self.day % self.context_generation_rate == 0:
            self.contexts, self.learners = self.context_generator.get_best_contexts(self.incremental_generation)

    def get_optimal_arm(self):
        """Get the optimal arm for each context in the structure

        Returns:
            np.ndarray: the optimal arm for each context in the structure
        """

        return np.array([learner.get_optimal_arm() for learner in self.learners])

    def get_optimal_arm_per_class(self):
        """Get the optimal arm for each customer class

        Returns:
            np.ndarray: the optimal arm for each customer class
        """

        prices_per_context = [learner.get_optimal_arm() for learner in self.learners]
        prices_per_class = [0 for _ in range(len(self.customer_classes))]
        for context_idx, context in enumerate(self.contexts):
            for customer_class in context:
                prices_per_class[self.customer_classes.index(customer_class)] = prices_per_context[context_idx]
        return prices_per_class

    def get_expected_conversion_per_arm(self, arm_per_context):
        """Get the expected conversion rate for a specific arm for each context

        Args:
            arm_per_context (list): the arm for each context for which we want the conversion rate

        Returns:
            np.ndarray: the conversion rates
        """

        return np.array([learner.get_expected_conversion_per_arm(arm_per_context[idx]) for idx, learner in enumerate(self.learners)])

    def get_expected_return_per_arm(self, arm_per_context):
        """Get the expected number of returns for a specific arm for each context

        Args:
            arm_per_context (list): the arm for each context for which we want the number of returns

        Returns:
            np.ndarray: the number of returns
        """

        return np.array([learner.returns_estimators[self.arms.index(arm_per_context[idx])].mean() for idx, learner in enumerate(self.learners)])

    def get_contexts_weights(self):
        """Get the weights for each context in the structure

        Returns:
            np.ndarray: the weights for each context in the structure
        """

        total_customers = sum(list(map(lambda learner: learner.total_customers, self.learners)))
        if total_customers == 0:
            return 0
        return np.array([learner.total_customers / total_customers for learner in self.learners])