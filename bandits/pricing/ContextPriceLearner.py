class ContextPriceLearner(object):
    def __init__(self, arms, learner_class, features, customer_classes, returns_horizon, context_generator_class, context_generation_rate, confidence, incremental_generation):
        self.day = 0
        self.arms = arms
        self.context_generation_rate = context_generation_rate
        self.incremental_generation = incremental_generation
        self.customer_classes = customer_classes
        self.context_generator = context_generator_class(features, customer_classes, learner_class, arms, returns_horizon, confidence)
        self.contexts, self.learners = self.context_generator.get_best_contexts()

    def pull_arm(self):
        prices_per_context = [learner.pull_arm() for learner in self.learners]
        prices_per_class = [0 for _ in range(len(self.customer_classes))]
        for context_idx, context in enumerate(self.contexts):
            for customer_class in context:
                prices_per_class[self.customer_classes.index(customer_class)] = prices_per_context[context_idx]
        return self.contexts.copy(), prices_per_context, prices_per_class

    def update(self, customers, returns=[]):
        self.day += 1
        self.context_generator.update(customers)
        for context_idx, context in enumerate(self.contexts):
            self.learners[context_idx].update(list(filter(lambda customer: customer.customer_class in context, customers)),
                                              returns=list(filter(lambda customer: customer.customer_class in context, returns)))
        
        if self.day % self.context_generation_rate == 0:
            self.contexts, self.learners = self.context_generator.get_best_contexts(self.incremental_generation)

    def get_optimal_arm(self):
        return [learner.get_optimal_arm() for learner in self.learners]

    def get_expected_conversion_per_arm(self, arm_per_context):
        return [learner.get_expected_conversion_per_arm(arm_per_context[idx]) for idx, learner in enumerate(self.learners)]

    def get_expected_return_per_arm(self, arm_per_context):
        return [learner.returns_estimators[self.arms.index(arm_per_context[idx])].mean() for idx, learner in enumerate(self.learners)]

    def get_contexts_weights(self):
        total_customers = sum(list(map(lambda learner: learner.total_customers, self.learners)))
        return [learner.total_customers / total_customers for learner in self.learners]