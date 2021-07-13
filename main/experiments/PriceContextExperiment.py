from main.bandits.pricing.ContextPriceLearner import ContextPriceLearner
from main.experiments.Experiment import Experiment
from main.environment.Environment import Environment


class PriceContextExperiment(Experiment):
    def __init__(self, incremental_generation, contextGenerator, generation_rate, confidence, **kwargs):
        super().__init__(**kwargs)
        self.contextGenerator = contextGenerator
        self.generation_rate = generation_rate
        self.confidence = confidence
        self.incremental_generation = incremental_generation
        self.reward_per_experiment = [[[] for exp in range(self.n_exp)] for _ in range(len(self.scen.customer_classes))]

    def run(self):
        optimal_arms = []
        for exp in range(self.n_exp):
            env = Environment(self.scen)
            learner = ContextPriceLearner(self.scen.prices, 
                                          self.learner_class, 
                                          self.scen.features, 
                                          self.scen.customer_classes, 
                                          self.scen.returns_horizon, 
                                          self.contextGenerator, 
                                          self.generation_rate,
                                          self.confidence,
                                          self.incremental_generation)
            customers_per_day = []

            for day in range(self.scen.rounds_horizon):
                contexts, prices_per_context, prices_per_class = learner.pull_arm()
                customers, returns = env.round(self.fixed_bid, prices_per_class)
                customers_per_day.append(customers)

                if day > 30:
                    delayed_customers = customers_per_day.pop(0)
                    learner.update(delayed_customers)
                    for class_idx, customer_class in enumerate(self.scen.customer_classes):
                        class_customers = list(filter(lambda customer: customer.customer_class == customer_class, delayed_customers))
                        self.reward_per_experiment[class_idx][exp].append(
                            sum(list(map(
                                lambda x: x.conversion * x.conversion_price * (1 + x.returns_count) - x.cost_per_click, 
                                class_customers))))
            contexts_result = [('' if customer_class.feature_values[0] else 'not ') + customer_class.feature_labels[0] + \
                                      (' & ' if customer_class.feature_values[1] else ' & not ') + customer_class.feature_labels[1] \
                                          for context in contexts for customer_class in context]
            optimal_arms.append((contexts_result, learner.get_optimal_arm_per_class()))
        return optimal_arms
            