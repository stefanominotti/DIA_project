from main.experiments.Experiment import Experiment
from main.environment.Environment import Environment


class PriceExperiment(Experiment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reward_per_experiment = [[[] for _ in range(self.n_exp)]]

    def run(self):
        optimal_arms = []
        for exp in range(self.n_exp):
            env = Environment(self.scen)
            learner = self.learner_class(self.scen.prices, self.scen.returns_horizon)
            customers_per_day = []
            for day in range(1, self.scen.rounds_horizon):
                price = learner.pull_arm()
                customers, returns = env.round(self.fixed_bid, [price for _ in range(len(self.scen.customer_classes))])
                
                customers_per_day.append(customers)
                if day > self.scen.returns_horizon:
                    delayed_customers = customers_per_day.pop(0)
                    self.reward_per_experiment[0][exp].append(sum(list(map(lambda x: x.conversion * x.conversion_price * (1 + x.returns_count) - x.cost_per_click, delayed_customers))))
                    learner.update(delayed_customers)
            optimal_arms.append(learner.get_optimal_arm())
        return optimal_arms
            