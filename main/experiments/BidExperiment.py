from main.experiments.Experiment import Experiment
from main.environment.Environment import Environment


class BidExperiment(Experiment):
    def __init__(self, negative_probability_threshold, **kwargs):
        super().__init__(**kwargs)
        self.negative_probability_threshold = negative_probability_threshold
        self.reward_per_experiment = [[[] for _ in range(self.n_exp)]]

    def run(self):
        optimal_arms = []
        for exp in range(self.n_exp):
            env = Environment(self.scen)
            learner = self.learner_class(self.scen.bids, 
                                        [self.fixed_price[0]], 
                                        self.negative_probability_threshold, 
                                        self.scen.returns_horizon)
            customers_per_day = []
            bids_per_day = []

            for day in range(self.scen.rounds_horizon):
                
                bid, _ = learner.pull_arm()
                customers, returns = env.round([bid], self.fixed_price)
                customers_per_day.append(customers)
                bids_per_day.append(bid)
                reward = 0

                if day > 30:
                    delayed_customers = customers_per_day.pop(0)
                    bid = bids_per_day.pop(0)
                    learner.update(bid, delayed_customers)
                    reward = 0
                    converted_customers = list(filter(lambda customer: customer.conversion == 1, delayed_customers))
                    reward += sum(list(map(lambda customer: customer.conversion_price * (1 + customer.returns_count), converted_customers)))
                    reward -= sum(list(map(lambda customer: customer.cost_per_click, delayed_customers)))
                    
                    self.reward_per_experiment[0][exp].append(reward)
            optimal_arms.append(learner.get_optimal_arm())
        return optimal_arms
            

            