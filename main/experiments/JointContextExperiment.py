from main.experiments.Experiment import Experiment
from main.environment.Environment import Environment


class JointContextExperiment(Experiment):
    def __init__(self, negative_probability_threshold, incremental_generation, contextGenerator, generation_rate, confidence, approximate, **kwargs):
        super().__init__(**kwargs)
        self.negative_probability_threshold = negative_probability_threshold
        self.incremental_generation = incremental_generation
        self.contextGenerator = contextGenerator
        self.generation_rate = generation_rate
        self.confidence = confidence
        self.approximate = approximate
        self.reward_per_experiment = [[[] for exp in range(self.n_exp)] for _ in self.scen.customer_classes]

    def run(self):
        optimal_arms = []
        for exp in range(self.n_exp):
            env = Environment(self.scen)
            learner = self.learner_class(self.scen.bids, 
                                         self.scen.prices, 
                                         self.negative_probability_threshold, 
                                         self.scen.returns_horizon, 
                                         price_discrimination=self.price_discrimination, 
                                         features=self.scen.features, 
                                         customer_classes=self.scen.customer_classes, 
                                         context_generator_class=self.contextGenerator, 
                                         context_generation_rate=self.generation_rate, 
                                         confidence=self.confidence,
                                         incremental_generation=self.incremental_generation,
                                         approximate=self.approximate)
            customers_per_day = []
            prices_per_day = []
            bids_per_day = []

            for day in range(self.scen.rounds_horizon + self.scen.returns_horizon):        

                self.printProgressBar(day + 1, self.scen.rounds_horizon + self.scen.returns_horizon, prefix='Exp: '+str(exp+1)+' - Progress:', suffix='Complete', length=50)
                        
                bid, contexts, price_per_context, price_per_class = learner.pull_arm()
                customers, returns = env.round([bid], price_per_class)
                customers_per_day.append(customers)
                prices_per_day.append(price_per_class)
                bids_per_day.append(bid)
                reward = 0
                
                if day > self.scen.returns_horizon:
                    delayed_customers = customers_per_day.pop(0)
                    bid = bids_per_day.pop(0)
                    if day < self.scen.rounds_horizon:
                        learner.update(bid, delayed_customers)
                    for class_idx, customer_class in enumerate(self.scen.customer_classes):
                        class_customers = list(filter(lambda customer: customer.customer_class == customer_class, delayed_customers))
                        reward = 0
                        converted_customers = list(filter(lambda customer: customer.conversion == 1, class_customers))
                        reward += sum(list(map(lambda customer: customer.conversion_price * (1 + customer.returns_count), converted_customers)))
                        reward -= sum(list(map(lambda customer: customer.cost_per_click, class_customers)))
                        self.reward_per_experiment[class_idx][exp].append(reward)
            optimal_arms.append(learner.get_optimal_arm())
        return optimal_arms
            

            