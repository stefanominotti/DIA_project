from ContextPriceLearner import ContextPriceLearner
from Environment import Environment
from Scenario import Scenario
from PriceTSLearner import PriceTSLearner
from PriceUCBLearner import PriceUCBLearner
from ObjectiveFunction import ObjectiveFunction
from PriceGreedyContextGenerator import PriceGreedyContextGenerator
from PriceBruteForceContextGenerator import PriceBruteForceContextGenerator
import numpy as np
import matplotlib.pyplot as plt

scen = Scenario('scenario_example')
bid = [4]

print(f'Bid {bid}')

objectiveFunction = ObjectiveFunction(scen, bids=bid)

print(f'Bid {bid}')

optimal, p1, b1 = objectiveFunction.get_optimal(price_discrimination=True)
print(f'optimal price {p1}')
print(f'optimal {optimal}')

fig, axes = plt.subplots(2)

for idx, learner_class in enumerate([PriceTSLearner]):
    print(learner_class.__name__)
    n_exp = 1
    reward_per_class_per_experiment = [[[] for _ in range(n_exp)] for _ in range(len(scen.customer_classes))]

    for exp in range(n_exp):
        print(f'Exp: {exp+1}')
        env = Environment(scen)
        learner = ContextPriceLearner(scen.prices, learner_class, scen.features, scen.customer_classes, scen.returns_horizon, PriceBruteForceContextGenerator, 20, 0.05)
        customers_per_day = []

        for day in range(scen.rounds_horizon):
            print(f'Day: {day}')

            contexts, prices_per_context, prices_per_class = learner.pull_arm()

            print(len(contexts))
            print(prices_per_class)

            customers, returns = env.round(bid, prices_per_class)
            customers_per_day.append(customers)

            if day > 30:
                delayed_customers = customers_per_day.pop(0)
                learner.update(delayed_customers)
                for class_idx, customer_class in enumerate(scen.customer_classes):
                    class_customers = list(filter(lambda customer: customer.customer_class == customer_class, delayed_customers))
                    reward_per_class_per_experiment[class_idx][exp].append(sum(list(map(lambda x: x.conversion * x.conversion_price * (1 + x.returns_count) - x.cost_per_click, class_customers))))

        #print(returns_estimator.get_probabilities())
        print(learner.get_optimal_arm())

    for class_idx, customer_class in enumerate(scen.customer_classes):
        axes[0].plot(np.cumsum(np.mean(np.subtract(optimal[class_idx], reward_per_class_per_experiment[class_idx]), axis=0)), 'C' + str(class_idx))
        axes[1].plot(np.mean(reward_per_class_per_experiment[class_idx], axis=0), 'C' + str(class_idx))


plt.show()
