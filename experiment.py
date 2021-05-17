from Environment import Environment
from Scenario import Scenario
from PriceUCBLearner import PriceUCBLearner
from PriceTSLearner import PriceTSLearner
from ObjectiveFunction import ObjectiveFunction
import numpy as np
import matplotlib.pyplot as plt

scen = Scenario('scenario_example')
bid = [np.random.choice(scen.bids)]
objectiveFunction = ObjectiveFunction(scen)

print(f'Bid {bid}')

optimal, p1, b1 = objectiveFunction.get_optimal_price_bid()
print(f'optimal price {p1}')

fig, axes = plt.subplots(2)

for idx, learner_class in enumerate((PriceUCBLearner, PriceTSLearner)):
    n_exp = 1
    reward_per_experiment = [[] for _ in range(n_exp)]

    for exp in range(n_exp):
        print(f'Exp: {exp+1}')
        env = Environment(scen)
        learner = learner_class(scen.prices)

        customers_per_day = []
        for day in range(1, scen.rounds_horizon):
            print(f'Day: {day}')

            price = learner.pull_arm()
            customers, returns = env.round(bid, [price for _ in range(len(scen.customer_classes))])
            customers_per_day.append(customers)
            if day > 30:
                delayed_customers = customers_per_day.pop(0)
                learner.update(delayed_customers)
                reward_per_experiment[exp].append(sum(list(map(lambda x: x.conversion_price * (1 + x.returns_count), list(filter(lambda customer: customer.conversion == 1, delayed_customers))))) / len(delayed_customers))
            print(price)

        print(learner.get_optimal_arm())


    axes[0].plot(np.cumsum(np.mean(np.subtract(optimal, reward_per_experiment), axis=0)), 'C' + str(idx))
    axes[1].plot(np.mean(reward_per_experiment, axis=0), 'C' + str(idx))

plt.show()
