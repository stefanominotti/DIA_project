from Environment import Environment
from Scenario import Scenario
from PriceUCBLearner import PriceUCBLearner
from PriceTSLearner import PriceTSLearner
from ReturnsEstimator import ReturnsEstimator
from PriceGreedyContextGenerator import PriceGreedyContextGenerator
from ObjectiveFunction import ObjectiveFunction
import numpy as np
import matplotlib.pyplot as plt

scen = Scenario('scenario_example')
bid = [np.random.choice(scen.bids)]

print(f'Bid {bid}')

objectiveFunction = ObjectiveFunction(scen, bids=bid)

print(f'Bid {bid}')

optimal, p1, b1 = objectiveFunction.get_optimals_price_bid_per_class()
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
        context_generator = PriceGreedyContextGenerator(scen.features, scen.customer_classes, learner_class, scen.prices, scen.returns_horizon, 0.01)

        contexts, learners = context_generator.get_best_contexts()
        customers_per_day = []

        for day in range(scen.rounds_horizon):
            print(f'Day: {day}')

            if day > 30 and day % 30 == 0:
                print("Gen")
                contexts, learners = context_generator.get_best_contexts()
                print(list(map(lambda x: list(map(lambda y: y.feature_values, x)), contexts)), len(learners))

            price_per_context = [learner.pull_arm() for learner in learners]
            prices = [0 for _ in range(len(scen.customer_classes))]
            for context_idx, context in enumerate(contexts):
                for customer_class in context:
                    prices[scen.customer_classes.index(customer_class)] = price_per_context[context_idx]

            print(prices)

            customers, returns = env.round(bid, prices)
            customers_per_day.append(customers)

            context_generator.update(customers)

            for context_idx, context in enumerate(contexts):
                learners[context_idx].update(list(filter(lambda customer: customer.customer_class in context, customers)),
                                             list(filter(lambda customer: customer.customer_class in context, returns)))

            if day > 30:
                delayed_customers = customers_per_day.pop(0)

                for class_idx, customer_class in enumerate(scen.customer_classes):
                    class_customers = list(filter(lambda customer: customer.customer_class == customer_class, delayed_customers))
                    reward_per_class_per_experiment[class_idx][exp].append(sum(list(map(lambda x: x.conversion * x.conversion_price * (1 + x.returns_count) - x.cost_per_click, class_customers))))

        #print(returns_estimator.get_probabilities())
        print(list(map(lambda x: x.get_optimal_arm(), learners)))

    for class_idx, customer_class in enumerate(scen.customer_classes):
        axes[idx].plot(np.cumsum(np.mean(np.subtract(optimal[class_idx], reward_per_class_per_experiment[class_idx]), axis=0)), 'C' + str(class_idx))

    # plt.subplot(212)
    # plt.plot(np.mean(UCB_reward_per_experiment, axis=0), 'r')
    # plt.plot(np.mean(TS_reward_per_experiment, axis=0), 'g')

    # plt.plot(np.mean(opt_reward_per_experiment, axis=0), 'r')
    # plt.plot(np.mean(opt2_reward_per_experiment, axis=0), 'g')

plt.show()
