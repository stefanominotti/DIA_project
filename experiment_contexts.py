from Environment import Environment
from Scenario import Scenario
from PriceUCBLearner import PriceUCBLearner
from PriceTSLearner import PriceTSLearner
from ReturnsEstimator import ReturnsEstimator
from PriceGreedyContextGenerator import PriceGreedyContextGenerator
import numpy as np
import matplotlib.pyplot as plt

scen = Scenario('scenario_example')
bid = [np.random.choice(scen.bids)]

print(f'Bid {bid}')

optimal_per_class = []
for customer_class in scen.customer_classes:
    true_rewards = []
    for p in scen.prices:
        true_rewards.append(customer_class.conversion(p, discrete=False) * p)
    optimal_per_class.append(true_rewards[np.argmax(true_rewards)])

n_exp = 1
reward_per_class_per_experiment = [[[] for _ in range(n_exp)] for _ in range(len(scen.customer_classes))]

for exp in range(n_exp):
    print(f'Exp: {exp+1}')
    env = Environment(scen)
    context_generator = PriceGreedyContextGenerator(scen.features, scen.customer_classes, PriceUCBLearner, scen.prices, 0.01)
    returns_estimator = ReturnsEstimator(scen.customer_classes, scen.rounds_horizon, scen.returns_horizon)

    for day in range(scen.rounds_horizon):
        print(f'Day: {day}')

        if day % 20 == 0:
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
        context_generator.update(customers)
        for context_idx, context in enumerate(contexts):
            learners[context_idx].update(list(filter(lambda customer: customer.customer_class in context, customers)))

        for class_idx, customer_class in enumerate(scen.customer_classes):
            class_customers = list(filter(lambda customer: customer.customer_class == customer_class, customers))
            reward_per_class_per_experiment[class_idx][exp].append(prices[class_idx] * len(list(filter(lambda customer: customer.conversion == 1, class_customers))) / len(class_customers))

        returns_estimator.update(list(filter(lambda customer: customer.conversion == 1, customers)), returns)

    #print(returns_estimator.get_probabilities())
    print(list(map(lambda x: x.get_optimal_arm(), learners)))

plt.figure()

for class_idx, customer_class in enumerate(scen.customer_classes):
    plt.plot(np.cumsum(np.mean(np.subtract(optimal_per_class[class_idx], reward_per_class_per_experiment[class_idx]), axis=0)), 'C' + str(class_idx))

# plt.subplot(212)
# plt.plot(np.mean(UCB_reward_per_experiment, axis=0), 'r')
# plt.plot(np.mean(TS_reward_per_experiment, axis=0), 'g')

# plt.plot(np.mean(opt_reward_per_experiment, axis=0), 'r')
# plt.plot(np.mean(opt2_reward_per_experiment, axis=0), 'g')

plt.show()
