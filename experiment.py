from Environment import Environment
from Scenario import Scenario
from PriceUCBLearner import PriceUCBLearner
from PriceTSLearner import PriceTSLearner
from ReturnsEstimator import ReturnsEstimator
from ContexGeneratorLearner import ContextGeneratorLEarner
import numpy as np
import matplotlib.pyplot as plt

scen = Scenario('scenario_example')
bid = [np.random.choice(scen.bids)]

print(f'Bid {bid}')

true_rewards = []
for p in scen.prices:
    true_rewards.append(np.array([customer_class.conversion(p, discrete=False)*customer_class.daily_clicks(bid[0], noise=False) for customer_class in scen.customer_classes]).sum() / np.array([customer_class.daily_clicks(bid[0], noise=False) for customer_class in scen.customer_classes]).sum() * p)
optimal = true_rewards[np.argmax(true_rewards)]

n_exp = 1
UCB_reward_per_experiment = [[] for _ in range(n_exp)]
TS_reward_per_experiment = [[] for _ in range(n_exp)]

for exp in range(n_exp):
    print(f'Exp: {exp+1}')
    env_UCB = Environment(scen)
    env_TS = Environment(scen)
    learner_Context = ContextGeneratorLEarner(scen.prices, scen.customer_classes, PriceUCBLearner, 0.1)
    learner_UCB = PriceUCBLearner(scen.prices)
    learner_TS = PriceTSLearner(scen.prices)
    returns_estimator = ReturnsEstimator(scen.customer_classes, scen.rounds_horizon, scen.returns_horizon)

    for day in range(1, scen.rounds_horizon + 1):
        print(f'Day: {day}')

        price = learner_UCB.pull_arm()
        customers, returns = env_UCB.round(bid, [price for _ in range(len(scen.customer_classes))])
        learner_UCB.update(customers)
        learner_Context.update(customers)
        UCB_reward_per_experiment[exp].append(price * len(list(filter(lambda customer: customer.conversion == 1, customers))) / len(customers))

        price = learner_TS.pull_arm()
        customers, returns = env_TS.round(bid, [price for _ in range(len(scen.customer_classes))])
        learner_TS.update(customers)
        TS_reward_per_experiment[exp].append(price * len(list(filter(lambda customer: customer.conversion == 1, customers))) / len(customers))

        returns_estimator.update(list(filter(lambda customer: customer.conversion == 1, customers)), returns)

        if day % 4 == 0:
            print(learner_Context.generate_contex_tree(learner_UCB, [[0,0], [0,1], [1,0]]))

    #print(returns_estimator.get_probabilities())
    print(learner_UCB.get_optimal_arm(), learner_TS.get_optimal_arm())

plt.figure()

plt.subplot(211)
plt.plot(np.cumsum(np.mean(np.subtract(optimal, UCB_reward_per_experiment), axis=0)), 'r')
plt.plot(np.cumsum(np.mean(np.subtract(optimal, TS_reward_per_experiment), axis=0)), 'g')

plt.subplot(212)
plt.plot(np.mean(UCB_reward_per_experiment, axis=0), 'r')
plt.plot(np.mean(TS_reward_per_experiment, axis=0), 'g')

# plt.plot(np.mean(opt_reward_per_experiment, axis=0), 'r')
# plt.plot(np.mean(opt2_reward_per_experiment, axis=0), 'g')

plt.show()
