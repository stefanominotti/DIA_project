from Environment import Environment
from Scenario import Scenario
from PriceUCBLearner import PriceUCBLearner
from PriceTSLearner import PriceTSLearner
from ReturnsEstimator import ReturnsEstimator
from ObjectiveFunction import ObjectiveFunction
import numpy as np
import matplotlib.pyplot as plt

scen = Scenario('scenario_example')
bid = [np.random.choice(scen.bids)]
objectiveFunction = ObjectiveFunction(scen)

print(f'Bid {bid}')

optimal, p1, b1 = objectiveFunction.get_optimal_price_bid()
print(p1)


n_exp = 1
UCB_reward_per_experiment = [[] for _ in range(n_exp)]
TS_reward_per_experiment = [[] for _ in range(n_exp)]

for exp in range(n_exp):
    print(f'Exp: {exp+1}')
    env_UCB = Environment(scen)
    env_TS = Environment(scen)
    learner_UCB = PriceUCBLearner(scen.prices, scen.returns_horizon)
    learner_TS = PriceTSLearner(scen.prices, scen.returns_horizon)
    returns_estimator_UCB = ReturnsEstimator(scen.customer_classes, scen.rounds_horizon, scen.returns_horizon)
    returns_estimator_TS = ReturnsEstimator(scen.customer_classes, scen.rounds_horizon, scen.returns_horizon)

    for day in range(1, scen.rounds_horizon):
        print(f'Day: {day}')

        price = learner_UCB.pull_arm()
        customers, returns = env_UCB.round(bid, [price for _ in range(len(scen.customer_classes))])
        learner_UCB.update(customers)
        print(price)
        returns_estimator_UCB.update(list(filter(lambda customer: customer.conversion == 1, customers)), returns)
        if day > scen.returns_horizon:
            UCB_reward_per_experiment[exp].append(sum([c.conversion * c.conversion_price * (1 + c.returns_count) for c in learner_UCB.customers_per_day[day - scen.returns_horizon]]))

        #price = learner_TS.pull_arm()
        #customers, returns = env_TS.round(bid, [price for _ in range(len(scen.customer_classes))])
        #learner_TS.update(customers)

        #returns_estimator_TS.update(list(filter(lambda customer: customer.conversion == 1, customers)), returns)

        #TS_reward_per_experiment[exp].append(price * len(list(filter(lambda customer: customer.conversion == 1, customers))))
        #print([returns_estimator_TS.pdf(x) for x in range(10)])


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
print(f'optimal: {p1}')
plt.show()
