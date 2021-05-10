from Environment import Environment
from Scenario import Scenario
from PriceUCBLearner import PriceUCBLearner
from PriceTSLearner import PriceTSLearner
from ReturnsEstimator import ReturnsEstimator
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
horizon = 60
UCB_reward_per_experiment = [[] for _ in range(n_exp)]
UCB_optimal_per_experiment = [[] for _ in range(n_exp)]
TS_reward_per_experiment = [[] for _ in range(n_exp)]
TS_optimal_per_experiment = [[] for _ in range(n_exp)]

for exp in range(n_exp):
    print(f'Exp: {exp}')
    env_UCB = Environment(scen)
    env_TS = Environment(scen)
    env_opt = Environment(scen)

    learner_UCB = PriceUCBLearner(scen.prices)
    learner_TS = PriceTSLearner(scen.prices)
    returns_estimator = ReturnsEstimator(scen.customer_classes, scen.rounds_horizon, scen.returns_horizon)

    total_customers = 0
    for day in range(1, scen.rounds_horizon + 1):
        print(f'Day: {day}')

        price = learner_UCB.pull_arm()
        daily_customers, daily_returns = env_UCB.round(bid, [price for _ in range(len(scen.customer_classes))])
        learner_UCB.update(price, daily_customers)
        daily_conv_count = len(list(filter(lambda x: x.conversion == 1, daily_customers)))
        UCB_reward_per_experiment[exp].append(price * daily_conv_count)
        UCB_optimal_per_experiment[exp].append(optimal * len(daily_customers))

        price = learner_TS.pull_arm()
        daily_customers, daily_returns = env_TS.round(bid, [price for _ in range(len(scen.customer_classes))])
        learner_TS.update(price, daily_customers)
        daily_conv_count = len(list(filter(lambda x: x.conversion == 1, daily_customers)))
        TS_reward_per_experiment[exp].append(price * daily_conv_count)
        TS_optimal_per_experiment[exp].append(optimal * len(daily_customers))

        returns_estimator.update(list(filter(lambda x: x.conversion == 1, daily_customers)), daily_returns)

        print(returns_estimator.returns)

    print(returns_estimator.get_probabilities())



    



plt.figure()

plt.subplot(211)
plt.plot(np.cumsum(np.mean(np.subtract(UCB_optimal_per_experiment, UCB_reward_per_experiment), axis=0)), 'r')
plt.plot(np.cumsum(np.mean(np.subtract(TS_optimal_per_experiment, TS_reward_per_experiment), axis=0)), 'g')

plt.subplot(212)
plt.plot(np.mean(UCB_reward_per_experiment, axis=0), 'r')
plt.plot(np.mean(TS_reward_per_experiment, axis=0), 'g')

plt.show()


#price: 12, conversions: 988
#price: 13, conversions: 848
#price: 14, conversions: 539
