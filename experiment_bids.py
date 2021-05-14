from Environment import Environment
from Scenario import Scenario
from PriceUCBLearner import PriceUCBLearner
from PriceTSLearner import PriceTSLearner
from ReturnsEstimator import ReturnsEstimator
from PriceGreedyContextGenerator import PriceGreedyContextGenerator
from BidLearner import BidLearner
import numpy as np
import matplotlib.pyplot as plt

scen = Scenario('scenario_example')
price = 60
prices = [price, price, price]

print(f'Price {price}')

true_rewards = []
for b in scen.bids:
    conversion_rate = np.array([customer_class.conversion(price, discrete=False)*customer_class.daily_clicks(b, noise=False) for customer_class in scen.customer_classes]).sum() / np.array([customer_class.daily_clicks(b, noise=False) for customer_class in scen.customer_classes]).sum()
    returns = np.array([customer_class.returns_function_params['mean']*customer_class.daily_clicks(b, noise=False) for customer_class in scen.customer_classes]).sum() / np.array([customer_class.daily_clicks(b, noise=False) for customer_class in scen.customer_classes]).sum()
    cpc = np.array([customer_class.cost_per_click(b, noise=False)*customer_class.daily_clicks(b, noise=False) for customer_class in scen.customer_classes]).sum() / np.array([customer_class.daily_clicks(b, noise=False) for customer_class in scen.customer_classes]).sum()
    daily_clicks = np.array([customer_class.daily_clicks(b, noise=False) for customer_class in scen.customer_classes]).sum()
    true_rewards.append(daily_clicks * (conversion_rate * price * (1 + returns) - cpc))
optimal = true_rewards[np.argmax(true_rewards)]
optimal_bid = scen.bids[np.argmax(true_rewards)]

print(true_rewards)
print(optimal, optimal_bid)

n_exp = 1
reward_per_experiment = [[] for _ in range(n_exp)]

for exp in range(n_exp):
    print(f'Exp: {exp+1}')
    env = Environment(scen)
    learner = BidLearner(scen.bids, 0.2)
    returns_estimator = ReturnsEstimator(scen.customer_classes, scen.rounds_horizon, scen.returns_horizon)

    for day in range(scen.rounds_horizon):
        print(f'Day: {day}')

        bid = learner.pull_arm()
        customers, returns = env.round([bid], prices)
        returns_estimator.update(list(filter(lambda customer: customer.conversion == 1, customers)), returns)
        reward = 0
        for customer_class in scen.customer_classes:
            class_customers = list(filter(lambda customer: customer.customer_class == customer_class, customers))
            print(sum(list(map(lambda x: x.cost_per_click, class_customers))))
            reward += price * len(list(filter(lambda x: x.conversion == 1, class_customers))) * (1 + returns_estimator.get_average_returns()[scen.customer_classes.index(customer_class)]) - sum(list(map(lambda x: x.cost_per_click, class_customers)))
        learner.update(bid, reward)
        reward_per_experiment[exp].append(reward)

        print(learner.get_optimal_arm())
    #print(returns_estimator.get_probabilities())


plt.figure()

plt.plot(np.cumsum(np.mean(np.subtract(optimal, reward_per_experiment), axis=0)), 'C0')

# plt.subplot(212)
# plt.plot(np.mean(UCB_reward_per_experiment, axis=0), 'r')
# plt.plot(np.mean(TS_reward_per_experiment, axis=0), 'g')

# plt.plot(np.mean(opt_reward_per_experiment, axis=0), 'r')
# plt.plot(np.mean(opt2_reward_per_experiment, axis=0), 'g')

plt.show()
