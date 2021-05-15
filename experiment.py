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

true_rewards = []
for p in scen.prices:
    conversion_rate = np.array([customer_class.conversion(p, discrete=False)*customer_class.daily_clicks(bid[0], noise=False) for customer_class in scen.customer_classes]).sum() / np.array([customer_class.daily_clicks(bid[0], noise=False) for customer_class in scen.customer_classes]).sum()
    returns = np.array([customer_class.returns_function_params['mean']*customer_class.daily_clicks(bid[0], noise=False) for customer_class in scen.customer_classes]).sum() / np.array([customer_class.daily_clicks(bid[0], noise=False) for customer_class in scen.customer_classes]).sum()
    true_rewards.append(conversion_rate * p * (1 + returns))
optimal = true_rewards[np.argmax(true_rewards)]

n_exp = 1
UCB_reward_per_experiment = [[] for _ in range(n_exp)]
TS_reward_per_experiment = [[] for _ in range(n_exp)]

for exp in range(n_exp):
    print(f'Exp: {exp+1}')
    env_UCB = Environment(scen)
    env_TS = Environment(scen)
    context_generator = PriceGreedyContextGenerator(scen.features, scen.customer_classes, PriceUCBLearner, scen.prices, 0.1)
    learner_UCB = PriceTSLearner(scen.prices)
    learner_TS = PriceTSLearner(scen.prices)

    customers_per_day = []
    for day in range(1, scen.rounds_horizon + 1):
        print(f'Day: {day}')

        price = learner_UCB.pull_arm()
        customers, returns = env_UCB.round(bid, [price for _ in range(len(scen.customer_classes))])
        customers_per_day.append(customers)
        if day > 30:
            delayed_customers = customers_per_day.pop(0)
            learner_UCB.update(delayed_customers)
            UCB_reward_per_experiment[exp].append(sum(list(map(lambda x: x.conversion_price * (1 + x.returns_count), list(filter(lambda customer: customer.conversion == 1, delayed_customers))))) / len(delayed_customers))
        print(price)
        # price = learner_TS.pull_arm()
        # customers, returns = env_TS.round(bid, [price for _ in range(len(scen.customer_classes))])
        # learner_TS.update(customers)
        # TS_reward_per_experiment[exp].append(price * len(list(filter(lambda customer: customer.conversion == 1, customers))) / len(customers))

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
