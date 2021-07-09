from JointLearner import JointLearner
from BidGTSLearner import BidGTSLearner
from ObjectiveFunction import ObjectiveFunction
from Environment import Environment
from Scenario import Scenario
from PriceUCBLearner import PriceUCBLearner
from PriceTSLearner import PriceTSLearner
from ReturnsEstimator import ReturnsEstimator
from PriceGreedyContextGenerator import PriceGreedyContextGenerator
from BidGPTSLearner import BidGPTSLearner
import numpy as np
import matplotlib.pyplot as plt

scen = Scenario('scenario_example')
price = 2.5
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

objective_function = ObjectiveFunction(scenario=scen, prices=[price])
optimal, optimal_price, optimal_bid = objective_function.get_optimal_no_discrimination()

print(optimal, optimal_price, optimal_bid)

conversion_rates = []
returns_values = []
conversion_rates_per_class = np.array([customer_class.conversion(price, discrete=False) for customer_class in scen.customer_classes])
returns_values_per_class = np.array([customer_class.returns_function.mean() for customer_class in scen.customer_classes])

for b in scen.bids:
    daily_clicks = np.array([customer_class.daily_clicks(b, noise=False) for customer_class in scen.customer_classes])
    conversion_rates.append((conversion_rates_per_class * daily_clicks).sum() / daily_clicks.sum())
    returns_values.append((returns_values_per_class * daily_clicks).sum() / daily_clicks.sum())

conversion_rates = np.array(conversion_rates)
returns_values = np.array(returns_values)

n_exp = 5
reward_per_experiment = [[] for _ in range(n_exp)]

for exp in range(n_exp):
    print(f'Exp: {exp+1}')
    env = Environment(scen)
    learner = JointLearner(scen.prices, scen.bids, 0.2)
    context_generator = PriceGreedyContextGenerator(scen.features, scen.customer_classes, PriceTSLearner, scen.prices, scen.returns_horizon, 0.01)
    customers_per_day = []
    prices_per_day = []
    bids_per_day = []

    contexts, price_learners = context_generator.get_best_contexts()
    learner.set_price_learner()

    for day in range(scen.rounds_horizon):
        print(f'Day: {day+1}')
        
        price, bid = learner.pull_arm()
        customers, returns = env.round([bid], [price, price, price])
        customers_per_day.append(customers)
        prices_per_day.append(price)
        bids_per_day.append(bid)
        reward = 0
        learner.update(price, bid, customers, returns)

        print(price, bid)

        if day > 30:
            delayed_customers = customers_per_day.pop(0)
            bid = bids_per_day.pop(0)
            reward = 0
            converted_customers = list(filter(lambda customer: customer.conversion == 1, delayed_customers))
            reward += sum(list(map(lambda customer: customer.conversion_price * (1 + customer.returns_count), converted_customers)))
            reward -= sum(list(map(lambda customer: customer.cost_per_click, delayed_customers)))
            
            reward_per_experiment[exp].append(reward)
            
    #print(returns_estimator.get_probabilities())


plt.figure()

plt.plot(np.cumsum(np.mean(np.subtract(optimal, reward_per_experiment), axis=0)), 'C0')

# plt.subplot(212)
# plt.plot(np.mean(UCB_reward_per_experiment, axis=0), 'r')
# plt.plot(np.mean(TS_reward_per_experiment, axis=0), 'g')

# plt.plot(np.mean(opt_reward_per_experiment, axis=0), 'r')
# plt.plot(np.mean(opt2_reward_per_experiment, axis=0), 'g')

plt.show()
