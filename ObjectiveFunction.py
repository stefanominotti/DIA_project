import numpy as np
import scipy.stats as stats
import math


class ObjectiveFunction(object):
    def __init__(self, scenario, prices=None, bids=None):
        self.scen = scenario
        self.prices = prices if prices != None else self.scen.prices
        self.bids = bids if bids != None else self.scen.bids

    def get_optimals_price_bid_per_class(self):
        rewards_per_prices = []
        returns_values = np.array([sum([x*customer_class.returns_function.pdf(x) for x in range(self.scen.returns_horizon)]) for customer_class in self.scen.customer_classes])
        for p in self.prices:
            rewards_per_prices.append(np.array([customer_class.conversion(p, discrete=False) * p for customer_class in self.scen.customer_classes]))
        rewards_per_prices = np.concatenate(list(map(lambda x: np.transpose([x]), rewards_per_prices)), axis=1)
        prices_rewards = np.max(rewards_per_prices, axis=1)
        optimal_prices =  np.take(self.prices, np.argmax(rewards_per_prices, axis=1))
        total_rewards = []
        for b in self.bids:
            daily_clicks = np.array([customer_class.daily_clicks(b, noise=False) for customer_class in self.scen.customer_classes])
            cpc = np.array([customer_class.cost_per_click(b, noise=False) for customer_class in self.scen.customer_classes])
            total_rewards.append((prices_rewards*(1 + returns_values)-cpc)*daily_clicks)
        total_rewards = np.concatenate(list(map(lambda x: np.transpose([x]), total_rewards)), axis=1)
        optimals = np.max(total_rewards, axis=1)
        optimal_bids =  np.take(self.bids, np.argmax(total_rewards, axis=1))
        return optimals, optimal_prices, optimal_bids

    def get_optimal_price_bid(self):
        total_rewards = []
        best_bids = []
        returns_values = np.array([sum([x*(customer_class.returns_function.cdf(x+0.5) - customer_class.returns_function.cdf(x-0.5)) for x in range(self.scen.returns_horizon)]) for customer_class in self.scen.customer_classes])
        for p in self.prices:
            price_reward = np.array([customer_class.conversion(p, discrete=False) * p for customer_class in self.scen.customer_classes])
            bid_rewards = []
            for b in self.bids:
                daily_clicks = np.array([customer_class.daily_clicks(b, noise=False) for customer_class in self.scen.customer_classes])
                cpc = np.array([customer_class.cost_per_click(b, noise=False) for customer_class in self.scen.customer_classes])
                bid_rewards.append(((price_reward * (1 + returns_values) - cpc)*daily_clicks).sum())
            total_rewards.append(np.max(bid_rewards))
            best_bids.append(self.bids[np.argmax(bid_rewards)])
        optimal = np.max(total_rewards)
        optimal_price = self.prices[np.argmax(total_rewards)]
        optimal_bid =  best_bids[np.argmax(total_rewards)]
        return optimal, optimal_price, optimal_bid


