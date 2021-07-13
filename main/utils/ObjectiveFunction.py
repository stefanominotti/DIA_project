import numpy as np


class ObjectiveFunction(object):
    def __init__(self, scenario, prices=None, bids=None):
        self.scen = scenario
        self.prices = prices if prices != None else self.scen.prices
        self.bids = bids if bids != None else self.scen.bids

    def get_optimal(self, price_discrimination=False):
        returns_per_class = np.array([customer_class.returns_function.mean() for customer_class in self.scen.customer_classes])
        optimal_price_per_bid = []
        rewards_per_bid_per_class = []
        rewards_per_bid = []
        for bid in self.bids:
            daily_clicks_per_class = np.array([customer_class.daily_clicks(bid, noise=False) for customer_class in self.scen.customer_classes])
            cpc_per_class = np.array([customer_class.cost_per_click(bid, noise=False) for customer_class in self.scen.customer_classes])
            price_rewards_per_price = []
            price_reward_per_class_per_price = []
            for price in self.prices:
                conversion_rate_per_class = np.array([customer_class.conversion(price, discrete=False) for customer_class in self.scen.customer_classes]) 
                price_reward_per_class_per_price.append(conversion_rate_per_class * price * (1 + returns_per_class))
    
                # aggregate_returns = (daily_clicks_per_class * conversion_rate_per_class * returns_per_class).sum() / (daily_clicks_per_class * conversion_rate_per_class).sum()
                # aggregate_conversion_rate = (daily_clicks_per_class * conversion_rate_per_class).sum() / daily_clicks_per_class.sum()
                # price_rewards_per_price.append(aggregate_conversion_rate * price * (1 + aggregate_returns))

            if price_discrimination:
                optimal_price_per_bid.append(np.take(self.prices, np.argmax(price_reward_per_class_per_price, axis=0)))
                optimal_price_reward_per_class = np.max(price_reward_per_class_per_price, axis=0)
            else:
                price_rewards_per_price = np.sum(price_reward_per_class_per_price * daily_clicks_per_class, axis=1) / daily_clicks_per_class.sum()
                optimal_price_per_bid.append(np.take(self.prices, np.argmax(price_rewards_per_price)))
                optimal_price_reward_per_class = np.take(price_reward_per_class_per_price, np.argmax(price_rewards_per_price), axis=0)
            
            rewards_per_bid_per_class.append(daily_clicks_per_class * (optimal_price_reward_per_class - cpc_per_class))
            rewards_per_bid.append((daily_clicks_per_class * (optimal_price_reward_per_class - cpc_per_class)).sum())
        
        optimal_bid = np.take(self.bids, np.argmax(rewards_per_bid))
        optimal_price = np.take(optimal_price_per_bid, np.argmax(rewards_per_bid), axis=0)

        if price_discrimination:
            optimal_reward = np.take(rewards_per_bid_per_class, np.argmax(rewards_per_bid), axis=0)
        else:
            optimal_reward = [np.max(rewards_per_bid)]

        return optimal_reward, optimal_price, optimal_bid

    def get_optimal_discrimination(self):
        rewards_per_prices = []
        returns_values = np.array([sum([x*(customer_class.returns_function.cdf(x+0.5) - customer_class.returns_function.cdf(x-0.5)) for x in range(self.scen.returns_horizon + 1)]) for customer_class in self.scen.customer_classes])
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

    def get_optimal_no_discrimination(self):
        total_rewards = []
        best_bids = []
        returns_values = np.array([sum([x*(customer_class.returns_function.cdf(x+0.5) - customer_class.returns_function.cdf(x-0.5)) for x in range(self.scen.returns_horizon + 1)]) for customer_class in self.scen.customer_classes])
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


