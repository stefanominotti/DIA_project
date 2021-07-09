import numpy as np
import scipy.stats as stats
import math


class Environment(object):
    def __init__(self, scenario):
        self.scenario = scenario
        self.day = 0
        self.returns = [[] for _ in range(scenario.returns_horizon)]
        self.sub_campaigns = [SubCampaign(scenario.customer_classes)]

    def set_sub_campaigns(self, sub_campaigns):
        customer_classes = [customer_class for sub_campagin in sub_campaigns for customer_class in sub_campagin.customer_classes]
        if len(customer_classes) != len(self.scenario.customer_classes) or set(customer_classes) != self.scenario.customer_classes:
            raise Exception("Invalid sub campaigns")
        self.sub_campaigns = sub_campaigns

    def round(self, bids, prices):
        self.day += 1
        if len(bids) != len(self.sub_campaigns):
            raise Exception("Bids not matching with sub campaigns")
        for sub_campaign_idx, sub_campaign in enumerate(self.sub_campaigns):
            sub_campaign.generate_daily_clicks(bids[sub_campaign_idx])

        if len(prices) != len(self.scenario.customer_classes):
            raise Exception("Prices not matching with customer classes")

        daily_returns = self.returns.pop(0)
        self.returns.append([])
        for customer in daily_returns:
            customer.increment_return()

        daily_customers = []
        for sub_campaign in self.sub_campaigns:
            for customer in sub_campaign.daily_customers:
                customer.convert(self.day, prices[self.scenario.customer_classes.index(customer.customer_class)])
                daily_customers.append(customer)
                if customer.conversion != 0:
                    returns = customer.customer_class.returns()
                    if returns != 0:
                        days_step = math.floor(30 / returns)
                        for idx in range(returns):
                            self.returns[(idx + 1) * days_step - 1].append(customer)

        return daily_customers, daily_returns


class CustomerClass(object):
    def __init__(self, feature_labels, feature_values, conversion_function_interpolation_values, daily_clicks_function_params, returns_function_params, cost_per_click_function_params):
        if len(feature_labels) != len(set(feature_labels)):
            raise Exception("Duplicate features")
        if len(feature_labels) != len(feature_values):
            raise Exception("Feature labels and values not matching")
        self.feature_labels = feature_labels
        self.feature_values = feature_values
        self.conversion_function_interpolation_values = conversion_function_interpolation_values
        self.daily_clicks_function_params = daily_clicks_function_params

        self.returns_function_params = returns_function_params
        returns_mean = self.returns_function_params['mean']
        returns_std = self.returns_function_params['std']
        returns_min = self.returns_function_params['min']
        returns_max = self.returns_function_params['max']
        self.returns_function = stats.truncnorm((returns_min - 0.5 - returns_mean) / returns_std,
                                                 (returns_max + 0.5 - returns_mean) / returns_std,
                                                 loc=returns_mean,
                                                 scale=returns_std)

        self.cost_per_click_function_params = cost_per_click_function_params

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.feature_values == other.feature_values and self.feature_labels == other.feature_labels
        return False

    def __str__(self):
        return str(self.feature_values)

    def conversion(self, price, discrete=True):
        try:
            interpolation_x = self.conversion_function_interpolation_values['x']
            interpolation_y = self.conversion_function_interpolation_values['y']
        except KeyError:
            raise Exception("Missing parameter for conversion function")
        probability = np.interp(price, interpolation_x, interpolation_y, left=1, right=0)
        if discrete:
            return np.random.binomial(1, probability)
        return probability

    def daily_clicks(self, bid, noise=True):
        try:
            max_clicks = self.daily_clicks_function_params['max']
            slope = -np.log(0.01) / self.daily_clicks_function_params['saturating_x']
            noise_std = self.daily_clicks_function_params['noise_std']
        except KeyError:
            raise Exception("Missing parameter for daily clicks function")
        real_func = max_clicks * (1 - np.exp(-slope*bid))
        if noise:
            return round(np.random.normal(real_func, noise_std*real_func))
        return int(round(real_func))

    def cost_per_click(self, bid, noise=True):
        try:
            coefficient = self.cost_per_click_function_params['coefficient']
            noise_std = self.cost_per_click_function_params['noise_std']
        except KeyError:
            raise Exception("Missing parameter for cost per click function")
        real_func = coefficient * bid
        if noise:
            return np.random.normal(real_func, noise_std*real_func)
        return real_func

    def returns(self):
        return int(round(self.returns_function.rvs()[0]))


class Customer(object):
    def __init__(self, customer_class, click_bid, cost_per_click):
        self.customer_class = customer_class
        self.returns_count = 0
        self.click_bid = click_bid
        self.cost_per_click = cost_per_click

    def convert(self, day, price):
        self.conversion = self.customer_class.conversion(price)
        self.conversion_price = price
        self.conversion_day = day

    def increment_return(self):
        self.returns_count += 1


class SubCampaign(object):
    def __init__(self, customer_classes):
        self.customer_classes = customer_classes

    def generate_daily_clicks(self, bid):
        self.daily_clicks = [customer_class.daily_clicks(bid) for customer_class in self.customer_classes]
        self.daily_customers = []
        for customer_class_idx, customer_class in enumerate(self.customer_classes):
            cost_per_click = customer_class.cost_per_click(bid) 
            self.daily_customers.extend([Customer(customer_class, bid, cost_per_click) for _ in range(self.daily_clicks[customer_class_idx])])