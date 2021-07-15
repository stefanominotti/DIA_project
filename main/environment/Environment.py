import numpy as np
import scipy.stats as stats
import math


class Environment(object):
    """
    The environment class
    """

    def __init__(self, scenario):
        """Class constructor

        Args:
            scenario (Scenario): scenario on which the environment is based
        """

        self.scenario = scenario
        self.day = 0
        self.returns = [[] for _ in range(scenario.returns_horizon)]
        self.sub_campaigns = [SubCampaign(scenario.customer_classes)]

    def round(self, bids, prices):
        """Start a new round (day)

        Args:
            bids (list): bid for each campaign (we will always use one campaign)
            prices (list): price for each class

        Raises:
            Exception: invalid number of bids 
            Exception: invalid number of prices

        Returns:
            tuple: tuple containing daily new customers and daily returning customers
        """

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
    """
    Object representing a customer class
    """

    def __init__(self, feature_labels, feature_values, conversion_function_interpolation_values, daily_clicks_function_params, returns_function_params, cost_per_click_function_params):
        """Class constructore

        Args:
            feature_labels (list): labels for the features
            feature_values (list): values for the features
            conversion_function_interpolation_values (dict): dict of x and y values for the conversion function
            daily_clicks_function_params (dict): parameters for the daily clicks function
            returns_function_params (dict): parameters for the returns probability distribution function
            cost_per_click_function_params (dict): parameters for the daily clicks function

        Raises:
            Exception: features are duplicates
            Exception: feature labels and values not matching
        """

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
        """Define equivalence for two customer classes (feature labels and values have to be the same)

        Args:
            other (object): object to compare

        Returns:
            bool: comparison result
        """

        if isinstance(other, self.__class__):
            return self.feature_values == other.feature_values and self.feature_labels == other.feature_labels
        return False

    def __str__(self):
        """Convert class to string

        Returns:
            string: string describing the customer class
        """

        return str(self.feature_values)

    def conversion(self, price, discrete=True):
        """Return the conversion value given a price

        Args:
            price (float): the price at which evaluating the function
            discrete (bool, optional): choose whether returning a probability or a sample (0, 1). Defaults to True.

        Raises:
            Exception: missing parameter for conversion function

        Returns:
            float: 0 or 1 if discrete is True else the probability
        """

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
        """Return the number of daily clicks given a bid

        Args:
            bid (bool): the bid at which evaluating the function
            noise (bool, optional): choose whether adding noise or not. Defaults to True.

        Raises:
            Exception: missing parameter for daily clicks function

        Returns:
            float: the value of the function (+ noise)
        """

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
        """Return cost per click given a bid

        Args:
            bid (bool): the bid at which evaluating the function
            noise (bool, optional): choose whether adding noise or not. Defaults to True.

        Raises:
            Exception: missing parameter for cost per click function

        Returns:
            integer/float: the value of the function (+ noise)
        """

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
    """
    Class representing a customer
    """

    def __init__(self, customer_class, click_bid):
        """Class constructor

        Args:
            customer_class (CustomerClass): the customer class to which the customer belong
            click_bid (float): the bid with which the customer clicked on the advertisement
        """

        self.customer_class = customer_class
        self.returns_count = 0
        self.click_bid = click_bid
        self.cost_per_click = customer_class.cost_per_click(click_bid) 

    def convert(self, day, price):
        """Return 0 or 1 if the customer convertef given a price

        Args:
            day (integer): the actual day of the conversion
            price (float): the price proposed to the customer
        """

        self.conversion = self.customer_class.conversion(price)
        self.conversion_price = price
        self.conversion_day = day

    def increment_return(self):
        """
        Increment the number of times the user returned to buy the product
        """

        self.returns_count += 1


class SubCampaign(object):
    """
    Class representing a sub campaing
    """

    def __init__(self, customer_classes):
        """Class constructor

        Args:
            customer_classes (list): customer classes belonging to the campaign
        """

        self.customer_classes = customer_classes

    def generate_daily_clicks(self, bid):
        """Generate daily clicks given a bid

        Args:
            bid (float): bid chosen for the campaign
        """

        self.daily_clicks = [customer_class.daily_clicks(bid) for customer_class in self.customer_classes]
        self.daily_customers = []
        for customer_class_idx, customer_class in enumerate(self.customer_classes):
            self.daily_customers.extend([Customer(customer_class, bid) for _ in range(self.daily_clicks[customer_class_idx])])