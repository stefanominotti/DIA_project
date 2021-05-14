import numpy as np


class ReturnsEstimator(object):
    def __init__(self, customer_classes, rounds_horizon, returns_horizon):
        self.t = 0
        self.rounds_horizon = rounds_horizon
        self.returns_horizon = returns_horizon
        self.customer_classes = customer_classes
        self.total_customers = np.zeros(len(self.customer_classes))
        self.returns = np.zeros((self.returns_horizon + 1, len(customer_classes)))

    def update(self, new_conversions, returns):
        self.t += 1

        if self.t > self.rounds_horizon - self.returns_horizon:
            customers = list(filter(lambda x: x.conversion_day <= self.rounds_horizon - self.returns_horizon, returns))
        else:
            customers = [*new_conversions, *returns]

        for customer in customers:
            customer_class_idx = self.customer_classes.index(customer.customer_class)
            if customer.returns_count == 0:
                self.total_customers[customer_class_idx] += 1
            self.returns[customer.returns_count, customer_class_idx] += 1
            if customer.returns_count != 0:
                self.returns[customer.returns_count-1, customer_class_idx] -= 1

    def get_probabilities(self):
        return self.returns / self.total_customers

    def get_average_returns(self):
        indices = np.arange(len(self.returns))
        return (self.get_probabilities() * indices[:, np.newaxis]).sum(axis=0)
