import numpy as np
from scipy.stats import rv_discrete


class ReturnsEstimator(object):
    def __init__(self, returns_horizon):
        self.returns_horizon = returns_horizon
        self.total_customers = 0
        self.returns_counts = [0 for _ in range(self.returns_horizon + 1)]

    def new_customer(self, customer):
        self.total_customers += 1
        self.returns_counts[customer.returns_count] += 1

    def new_return(self, customer):
        self.returns_counts[customer.returns_count - 1] -= 1
        self.returns_counts[customer.returns_count] += 1

    def estimated_function(self):
        xk = range(self.returns_horizon + 1)
        pk = [self.pdf(x) for x in xk]
        return rv_discrete(values=(xk, pk))

    def sample(self):
        return np.random.normal(self.mean(), self.std())

    def pdf(self, x): 
        if self.total_customers == 0:
            return 1/(self.returns_horizon + 1)
        return self.returns_counts[x] / self.total_customers 

    def mean(self):
        return self.estimated_function().mean()

    def std(self):
        return self.estimated_function().std() / self.total_customers
