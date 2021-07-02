import numpy as np
from scipy.stats import rv_discrete


class ReturnsEstimator(object):
    def __init__(self, returns_horizon):
        self.returns_horizon = returns_horizon
        self.total_customers = 0
        self.returns_counts = [0 for _ in range(self.returns_horizon + 1)]
        
    def update(self, customers, returns):
        for customer in customers:
            self.total_customers += 1
            self.returns_counts[customer.returns_count] += 1
        for customer in returns:
            self.returns_counts[customer.returns_count - 1] -= 1
            self.returns_counts[customer.returns_count] += 1

    def rsv(self):
        xk = range(self.returns_horizon + 1)
        pk = [self.pdf(x) for x in xk]
        return rv_discrete(values=(xk, pk)).rvs()

    def pdf(self, x): 
        if self.total_customers == 0:
            return 1/(self.returns_horizon + 1)
        return self.returns_counts[x] / self.total_customers 

    def weihgted_sum(self):
        return np.sum([self.pdf(x)*x for x in range(self.returns_horizon + 1)])
