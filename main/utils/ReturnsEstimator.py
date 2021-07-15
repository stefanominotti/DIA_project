import numpy as np
from scipy.stats import rv_discrete


class ReturnsEstimator(object):
    """
    Estimator for the returns probability distribution
    """
    
    def __init__(self, returns_horizon):
        """Class constructor

        Args:
            returns_horizon (integer): maximum number of times a customer can return
        """

        self.returns_horizon = returns_horizon
        self.total_customers = 0
        self.returns_counts = [0 for _ in range(self.returns_horizon + 1)]

    def new_customer(self, customer):
        """Update the estimation with a new customer

        Args:
            customer (Customer): the customer
        """

        self.total_customers += 1
        self.returns_counts[customer.returns_count] += 1

    def estimated_function(self):
        """Return the estimated probability distribution

        Returns:
            rv_discrete: the probability function
        """

        xk = range(self.returns_horizon + 1)
        pk = [self.pdf(x) for x in xk]
        return rv_discrete(values=(xk, pk))

    def sample(self):
        """Returns a sample from the distribution

        Returns:
            integet: the sample
        """

        return np.random.normal(self.mean(), self.std() / self.total_customers)

    def pdf(self, x):
        """Return the pdf for a value

        Args:
            x (integer): the value on which calculating the pdf

        Returns:
            float: pdf value for x
        """

        if self.total_customers == 0:
            return 1/(self.returns_horizon + 1)
        return self.returns_counts[x] / self.total_customers 

    def mean(self):
        """Return the expected value of the distribution

        Returns:
            float: the expected value of the distribution
        """

        return self.estimated_function().mean()

    def std(self):
        """Return the standard deviation of the distribution

        Returns:
            float: the standard deviation of the distribution
        """

        return self.estimated_function().std()
