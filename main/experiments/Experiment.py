from math import remainder
import numpy as np

from main.utils.ObjectiveFunction import ObjectiveFunction


class Experiment(object):
    def __init__(self, scenario, learner_class, price_discrimination, fixed_bid=None, fixed_price=None, n_exp=1):
        self.scen = scenario
        self.learner_class = learner_class
        self.price_discrimination = price_discrimination
        self.fixed_bid = [fixed_bid] if fixed_bid else None
        self.fixed_price = [fixed_price, fixed_price, fixed_price] if fixed_price else None
        self.n_exp = n_exp
        self.objectiveFunction = ObjectiveFunction(self.scen, prices=self.fixed_price, bids=self.fixed_bid)
        self.optimal, self.price, self.bid = self.objectiveFunction.get_optimal(self.price_discrimination)
        self.reward_per_experiment = []

    def plot(self, axes, color, label):
        if len(self.reward_per_experiment) > 1:
            for idx in range(len(self.reward_per_experiment)):
                axes[0,idx].plot(np.cumsum(np.mean(np.subtract(self.optimal[idx], self.reward_per_experiment[idx]), axis=0)), color=color, label=label)
                axes[1,idx].plot(np.mean(self.reward_per_experiment[idx], axis=0), color=color, label=label)
                customer_class = self.scen.customer_classes[idx]
                axes[0,idx].set_title(('' if customer_class.feature_values[0] else 'not ') + customer_class.feature_labels[0] + \
                                      (' & ' if customer_class.feature_values[1] else ' & not ') + customer_class.feature_labels[1])
                axes[0,0].set_ylabel('Regret')   
                axes[1,idx].set_xlabel('Days') 
                axes[1,0].set_ylabel('Daily reward') 
        else:
            for idx in range(len(self.reward_per_experiment)):
                axes[0].plot(np.cumsum(np.mean(np.subtract(self.optimal[idx], self.reward_per_experiment[idx]), axis=0)), color=color, label=label)
                axes[1].plot(np.mean(self.reward_per_experiment[idx], axis=0), color=color, label=label)
                axes[0].set_ylabel('Regret')   
                axes[1].set_ylabel('Daily reward')  
                axes[1].set_xlabel('Days')   

    def print_optimal(self):
        print(f'optimal reward: {self.optimal}')
        print(f'fixed price: {self.fixed_price}' if self.fixed_price else f'optimal price: {self.price}') 
        print(f'fixed bid: {self.fixed_bid}' if self.fixed_bid else f'optimal bid: {self.bid}') 

    def printProgressBar(self, iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
        # Print New Line on Complete
        if iteration == total: 
            print()



