import os
import json
import numpy as np
from abc import ABC, abstractmethod

from main.utils.ObjectiveFunction import ObjectiveFunction


class Experiment(ABC):
    """
    Abstract class representing an experiment
    """
    
    def __init__(self, scenario, learner_class, price_discrimination, fixed_bid=None, fixed_price=None, n_exp=1):
        """Class constructor

        Args:
            scenario (Scenario): object representing the scenario
            learner_class (PriceLearner/PriceBidGTSLearner/PriceBidGPTSLearner): type of learner used
            price_discrimination (boolean): choose whether performing price discrimination
            fixed_bid (float, optional): bid if bid is fixed. Defaults to None.
            fixed_price (float, optional): price if bid is fixed. Defaults to None.
            n_exp (int, optional): number of iterations to perform. Defaults to 1.
        """

        self.scen = scenario
        self.learner_class = learner_class
        self.price_discrimination = price_discrimination
        self.fixed_bid = [fixed_bid] if fixed_bid else None
        self.fixed_price = [fixed_price, fixed_price, fixed_price] if fixed_price else None
        self.n_exp = n_exp
        self.objectiveFunction = ObjectiveFunction(self.scen, prices=self.fixed_price, bids=self.fixed_bid)
        self.optimal, self.price, self.bid = self.objectiveFunction.get_optimal(self.price_discrimination)
        self.reward_per_experiment = []

    @abstractmethod
    def run():
        """
        Run the experiment
        """

        pass

    def plot(self, axes, color, label):
        """Plot daily regret and daily reward

        Args:
            axes (list): list of axes on which plotting
            color (string): color for the plotted lines
            label (string): curve identifier in the legend
        """

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
        """
        Print optimal reward, price and bid for the chosen problem
        """

        print(f'optimal reward: {self.optimal}')
        print(f'fixed price: {self.fixed_price}' if self.fixed_price else f'optimal price: {self.price}') 
        print(f'fixed bid: {self.fixed_bid}' if self.fixed_bid else f'optimal bid: {self.bid}')

    def save_results(self, exp_name, optimal_arms):
        """Store the results on a file

        Args:
            exp_name (string): name to give to the file
            optimal_arms (list): list of optimal arms to store
        """

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                return json.JSONEncoder.default(self, obj)

        if not os.path.exists('results'):
            os.mkdir('results')
        
        result = {
            'optimal': self.optimal,
            'price':  self.price,
            'bid': self.bid,
            'rewards': self.reward_per_experiment,
            'optimal_arms': optimal_arms
        }

        with open('results/'+exp_name+'.json', 'w') as f:
            json.dump(result, f, cls=NumpyEncoder, indent=2)

    def printProgressBar(self, iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
        """Call in a loop to create terminal progress bar

        Args:
            iteration (integer): current iteration
            total (integer): total iterations
            prefix (string, optional): prefix string. Defaults to ''.
            suffix (string, optional): suffix string. Defaults to ''.
            decimals (integer, optional): positive number of decimals in percent complete. Defaults to 1.
            length (integer, optional): character length of bar. Defaults to 100.
            fill (string, optional): bar fill character. Defaults to '█'.
            printEnd (string, optional): end character (e.g. "\r", "\r\n"). Defaults to "\r".
        """

        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
        # Print New Line on Complete
        if iteration == total: 
            print()


