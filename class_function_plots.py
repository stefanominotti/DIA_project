import os
import argparse
import numpy as np

from matplotlib import rcParams
rcParams['font.family'] = 'DeJavu Serif'
rcParams['font.serif'] = ['Palatino']
from matplotlib import pyplot as plt

from main.environment.Scenario import Scenario
from main.environment.Environment import Environment

parser = argparse.ArgumentParser(description='DIA project plots')  
parser.add_argument('--scen', dest='scenario', default='scenario_example')
parser.add_argument('-f', dest='function', choices=['r', 'cr', 'cpc', 'dc'], required=True)

args = parser.parse_args()

if not args.scenario + '.json' in os.listdir('main/environment/scenarios'):
    raise Exception("Scenario doesn't exists")
scen = Scenario(args.scenario)
env = Environment(scen)

def plot_returns():
    for customer_class in scen.customer_classes:
        x = np.linspace(scen.prices[0],scen.prices[-1], 100)
        y = customer_class.returns_function.pdf(x)
        class_label = ('' if customer_class.feature_values[0] else 'not ') + customer_class.feature_labels[0]
        class_label = class_label + (' & ' if customer_class.feature_values[1] else ' & not ') + customer_class.feature_labels[1]
        plt.plot(x, y, label= class_label)
    plt.ylabel('pdf')
    plt.xlabel('number of retunrs in next 30 days')
    plt.legend()
    plt.show()

def plot_convertions():
    for customer_class in scen.customer_classes:
        x = scen.prices
        y = [customer_class.conversion(x_, discrete=False) for x_ in x]
        class_label = ('' if customer_class.feature_values[0] else 'not ') + customer_class.feature_labels[0]
        class_label = class_label + (' & ' if customer_class.feature_values[1] else ' & not ') + customer_class.feature_labels[1]
        plt.plot(x, y, label= class_label)
    plt.ylabel('conversion rate')
    plt.xlabel('price')
    plt.legend()
    plt.show()

def plot_cost_per_click():
    for customer_class in scen.customer_classes:
        x = scen.bids
        y = [customer_class.cost_per_click(x_, noise=False) for x_ in x]
        class_label = ('' if customer_class.feature_values[0] else 'not ') + customer_class.feature_labels[0]
        class_label = class_label + (' & ' if customer_class.feature_values[1] else ' & not ') + customer_class.feature_labels[1]
        plt.plot(x, y, label= class_label)
    plt.ylabel('cost per click')
    plt.xlabel('bids')
    plt.legend()
    plt.show()

def plot_daily_click():
    for customer_class in scen.customer_classes:
        x = scen.bids
        y = [customer_class.daily_clicks(x_, noise=False) for x_ in x]
        class_label = ('' if customer_class.feature_values[0] else 'not ') + customer_class.feature_labels[0]
        class_label = class_label + (' & ' if customer_class.feature_values[1] else ' & not ') + customer_class.feature_labels[1]
        plt.plot(x, y, label= class_label)
    plt.ylabel('daily click')
    plt.xlabel('bids')
    plt.legend(loc='upper left')
    plt.show()

plots = {
    'r': plot_returns,
    'cr': plot_convertions,
    'cpc': plot_cost_per_click,
    'dc': plot_daily_click
}

plots[args.function]()