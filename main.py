import os
from matplotlib import rcParams
rcParams['font.family'] = 'DeJavu Serif'
rcParams['font.serif'] = ['Palatino']
import matplotlib.pyplot as plt

from main.environment.Scenario import Scenario
from main.experiments.PriceExperiment import PriceExperiment
from main.experiments.PriceContextExperiment import PriceContextExperiment
from main.experiments.BidExperiment import BidExperiment
from main.experiments.JointExperiment import JointExperiment
from main.experiments.JointContextExperiment import JointContextExperiment

from main.bandits.pricing.PriceTSLearner import PriceTSLearner
from main.bandits.pricing.PriceUCBLearner import PriceUCBLearner

from main.contexts.PriceGreedyContextGenerator import PriceGreedyContextGenerator
from main.contexts.PriceBruteForceContextGenerator import PriceBruteForceContextGenerator

from main.bandits.joint.PriceBidGTSLearner import PriceBidGTSLearner
from main.bandits.joint.PriceBidGPTSLearner import PriceBidGPTSLearner

import argparse
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='DIA project experiment')  
parser.add_argument('--exp', dest='experiment', choices=['price', 'bid', 'joint'], required=True)
parser.add_argument('--scen', dest='scenario', required=True)
parser.add_argument('--pd', dest='price_discrimination', default='F', choices=['F', 'T'])
parser.add_argument('-b', dest='fixed_bid', type=float)
parser.add_argument('-p', dest='fixed_price', type=float)
parser.add_argument('--ne', dest='n_exp', default=1, type=int)
parser.add_argument('--npt', dest='negative_probability_threshold', default=0.2, type=float)
parser.add_argument('--incgen', dest='incremental_generation', default='T', choices=['F', 'T'])
parser.add_argument('--genrate', dest='generation_rate', type=int)
parser.add_argument('--conf', dest='confidence', default=0.5, type=float)
parser.add_argument('--approx', dest='approximate', default='T', choices=['F', 'T'])
parser.add_argument('-l', dest='learner_class', choices=['UCB', 'TS', 'GTS', 'GPTS'], required=True, nargs='+')
parser.add_argument('-c', dest='contextGenerator', choices=['G', 'BF'], nargs='+')
args = parser.parse_args()

# Transfor boolean variables from string to bool
price_discrimination = True if args.price_discrimination == 'T' else False
incremental_generation = True if args.incremental_generation == 'T' else False
approximate = True if args.approximate == 'T' else False

# Check and load scenario
if not args.scenario + '.json' in os.listdir('main/environment/scenarios'):
    raise Exception("Scenario doesn't exists")
scenario = Scenario(args.scenario)

# Check if fixed bid and price are in scenrario
if (args.fixed_bid and not args.fixed_bid in scenario.bids) or \
    (args.fixed_price and not args.fixed_price in scenario.prices):
    raise Exception('Invalid bid or price')

# Select the experiment
experiment = {
    'price': PriceContextExperiment if price_discrimination else PriceExperiment,
    'bid': BidExperiment,
    'joint': JointContextExperiment if price_discrimination else JointExperiment,
}
experiment = experiment[args.experiment]

if args.experiment == 'price' and (args.fixed_bid == None or args.fixed_price != None):
    raise Exception("Add fixed bid and remove fixed price")
if args.experiment == 'bid' and (args.fixed_price == None or args.fixed_bid != None):
    raise Exception("Add fixed price and remove fixed bid")
if args.experiment == 'joint' and (args.fixed_price != None or args.fixed_bid != None):
    raise Exception("Remove fixed price and fixed bid")

# Select and check the learner
learner_class_map = {
    'UCB': PriceUCBLearner if args.experiment == 'price' else None,
    'TS': PriceTSLearner if args.experiment == 'price' else None,
    'GTS': PriceBidGTSLearner if args.experiment != 'price' else None,
    'GPTS': PriceBidGPTSLearner if args.experiment != 'price' else None
}
learner_classes = [learner_class_map[arg] for arg in args.learner_class]
if None in learner_classes:
    raise Exception("Experiment and Learner are inconsisten")

# Check and add the necessary arguments
context_generators = []
kwargs = {}
if price_discrimination:
    if (args.contextGenerator == None or args.generation_rate == None or args.confidence == None):
        raise Exception("Missing arguments for context generators")
    context_generator_map = {
        'G': PriceGreedyContextGenerator,
        'BF': PriceBruteForceContextGenerator
    }
    context_generators = [context_generator_map[arg] for arg in args.contextGenerator]
    kwargs['generation_rate'] = args.generation_rate
    kwargs['confidence'] = args.confidence
    kwargs['incremental_generation'] = incremental_generation

if args.experiment != 'price':
    if args.negative_probability_threshold == None:
        raise Exception("Missing negative probability threshold")
    kwargs['negative_probability_threshold'] = args.negative_probability_threshold

if args.experiment == 'joint':
    kwargs['approximate'] = approximate

kwargs_list = []
if len(context_generators) > 0:
    for context_generator in context_generators:
        kwargs_list.append(kwargs.copy())
        kwargs_list[-1]['contextGenerator'] = context_generator
else:
    kwargs_list.append(kwargs.copy())

fig, axes = plt.subplots(2,3 if price_discrimination else 1)
color_idx = 0
for learner_class in learner_classes:
    for kwargs in kwargs_list:
        learner_name = list(learner_class_map.keys())[list(learner_class_map.values()).index(learner_class)]
        context_generator_name = '-'+list(context_generator_map.keys())[list(context_generator_map.values()).index(kwargs['contextGenerator'])] if price_discrimination else ''
        exp = experiment(scenario=scenario,
                         learner_class=learner_class,
                         price_discrimination= price_discrimination,
                         fixed_bid=args.fixed_bid,
                         fixed_price=args.fixed_price,
                         n_exp=args.n_exp,
                         **kwargs)
        optimal_arms = exp.run()
        print(learner_name + context_generator_name, optimal_arms)
        exp.print_optimal()
        exp.plot(axes, 
                 color='C'+str(color_idx if color_idx < 3 else color_idx+1), 
                 label= learner_name + context_generator_name)
        color_idx += 1

for idx in range(len(exp.reward_per_experiment)):
    (axes[1, idx] if price_discrimination else axes[1]).hlines(exp.optimal[idx], color='r', xmin=0, xmax=365, linestyle='-', label='Clairvoyant')

(axes[1,-1] if price_discrimination else axes[1]).legend(bbox_to_anchor=(1, 2.225), loc='upper left')
plt.show()