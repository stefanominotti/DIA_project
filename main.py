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
parser.add_argument('--exp', dest='experiment', choices=['price', 'bid', 'joint'], required=True, help='Choose if maximizing price, bid or both')
parser.add_argument('--scen', dest='scenario', required=True, help='The scenario JSON name located in main/environments/scenarios')
parser.add_argument('--disc', dest='price_discrimination', default='F', choices=['F', 'T'], help='Choose whether performing or not class discrimination for pricing (default F)')
parser.add_argument('--bid', dest='fixed_bid', type=float, help='Value for the fixed bid, only if --exp is set to \'price\'')
parser.add_argument('--price', dest='fixed_price', type=float, help='Value for the fixed price, only if --exp is set to \'bid\'')
parser.add_argument('--ne', dest='n_exp', default=1, type=int, help='Number of iterations of the experiment to perform, the daily results will be averaged (default 1)')
parser.add_argument('--npt', dest='negative_probability_threshold', default=0.2, type=float, help='Reward negative probability threshold under which an arm can\'t be pulled (default 0.2)')
parser.add_argument('--incgen', dest='incremental_generation', default='T', choices=['F', 'T'], help='Choose whether the context generation should be incremental or not (default T), only if --disc is set to \'T\'')
parser.add_argument('--genrate', dest='generation_rate', type=int, help='Frequency (in days) for context generation, only if --disc is set to \'T\'')
parser.add_argument('--conf', dest='confidence', default=0.05, type=float, help='Confidence for Hoeffding lower bound in context generation (default 0.05), only if --disc is set to \'T\'')
parser.add_argument('--approx', dest='approximate', default='T', choices=['F', 'T'], help='Choose whether the joint algorithm should be approximated or not (default \'T\'), only if --exp is set to \'joint\'')
parser.add_argument('--learners', dest='learner_class', choices=['UCB', 'TS', 'GTS', 'GPTS'], required=True, nargs='+', help='Select the learners to use, \'UCB\' and \'TS\' can be selected only if --exp is set to \'price\'; \'GTS\' and \'GPTS\' can be selectedn only if --exp is set to \'bid\' or \'joint\'')
parser.add_argument('--cgens', dest='contextGenerator', choices=['G', 'BF'], nargs='+', help='Choose the type of context generation between Greedy and Brute-force or both, only if --disc is set to \'T\'')
parser.add_argument('--save', dest='save', choices=['T', 'F'], required=True, help='Choose whether to save or not the results as JSON file')
args = parser.parse_args()

# Transform boolean variables from string to bool
price_discrimination = True if args.price_discrimination == 'T' else False
incremental_generation = True if args.incremental_generation == 'T' else False
approximate = True if args.approximate == 'T' else False
save = True if args.save == 'T' else False

# Check and load scenario
if not args.scenario + '.json' in os.listdir('main/environment/scenarios'):
    raise Exception("Scenario doesn't exists")
scenario = Scenario(args.scenario)

# Check if fixed bid and price are in scenario
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
        print(learner_name + context_generator_name)
        exp = experiment(scenario=scenario,
                         learner_class=learner_class,
                         price_discrimination= price_discrimination,
                         fixed_bid=args.fixed_bid,
                         fixed_price=args.fixed_price,
                         n_exp=args.n_exp,
                         **kwargs)
        optimal_arms = exp.run()
        incgen_string = '-inc' if (price_discrimination and incremental_generation) else ''
        approx_string = '-approx' if (approximate and args.experiment == 'joint') else ''
        if save:
            exp.save_results(args.experiment + '-' + learner_name + context_generator_name + incgen_string + approx_string, optimal_arms)
        print(optimal_arms)
        exp.print_optimal()
        exp.plot(axes, 
                 color='C'+str(color_idx if color_idx < 3 else color_idx+1), 
                 label= learner_name + context_generator_name)
        color_idx += 1

for idx in range(len(exp.reward_per_experiment)):
    (axes[1, idx] if price_discrimination else axes[1]).hlines(exp.optimal[idx], color='r', xmin=0, xmax=365, linestyle='-', label='Clairvoyant')

(axes[1,-1] if price_discrimination else axes[1]).legend(bbox_to_anchor=(1, 2.225), loc='upper left')
plt.show()