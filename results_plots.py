import os
import json
import argparse
import numpy as np

from matplotlib import rcParams
rcParams['font.family'] = 'DeJavu Serif'
rcParams['font.serif'] = ['Palatino']
rcParams['font.size'] = 16
from matplotlib import pyplot as plt
SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)   # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

from main.environment.Scenario import Scenario

parser = argparse.ArgumentParser(description='DIA project resluts plots') 
scenarios = list(map(lambda x: x.replace('.json', ''), os.listdir('main/environment/scenarios')))
experiments = list(map(lambda x: x.replace('.json', ''), os.listdir('results/')))
parser.add_argument('-s', dest='scenario', default=scenarios[0], choices=scenarios, required=True)
parser.add_argument('-e', dest='exp_names', choices=experiments, required=True, nargs='+')
parser.add_argument('-d', dest='price_discrimination', choices=['F', 'T'], required=True)
parser.add_argument('--save', dest='save', default='F', choices=['T', 'F'], help='Choose whether to save or not the results as JSON file')
args = parser.parse_args()
price_discrimination = True if args.price_discrimination == 'T' else False
save = True if args.save == 'T' else False

scen = Scenario(args.scenario)

if not os.path.exists('img'):
    os.mkdir('img')

dpi = 90
iter_for = scen.customer_classes if price_discrimination else [0]

for class_idx, customer_class in enumerate(iter_for):
    fig, axes = plt.subplots(2, figsize=(1920/dpi, 894/dpi))
    fig.subplots_adjust(top=0.960, bottom=0.055, right=0.880, left=0.050, hspace=0.2, wspace=0.2)
    color_idx = 0
    for exp_name in args.exp_names:
        with open('results/'+exp_name+'.json', 'r') as f:
            results = json.load(f)
        optimal = results['optimal']
        reward_per_experiment = results['rewards']
        class_name = ('-context-' + ('' if customer_class.feature_values[0] else 'not ') + customer_class.feature_labels[0] + \
                     (' & ' if customer_class.feature_values[1] else ' & not ') + customer_class.feature_labels[1])\
                         if price_discrimination else ''

        regret_mean = np.cumsum(np.mean(np.subtract(optimal[class_idx], reward_per_experiment[class_idx]), axis=0))
        axes[0].plot(regret_mean, color='C'+str(color_idx), label='-'.join(exp_name.split('-')[1:3]))
            
        reward_mean = np.mean(reward_per_experiment[class_idx], axis=0)
        reward_std = np.std(reward_per_experiment[class_idx], axis=0)
        axes[1].plot(reward_mean, color='C'+str(color_idx), label='-'.join(exp_name.split('-')[1:3]))
        #axes[1].fill_between(range(scen.rounds_horizon-1), reward_mean-reward_std, reward_mean+reward_std, alpha=.5, fc='b', color='C'+str(color_idx), ec='None', label='95% conf interval')

        axes[0].set_title(class_name)
        axes[0].set_ylabel('Regret')   
        axes[1].set_xlabel('Days') 
        axes[1].set_ylabel('Daily reward')
        axes[0].margins(0.01, 0.1)
        axes[1].margins(0.01, 0.1)
        color_idx += 1 if color_idx != 2 else 2

    axes[1].hlines(optimal[class_idx], color='r', xmin=0, xmax=365, linestyle='-', label='Clairvoyant')
    axes[1].legend(bbox_to_anchor=(1, 2.225), loc='upper left')
    exp_string = exp_name.split('-')[0] +\
         ('-' + exp_name.split('-')[-1] if exp_name.split('-')[-1] == 'inc' or exp_name.split('-')[-1] == 'approx' else '')
    if save:
        plt.savefig('img/' + exp_string + class_name.replace(' ', '-') + '.png', dpi=dpi)

if not save:
    plt.show()