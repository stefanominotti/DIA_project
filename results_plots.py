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
parser.add_argument('-s', dest='scenario', choices=list(map(lambda x: x.replace('.json', ''), 
                                                    os.listdir('main/environment/scenarios'))), required=True)
parser.add_argument('-e', dest='exp_names', choices=list(map(lambda x: x.replace('.json', ''), 
                                                    os.listdir('results/'))), required=True, nargs='+')
parser.add_argument('-d', dest='price_discrimination', choices=['F', 'T'], required=True)
args = parser.parse_args()
price_discrimination = True if args.price_discrimination == 'T' else False
scen = Scenario(args.scenario)

if not os.path.exists('img'):
    os.mkdir('img')

dpi = 90
if price_discrimination:
    for class_idx, customer_class in enumerate(scen.customer_classes):
        fig, axes = plt.subplots(2, figsize=(1920/dpi, 894/dpi))
        fig.subplots_adjust(top=0.960, bottom=0.055, right=0.880, left=0.050, hspace=0.2, wspace=0.2)
        color_idx = 0
        for exp_name in args.exp_names:
            with open('results/'+exp_name+'.json', 'r') as f:
                results = json.load(f)
            optimal = results['optimal']
            reward_per_experiment = results['rewards']
            class_name = ('' if customer_class.feature_values[0] else 'not ') + customer_class.feature_labels[0] + \
                         (' & ' if customer_class.feature_values[1] else ' & not ') + customer_class.feature_labels[1]
            axes[0].plot(np.cumsum(np.mean(np.subtract(optimal[class_idx], reward_per_experiment[class_idx]), axis=0)), color='C'+str(color_idx), label='-'.join(exp_name.split('-')[1:3]))
            axes[1].plot(np.mean(reward_per_experiment[class_idx], axis=0), color='C'+str(color_idx), label='-'.join(exp_name.split('-')[1:3]))
            axes[0].set_title(class_name)
            axes[0].set_ylabel('Regret')   
            axes[1].set_xlabel('Days') 
            axes[1].set_ylabel('Daily reward')
            color_idx += 1 if color_idx != 2 else 2
            axes[0].margins(0.01, 0.1)
            axes[1].margins(0.01, 0.1)
        axes[1].hlines(optimal[class_idx], color='r', xmin=0, xmax=365, linestyle='-', label='Clairvoyant')
        axes[1].legend(bbox_to_anchor=(1, 2.225), loc='upper left')
        exp_string = exp_name.split('-')[0] +\
             ('-' + exp_name.split('-')[-1] if exp_name.split('-')[-1] == 'inc' or exp_name.split('-')[-1] == 'approx' else '')
        plt.savefig('img/' + exp_string + '-context-' + class_name.replace(' ', '-') + '.png', dpi=dpi)


else:
    fig, axes = plt.subplots(2, figsize=(1920/dpi, 894/dpi))
    fig.subplots_adjust(top=0.960, bottom=0.055, right=0.880, left=0.050, hspace=0.2, wspace=0.2)
    color_idx = 0
    for exp_name in args.exp_names:
        with open('results/'+exp_name+'.json', 'r') as f:
            results = json.load(f)
        optimal = results['optimal']
        reward_per_experiment = results['rewards']
        axes[0].plot(np.cumsum(np.mean(np.subtract(optimal[0], reward_per_experiment[0]), axis=0)), color='C'+str(color_idx), label='-'.join(exp_name.split('-')[1:3]))
        axes[1].plot(np.mean(reward_per_experiment[0], axis=0), color='C'+str(color_idx), label='-'.join(exp_name.split('-')[1:3]))
        axes[0].set_ylabel('Regret')   
        axes[1].set_xlabel('Days') 
        axes[1].set_ylabel('Daily reward')
        color_idx += 1 if color_idx != 2 else 2
        axes[0].margins(0.01, 0.1)
        axes[1].margins(0.01, 0.1)
    axes[1].hlines(optimal[0], color='r', xmin=0, xmax=365, linestyle='-', label='Clairvoyant')
    axes[1].legend(bbox_to_anchor=(1, 2.225), loc='upper left')
    exp_string = exp_name.split('-')[0] + ('-' + exp_name.split('-')[-1] if exp_name.split('-')[-1] == 'approx' else '')
    plt.savefig('img/' + exp_string + '.png', dpi=dpi)