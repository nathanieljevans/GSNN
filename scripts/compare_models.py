'''
loads the results from two models (output of `eval.py`) and generate plot comparisons 

example usage: 
    $ python compare_models.py --uid1 ../output/gsnn/09461877-9965-4dd8-a668-97f77ef73717/ --uid2 ../output/nn/4a98c844-1be8-48c8-9ec8-03081cb7d391/ --xlabel GSNN --ylabel NN --verbose
'''

import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt 
import argparse 
import pickle as pkl
import seaborn as sbn 



def get_args(): 
    parser = argparse.ArgumentParser()

    parser.add_argument("--uid1", type=str,
                        help="path to data directory")
    
    parser.add_argument("--uid2", type=str,
                        help="path to data directory")
    
    parser.add_argument("--xlabel", type=str, default='model1',
                        help="plot x-axis label")
    
    parser.add_argument("--ylabel", type=str, default='model2',
                        help="plot y-axis label")
    
    parser.add_argument("--verbose", action='store_true',
                        help="print status to console")
    
    return parser.parse_args()


if __name__ == '__main__': 

    args = get_args()
    if args.verbose: print(args)

    r_gene_dict1 = pkl.load(open(f'{args.uid1}/r_gene_dict.pkl', 'rb'))
    r_cell_gene_dict1 = pkl.load(open(f'{args.uid1}/r_cell_gene_dict.pkl', 'rb'))
    r_cell_dict1 = pkl.load(open(f'{args.uid1}/r_cell_dict.pkl', 'rb'))
    r_drug_dict1 = pkl.load(open(f'{args.uid1}/r_drug_dict.pkl', 'rb'))
    res1 = [r_gene_dict1, r_cell_gene_dict1, r_cell_dict1, r_drug_dict1]

    r_gene_dict2 = pkl.load(open(f'{args.uid2}/r_gene_dict.pkl', 'rb'))
    r_cell_gene_dict2 = pkl.load(open(f'{args.uid2}/r_cell_gene_dict.pkl', 'rb'))
    r_cell_dict2 = pkl.load(open(f'{args.uid2}/r_cell_dict.pkl', 'rb'))
    r_drug_dict2 = pkl.load(open(f'{args.uid2}/r_drug_dict.pkl', 'rb'))
    res2 = [r_gene_dict2, r_cell_gene_dict2, r_cell_dict2, r_drug_dict2]

    res_names = ['within-gene corr.', 'within-gene corr. [drug,dose regressed-out]', 'within-cell corr.', 'within-drug corr.']

    f, axes = plt.subplots(2,2, figsize=(10,10))
    if args.verbose: print('plotting...')
    for d1, d2, name, ax in zip(res1, res2, res_names, axes.flat): 

        x = name+'_1'
        y =  name+'_2'
        df = pd.DataFrame(d1, index=[0]).T.rename({0:x}, axis=1).merge(pd.DataFrame(d2, index=[0]).T.rename({0:y}, axis=1), left_index=True, right_index=True, validate='1:1')
        df = df.assign(M1_adv = lambda l: l[x] >= l[y])
        min_ = min(df[x].values.tolist() + df[y].values.tolist())
        max_ = max(df[x].values.tolist() + df[y].values.tolist())
        ax.plot((min_,max_), (min_, max_), 'k--')
        ax.axvline(0, color='k')
        ax.axhline(0, color='k')
        sbn.scatterplot(x=x, y=y, data=df, ax=ax, hue='M1_adv')

        # center of mass 
        ax.plot(df[x].values.mean(), df[y].values.mean(), 'ks', markersize=10, label=f'CoM [PAM1={df.M1_adv.mean():.2f}]')
        ax.legend()
        ax.set_xlabel(args.xlabel)
        ax.set_ylabel(args.ylabel)
        ax.set_title(name)

    plt.tight_layout()
    if args.verbose: print('saving...')
    plt.savefig(args.uid1 + f'/model_comp__{args.uid1.split("/")[-2]}__{args.uid2.split("/")[-2]}.png')
    plt.savefig(args.uid2 + f'/model_comp__{args.uid1.split("/")[-2]}__{args.uid2.split("/")[-2]}.png')
