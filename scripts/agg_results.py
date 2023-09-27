'''
aggregate GSNN results from a given directory and sub-directories. The aggregated `results.csv` file will be saved in root directory.  
NOTE: install `gsnn_tf` conda env from `enviroment2.yml`. 

use: 
(gsnn_tf) $ python agg_results.py --root ../output/
'''

import os
import pandas as pd
from tbparse import SummaryReader
import argparse
import glob


__COLS__ = ['mean_r_val', 'mean_r_test', 'median_r_val', 'median_r_test', 'r2_test', 'r2_val', 'r_cell_test', 'r_cell_val', 'r_drug_test', 'r_drug_val', 'r_dose_test', 'r_dose_val', 'mse_test', 'mse_val', 'r_flat_test', 'r_flat_val', 'time_elapsed', 'eval_at_epoch', 'dir_name']

def get_results(root, exp): 
    reader = SummaryReader(root, pivot=True, extra_columns={'dir_name'})
    scalars = reader.scalars

    # check for missing columns (potentially older version) - add as none if missing 
    for c in __COLS__: 
        if c not in scalars.columns: 
            scalars = scalars.assign(**{c:None})    

    scalars = scalars[__COLS__].drop_duplicates()
    hparams = reader.hparams.drop_duplicates()

    all_results = hparams.merge(scalars, on='dir_name', how='left')

    all_results = all_results.assign(EXP_ID = exp) 
    return all_results

if __name__ == '__main__': 

    # set up command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", help="Root directory containing tensorboard log files", required=True)
    args = parser.parse_args()
    
    res = [] 
    # assume each sub dir of root is an exp file 
    exps = [x for x in os.listdir(args.root) if (os.path.isdir(args.root + '/' + x))] 
    print(exps)
    for i, exp in enumerate(exps):
        print(f'aggregating: {exp} [{i}/{len(exps)}]') 
        for k, fold in enumerate(os.listdir(args.root + '/' + exp)): 
            for j,model in enumerate([x for x in os.listdir(args.root + '/' + exp + '/' + fold) if x in ['GNN','GSNN','NN']]):
                print('\t', fold, model, end='\r')
                try: 
                    res.append(get_results(args.root + '/' + exp + '/' + fold + '/' + model, exp))
                except Exception as e: 
                    print(f'\texperiment failed: {exp}, {fold}, {model}')
                    print('###'*50)
                    print(e) 
                    print('###'*50)
    print('saving results...')
    all_results = pd.concat(res, axis=0)
    # save the DataFrame to a CSV file
    all_results.to_csv(args.root + '/results.csv', index=False)
    print('done.')
