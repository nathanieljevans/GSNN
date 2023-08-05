import os
import pandas as pd
from tbparse import SummaryReader
import argparse
import glob

# set up command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--root", help="Root directory containing tensorboard log files", required=True)
args = parser.parse_args()

reader = SummaryReader(args.root, pivot=True, extra_columns={'dir_name'})
scalars = reader.scalars[['r2_test', 'r2_val', 'r_cell_test', 'r_cell_val', 'r_drug_test', 'r_drug_val', 'r_dose_test', 'r_dose_val', 'mse_test', 'mse_val', 'r_flat_test', 'r_flat_val', 'time_elapsed', 'dir_name']].dropna().drop_duplicates()
hparams = reader.hparams.drop_duplicates()

all_results = hparams.merge(scalars, on='dir_name', how='left')

exp_ids = [] 
for dir_name in all_results.data.values: 
	idd = None 
	for spl in dir_name.split('/'): 
		if spl[:3] == 'exp':
			idd = spl 
	exp_ids.append(idd)

all_results = all_results.assign(EXP_ID = exp_ids) 

# save the DataFrame to a CSV file
all_results.to_csv(args.root + '/results.csv', index=False)

best_res = all_results.sort_values('r2_test', ascending=False).drop_duplicates(subset=['EXP_ID', 'model', 'cell_agnostic', 'randomize', 'gnn'])[['EXP_ID', 'model', 'cell_agnostic', 'randomize', 'gnn', 'r2_test', 'r2_val', 'r_cell_test', 'r_cell_val', 'r_drug_test', 'r_drug_val', 'r_dose_test', 'r_dose_val', 'mse_test', 'mse_val', 'r_flat_test', 'r_flat_val', 'time_elapsed']] 

best_res.to_csv(args.root + '/best_results.csv', index=False)
