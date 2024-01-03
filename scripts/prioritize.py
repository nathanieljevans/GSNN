'''
drug prioritization

(gsnn) $ python prioritize.py --proc ../output/exp1/proc/ --uid_dir ../output/exp1/FOLD-1/GSNN/1e808caf-2cb9-46b3-830d-c3f106f5ef6f/ --goals_path ../output/disease_prioritization_goals.csv --model model-40.pt --verbose



The priortization goals are specified by the `args.goals_path`, which will be in the format `/path/to/goal/dir/<goal_name>.csv

cell_iname | priortization_goal_name_1 | priortization_goal_name_2 |priortization_goal_name_3 |
    A      |          target           |           ...             |        background        |
    B      |          target           |           ...             |         target           |
    C      |          none             |           ...             |         target           |
   ...     |         ...               |           ...             |            ...           |
    Z      |          background       |           ...             |         none             |

where `cell_iname` should contain all the cell lines contained in `data.cellspace`. All other values must be either "target", "background" or "none".
Each column (after cell_iname) will be treated as an independent priortization goal and a drug priortization result will be generated and named `prioritization_goal_name_X.csv`. 

Results will be saved in: 

`args.uid_dir`/ 
    |-> prioritizations/
        |-> <goal_name>
            |-> priortization_goal_name_1.csv
            |-> priortization_goal_name_2.csv
            |-> priortization_goal_name_3.csv

'''
import argparse 
import pandas as pd 
import torch
import numpy as np
import os 
import shutil 
import sys 
sys.path.append('../')
from src.models.GSNN import GSNN
from src.models.NN import NN
from src.prioritize.utils import *

def get_args(): 
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, default='../../data/',
                        help="path to raw data directory")
    
    parser.add_argument("--proc", type=str,
                        help="path to raw data directory")
    
    parser.add_argument("--uid_dir", type=str,
                        help="path to uid directory. This is also treated as the output directory.")
    
    parser.add_argument("--drugs_path", type=str, default=None,
                        help="path to a comma separated list of drugs that should be screened. If None then all drugs will be tested.")
    
    parser.add_argument("--model", type=str, default='model-100.pt',
                        help="model name")
    
    parser.add_argument("--doses", nargs='+', default=[1e-1], type=float,
                        help='')
    
    parser.add_argument("--expr_batch", type=int, default=1024,
                        help="batch side to use when predicting exprssion via the GSNN model")
    
    parser.add_argument("--N", type=int, default=1000, 
                        help='The number of Monte Carlo simulations to run when computing drug priortization metrics.')
    
    parser.add_argument("--verbose", action='store_true',
                        help='Print script progress updates to console.')
    
    parser.add_argument("--combo", action='store_true',
                        help='Whether to screen 2-drug combinations. If False, will only screen single agents.')
    
    parser.add_argument("--goals_path", type=str,
                        help="path to a data frame that characterizes the prioritization goal(s). First column must be `cell_iname` and all following columns will be independed priortization goals. First column should contain all cell lines in `data.cellspace` and all other columns should be either 'target', 'background' or 'none' ")
    
    args = parser.parse_args()

    return args


if __name__ == '__main__': 

    args = get_args() 
    if args.verbose: print(args)
    if args.verbose: print()

    if args.verbose: print('loading models and data...')
    data = torch.load(f'{args.uid_dir}/Data.pt')
    expr_model = torch.load(f'{args.uid_dir}/{args.model}')
    viab_ensemble = torch.load(f'{args.uid_dir}/ViabNNEnsemble.pt')
    siginfo = pd.read_csv(f'{args.data}/siginfo_beta.txt', sep='\t', low_memory=False)
    cellinfo = pd.read_csv(f'{args.data}/cellinfo_beta.txt', sep='\t')[lambda x: x.cell_iname.isin(data.cellspace)]
    druginfo = pd.read_csv(f'{args.data}/compoundinfo_beta.txt', sep='\t')[lambda x: x.pert_id.isin([x.split('__')[1] for x in data.node_names if x.split('__')[0] == 'DRUG'])]
    drug_annot = pd.read_csv(f'{args.data}/repurposing_drugs_20180907.txt', sep='\t', encoding = "ISO-8859-1", skiprows=9)


    # get "target" and "background" lines
    if args.verbose: print('loading priortization goals...')
    goals_df = pd.read_csv(args.goals_path)
    assert 'cell_iname' in goals_df.columns, '`goals_path` csv does not have a `cell_iname` column'
    assert len(set(goals_df.cell_iname.unique().tolist()) - set(list(data.cellspace))) == 0, 'the `cell_inames` in `goals_path` csv does not match `data.cellspace`'

    # get drug + cell line x inputs 
    if args.verbose: print('making drug inputs...')
    # args, data, drugs, cells, doses
    cells_to_screen = goals_df.cell_iname.unique()
    if args.drugs_path is not None: 
        drugs_to_screen = open(args.drugs_path, 'r').read().split(',')
    else: 
        drugs_to_screen = [x.split('__')[-1] for x in data.node_names if 'DRUG__' in x]
        
    if args.verbose: print('# of drugs to screen:', len(drugs_to_screen))
    if args.verbose: print('# of cells to screen:', len(cells_to_screen))
    X, meta = make_drug_inputs(args=args, data=data, doses=args.doses, cells=cells_to_screen, drugs=drugs_to_screen, siginfo=siginfo)

    # use the GSNN model to predict perturbed expression 
    pred_expr = predict_expr(expr_model, X, data, device='cuda', batch=args.expr_batch, verbose=args.verbose)

    # generate output file structure
    goals_file_name = args.goals_path.split('/')[-1][:-4]
    if args.combo: goals_file_name += '_combo'
    if not os.path.exists(f'{args.uid_dir}/prioritizations/'): 
        os.mkdir(f'{args.uid_dir}/prioritizations/')
    if os.path.exists(f'{args.uid_dir}/prioritizations/{goals_file_name}/'): 
        if args.verbose: print(f'`priortizations/{goals_file_name}/` directory exists, deleting contents.')
        shutil.rmtree(f'{args.uid_dir}/prioritizations/{goals_file_name}/')
    os.mkdir(f'{args.uid_dir}/prioritizations/{goals_file_name}/')

    if args.verbose: print('generating prioritization results...')
    N_goals = goals_df.shape[1] - 1
    for i in range(1, goals_df.shape[1]):
        goal_name = goals_df.columns[i]
        print(f'\tprogress: {i}/{N_goals} [{goal_name}]')

        target_lines = goals_df[lambda x: x[goal_name] == 'target'].cell_iname.values
        background_lines = goals_df[lambda x: x[goal_name] == 'background'].cell_iname.values

        res = evaluate_drugs(viab_model     = viab_ensemble, 
                             pred_expr      = pred_expr, 
                             meta           = meta, 
                             res_lines      = background_lines, 
                             sens_lines     = target_lines, 
                             N              = args.N, 
                             verbose        = args.verbose)
        
        res = res.sort_values(by='p_sens', ascending=False)
        
        res.to_csv(f'{args.uid_dir}/prioritizations/{goals_file_name}/{goal_name}.csv', sep=',', index=False)
        if args.verbose: print()

    if args.verbose: print()
    if args.verbose: print('prioritizations complete.')
