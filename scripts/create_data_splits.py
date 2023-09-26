

import argparse 
import numpy  as np 
import pandas as pd
import torch 
import os 

import sys 
sys.path.append('../')
from src.proc.data_split import create_data_splits, keys2sids
from src.proc import utils

def get_args(): 
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, default='./proc/',
                        help="path to raw data directory")
    
    parser.add_argument("--proc", type=str, default='./proc/',
                        help="path to processed data directory")
    
    parser.add_argument("--out", type=str, default='../processed_data/',
                        help="path to data directory")
    
    parser.add_argument("--test_prop", type=float, default=0.2,
                        help="proportion of cell lines to hold-out for the test set")
    
    parser.add_argument("--val_prop", type=float, default=0.2,
                        help="proportion of cell lines to hold-out for the validation set")
    
    parser.add_argument("--hold_out", type=str, default='cell-drug',
                        help="how to split train/test/val; 'cell-drug' will split by cell-drug pairs, 'cell' will split by cell lines.")
    
    parser.add_argument("--min_num_drugs_per_cell_line", type=int, default=10,
                        help="test/val cell line inclusion criteria")
    
    parser.add_argument("--min_num_obs_per_cell_line", type=int, default=10,
                        help="test/val cell line inclusion criteria")
    
    parser.add_argument("--min_num_cell_lines_per_drug", type=int, default=3,
                        help="test/val drug inclusion criteria")
    
    parser.add_argument("--min_num_obs_per_drug", type=int, default=3,
                        help="test/val drug inclusion criteria")
    
    return parser.parse_args()

if __name__ == '__main__': 


    args = get_args()

    if not os.path.exists(args.out): 
        os.mkdir(args.out)

    with open(args.out + '/data_split_args.log', 'w') as f: 
        f.write(str(args))

    data = torch.load(args.proc + '/Data.pt')
    sig_ids = np.load(args.proc + '/sig_ids.npy', allow_pickle=True)
    siginfo = pd.read_csv(args.data + '/siginfo_beta.txt', sep='\t', low_memory=False).merge(pd.DataFrame({'sig_id':sig_ids}), on='sig_id', how='right')
    
    ######################################################################
    # Split Train/test/val by cell line
    ######################################################################

    print()
    print('creating LINCS train/test/val splits...')

    # balance by primary disease.
    #cellinfo = pd.read_csv(args.data + '/cellinfo_beta.txt', sep='\t', low_memory=False)
    #disease = pd.DataFrame({'cell_iname':data.cellspace}).merge(cellinfo, on='cell_iname', how='left').primary_disease.values
    #[train_cells, test_cells, val_cells] = utils.split_ids_by_attribute(list(data.cellspace), disease, [1 - (args.test_prop + args.val_prop), args.test_prop, args.val_prop])
    
    if args.hold_out == 'cell': 
        print('\tspliting train/test/val sets by unique cell line.')
        test_ixs = np.random.choice(np.arange(len(data.cellspace)), size=int(len(data.cellspace)*args.test_prop))
        test_mask = np.zeros((len(data.cellspace),), dtype=bool)
        test_mask[test_ixs] = True

        val_ixs = np.random.choice(np.arange(len(data.cellspace))[~test_mask], size=int(len(data.cellspace)*args.val_prop))
        val_mask = np.zeros((len(data.cellspace),), dtype=bool)
        val_mask[val_ixs] = True

        train_mask = ~(test_mask | val_mask)
        train_ixs = train_mask.nonzero()

        train_cells = data.cellspace[train_ixs]
        test_cells = data.cellspace[test_ixs]
        val_cells = data.cellspace[val_ixs]

        train_obs = siginfo[lambda x: x.cell_iname.isin(train_cells)].sig_id.values
        test_obs = siginfo[lambda x: x.cell_iname.isin(test_cells)].sig_id.values
        val_obs = siginfo[lambda x: x.cell_iname.isin(val_cells)].sig_id.values

        # check if there are any drugs that ONLY appearn in test or val. 
        test_val_drugs = set(data.drugspace) - set(siginfo[lambda x: x.cell_iname.isin(train_cells)].pert_id.unique().tolist())

        if len(test_val_drugs) > 0: 
            print(f'\tWARNING: There are {len(test_val_drugs)} drugs that only appear in test/val sets; removing these observations from test/val; NOTE: drug nodes will remain.')
            test_obs = siginfo[lambda x: x.cell_iname.isin(test_cells) & (~x.pert_id.isin(test_val_drugs))].sig_id.values
            val_obs = siginfo[lambda x: x.cell_iname.isin(val_cells) & (~x.pert_id.isin(test_val_drugs))].sig_id.values

        print(f'\t# train cell lines (# obs): {len(train_cells)} ({len(train_obs)})')
        print(f'\t# test cell lines (# obs): {len(test_cells)} ({len(test_obs)})')
        print(f'\t# val cell lines (# obs): {len(val_cells)} ({len(val_obs)})')

        assert len(set(list(train_cells)).intersection(set(list(test_cells)))) == 0, 'train/test set share cell lines'
        assert len(set(list(train_cells)).intersection(set(list(val_cells)))) == 0, 'train/val set share cell lines'
        assert len(set(list(test_cells)).intersection(set(list(val_cells)))) == 0, 'val/test set share cell lines'

        np.save(f'{args.out}/val_cells', val_cells)
        np.save(f'{args.out}/test_cells', test_cells)
        np.save(f'{args.out}/train_cells', train_cells)

    elif args.hold_out == 'cell-drug': 
        print('\tspliting train/test/val sets by unique (cell, drug) pairs.')
        train_keys, test_keys, val_keys, train_obs, test_obs, val_obs = create_data_splits(siginfo, 
                                                                                           TEST_SET_P = args.test_prop, 
                                                                                           VAL_SET_P = args.val_prop, 
                                                                                           TEST_SET_MIN_NUM_DRUGS_PER_CELL_LINE = args.min_num_drugs_per_cell_line, 
                                                                                           TEST_SET_MIN_NUM_OBS_PER_CELL_LINE = args.min_num_obs_per_cell_line, 
                                                                                           TEST_SET_MIN_NUM_CELL_LINES_PER_DRUG = args.min_num_cell_lines_per_drug, 
                                                                                           TEST_SET_MIN_NUM_OBS_PER_DRUG = args.min_num_obs_per_drug, 
                                                                                           _check_unique = True)
        
        print(f'\t# train keys (# obs): {len(train_keys)} ({len(train_obs)})')
        print(f'\t# test keys (# obs): {len(test_keys)} ({len(test_obs)})')
        print(f'\t# val keys (# obs): {len(val_keys)} ({len(val_obs)})')

        np.save(f'{args.out}/val_keys', val_keys)
        np.save(f'{args.out}/test_keys', test_keys)
        np.save(f'{args.out}/train_keys', train_keys)
        
    else:
        ValueError('unrecognized `hold_out` argument; options: cell-drug, cell')

    assert len(set(list(train_obs)).intersection(set(list(test_obs)))) == 0, 'train/test set share observations'
    assert len(set(list(train_obs)).intersection(set(list(val_obs)))) == 0, 'train/val set share observations'
    assert len(set(list(test_obs)).intersection(set(list(val_obs)))) == 0, 'val/test set share observations'

    print('\tsaving data...')
    np.save(f'{args.out}/lincs_val_obs', val_obs)
    np.save(f'{args.out}/lincs_test_obs', test_obs)
    np.save(f'{args.out}/lincs_train_obs', train_obs)

    # create prism splits 
    print()
    print('creating PRISM train/test/val splits...')
    prism = utils.load_prism(args.data, cellspace=data.cellspace, drugspace=data.drugspace)
    prism_ids = np.load(args.proc + '/prism_ids.npy', allow_pickle=True)
    prism = prism.merge(pd.DataFrame({'sig_id':prism_ids}), on='sig_id', how='right')
    
    if args.hold_out == 'cell': 
        train_obs2 = prism[lambda x: x.cell_iname.isin(train_cells)].sig_id.values
        test_obs2 = prism[lambda x: x.cell_iname.isin(test_cells)].sig_id.values
        val_obs2 = prism[lambda x: x.cell_iname.isin(val_cells)].sig_id.values
    elif args.hold_out == 'cell-drug': 
        test_obs2 = keys2sids(test_keys, prism)
        val_obs2 = keys2sids(val_keys, prism)

        train_obs2 = prism[lambda x: x.sig_id.isin(test_obs2) | x.sig_id.isin(val_obs2)]
    else: 
        raise ValueError('unrecognized `hold_out` argument.')

    np.save(f'{args.out}/prism_val_obs', val_obs2)
    np.save(f'{args.out}/prism_test_obs', test_obs2)
    np.save(f'{args.out}/prism_train_obs', train_obs2)

    print('prism train/test/val splits:')
    print(f'\ttrain: {len(train_obs2)}')
    print(f'\tval: {len(val_obs2)}')
    print(f'\ttest: {len(test_obs2)}')