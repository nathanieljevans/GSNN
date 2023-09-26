'''
create train, test, val splits. 

# Create train/test/val data splits

Train/test/val splits will be done such that each split will have unique (drug, cell_line) keys. 

See `config.py` for train/test/split proportions and relevant parameters. 

Test/Val -set criteria: 
- each drug must have at least `config.TEST_SET_MIN_NUM_OBS_PER_DRUG` number of observations
- each drug must have at least `config.TEST_SET_MIN_NUM_CELL_LINES_PER_DRUG` number of cell lines 
- each cell line must have at least `config.TEST_SET_MIN_NUM_DRUGS_PER_CELL_LINE` number of drugs 
- each cell line must have at least `config.TEST_SET_MIN_NUM_OBS_PER_CELL_LINE` number of observations 
- Test/val samples with an APC less than `config.TEST_SET_MIN_APC` will be removed. 

The train/test/val split proportions (see `config.py`) refer to the proportion of (cell_line,drug) pairs 
assigned to each split from candidate drug,line pairs - and will not necessarily reflect the number of 
observations in each split. Having a high value for `config.TEST_SET_MIN_APC` may remove a significant 
proportion of test/val samples. 

'''

import pandas as pd
import numpy as np 
import pickle as pkl

def create_data_splits(siginfo, TEST_SET_P, VAL_SET_P, TEST_SET_MIN_NUM_DRUGS_PER_CELL_LINE, 
                       TEST_SET_MIN_NUM_OBS_PER_CELL_LINE, TEST_SET_MIN_NUM_CELL_LINES_PER_DRUG, 
                       TEST_SET_MIN_NUM_OBS_PER_DRUG, _check_unique=True): 
    '''
    create train,test & val splits. 

    For an observation to be included in the test/val sets, it must fulfill the requirements: 
        -> the obs `cell_iname` must have at least `config.TEST_SET_MIN_NUM_DRUGS_PER_CELL_LINE` drugs and `config.TEST_SET_MIN_NUM_OBS_PER_CELL_LINE` observations. 
        -> the obs `pert_id` must have at least `config.TEST_SET_MIN_NUM_CELL_LINES_PER_DRUG` cell lines and `config.TEST_SET_MIN_NUM_OBS_PER_DRUG` observations.
    This precludes drugs and cell lines that are very lowly represented and therefore will probably not test well. 

    Observations are then characterized by their cell-drug key (cell_iname, pert_id), and split into distinct sets so that each set will 
    see novel combinations of cell and drug. e.g., 
        train set -> (cell line A, drug 1), (cell line B, drug 2)
        test set -> (cell line A, drug 2), (cell line B, drug 1)

    training set gets all remaining keys, irrespective of requirements stated above. 
    '''
    
    print('generating data splits...')
    
    cell_drug_counts = siginfo[['cell_iname', 'pert_id']].drop_duplicates().groupby('cell_iname').count()
    cell_obs_counts = siginfo[['cell_iname', 'sig_id']].drop_duplicates().groupby('cell_iname').count()
    cell_candidates = cell_drug_counts.merge(cell_obs_counts, on='cell_iname')[lambda x: (x.pert_id >= TEST_SET_MIN_NUM_DRUGS_PER_CELL_LINE) & (x.sig_id >= TEST_SET_MIN_NUM_OBS_PER_CELL_LINE)].index.values
    
    print('\t# of cell candidates:', len(cell_candidates))

    drug_cell_counts = siginfo[['cell_iname', 'pert_id']].drop_duplicates().groupby('pert_id').count()
    drug_obs_counts = siginfo[['pert_id', 'sig_id']].drop_duplicates().groupby('pert_id').count()
    drug_candidates = drug_cell_counts.merge(drug_obs_counts, on='pert_id')[lambda x: (x.cell_iname >= TEST_SET_MIN_NUM_CELL_LINES_PER_DRUG) & (x.sig_id >= TEST_SET_MIN_NUM_OBS_PER_DRUG)].index.values
    
    print('\t# of drug candidates:', len(drug_candidates))

    # drug/cell candidates are those that fulfill the requirements for test set .. e.g., the right number of obs, lines, drugs, etc 

    candidate_pairs = siginfo[['cell_iname', 'pert_id']].drop_duplicates()[lambda x: (x.pert_id.isin(drug_candidates)) & (x.cell_iname.isin(cell_candidates))].reset_index(drop=True)

    pair_ixs = candidate_pairs.index.values
    npairs = len(pair_ixs)
    test_ixs = np.random.choice(pair_ixs, size=int(npairs*TEST_SET_P), replace=False)
    pair_ixs =  np.delete(pair_ixs, test_ixs)
    val_ixs = np.random.choice(pair_ixs, size=int(npairs*VAL_SET_P), replace=False)

    test_pairs = candidate_pairs.iloc[test_ixs, :]
    val_pairs = candidate_pairs.iloc[val_ixs, :]

    _train_pairs = siginfo[['cell_iname', 'pert_id']].drop_duplicates().reset_index(drop=True).assign(ix=lambda x: x.index.values)
    test_val_pair_ixs = np.concatenate((_train_pairs.merge(test_pairs, on=['cell_iname', 'pert_id'], how='inner')['ix'].values, _train_pairs.merge(val_pairs, on=['cell_iname', 'pert_id'], how='inner')['ix'].values), axis=0)
    train_ixs = np.delete(_train_pairs.ix.values, test_val_pair_ixs)
    train_pairs = _train_pairs.iloc[train_ixs, :]

    test_keys = [(row.pert_id, row.cell_iname) for i,row in test_pairs.iterrows()]
    val_keys = [(row.pert_id, row.cell_iname) for i,row in val_pairs.iterrows()]
    train_keys = [(row.pert_id, row.cell_iname) for i,row in train_pairs.iterrows()]

    print('\t# of train keys:', len(train_keys))
    print('\t# of test keys:', len(test_keys))
    print('\t# of val keys:', len(val_keys))

    # save keys 
    #pkl.dump({'train':train_keys, 'test':test_keys, 'val':val_keys}, open(f'{config.OUT_DIR}/{uid}/dataset_key_splits.pkl', 'wb'))

    print('\tkey->sig_id [train]...', end='')
    train_obs = keys2sids(train_keys, siginfo.reset_index())
    print('done.')
    print('\tkey->sig_id [test]...', end='')
    test_obs = keys2sids(test_keys, siginfo.reset_index())
    print('done.')
    print('\tkey->sig_id [val]...', end='')
    val_obs = keys2sids(val_keys, siginfo.reset_index())
    print('done.')
    print('\t# of train obs.:', len(train_obs))
    print('\t# of test obs.:', len(test_obs))
    print('\t# of val obs.:', len(val_obs))

    train_obs = np.array(train_obs).astype(str)
    test_obs = np.array(test_obs).astype(str)
    val_obs = np.array(val_obs).astype(str)

    if _check_unique:
        print('running checks...')
        assert len(set(train_obs.tolist()).intersection(set(test_obs.tolist()))) == 0, 'train set indices are in test set indices'
        print('\t1', end='\r')
        assert len(set(test_obs.tolist()).intersection(set(val_obs.tolist()))) == 0, 'test set indices are in val set indices'
        print('\t2', end='\r')
        assert len(set(train_obs.tolist()).intersection(set(val_obs.tolist()))) == 0, 'train set indices are in bal set indices'
        print('\t3', end='\r')

    return train_keys, test_keys, val_keys, train_obs, test_obs, val_obs


def keys2sids(keys, instinfo): 
    '''
    convert (drug, cell_line) keys into obs `sig_id`'s. Assume sig_info is identical to that used for data processing, e.g., index matches obs file names. 
    '''

    sids = pd.DataFrame({'pert_id':[p for p,c in keys], 'cell_iname':[c for p,c in keys]}).merge(instinfo[['sig_id', 'cell_iname', 'pert_id']], on=['pert_id', 'cell_iname'], how='left').sig_id.unique().tolist()

    '''
    # NOTE: old method, bit slow 
    sids = []
    for i,(d,l) in enumerate(keys): 
        print(f'\t\tprogress: {100*i/len(keys):.2f} %', end='\r')
        sids += [sid for sid in instinfo[lambda x: (x.pert_id == d) & (x.cell_iname == l)].sig_id.values]
    '''
    return sids