
import pandas as pd 
import numpy as np 
from src.proc import utils

def VC2HotEnc(x): 

    feat_order = ['damaging','other non-conserving', 'silent', 'other conserving']
    mapping_idx = {k:i for i,k in enumerate(feat_order)}

    y = np.zeros((4,))

    for feat in feat_order: 
        if feat in x: 
            y[mapping_idx[feat]] = 1.

    return list(y)

def load_mut(path, extpath, include_variants=None): 
    '''
    loads CCLE mutation data and returns if a gene has any mutation (0=no, 1 is true.)

    variant types = 'damaging', 'other non-conserving', 'silent', 'other conserving
    '''

    mut = pd.read_csv(f'{path}/ccle_mutation.txt', sep=',', low_memory=False)
    dep2iname = utils.get_dep2iname(path).set_index('DepMap_ID').to_dict()['cell_iname']

    mut = mut.assign(cell_iname=[dep2iname[x] if x in dep2iname else None for x in mut.DepMap_ID.values])
    mut = mut[lambda x: ~x.cell_iname.isna()]

    mut = mut[['Hugo_Symbol', 'cell_iname', 'Variant_annotation']]

    if include_variants is not None: 
        mut = mut[lambda x: x.Variant_annotation.isin(include_variants)]

    uni2symb = pd.read_csv(f'{extpath}/omnipath_uniprot2genesymb.tsv', sep='\t')

    mut = mut.merge(uni2symb, left_on='Hugo_Symbol', right_on='To', how='inner')

    mut = mut.rename({'From':'uniprot'}, axis=1)

    mut = mut[['uniprot', 'cell_iname', 'Variant_annotation']].groupby(['uniprot', 'cell_iname']).agg(lambda x: '::'.join(x))

    mut = mut.reset_index()

    mut = mut.assign(has_mut=1.)

    mut = mut.pivot(index='cell_iname', columns='uniprot', values='has_mut')

    mut = mut.fillna(0)

    return mut