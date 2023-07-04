
import pandas as pd 
import numpy as np 
from src.proc import utils


def load_expr(path, extpath, zscore=False, clip_val=10): 

    expr = pd.read_csv(f'{path}/ccle_expression.txt', sep=',')
    gene_names = [x.split(' ')[0] for x in expr.columns[1:]]
    expr.columns = ['DepMap_ID'] + gene_names

    dep2iname = utils.get_dep2iname(path)

    expr = expr.merge(dep2iname, on='DepMap_ID', how='inner')

    expr = expr[['cell_iname'] + gene_names]

    if zscore: 
        _mean = np.nanmean(expr[expr.columns[1:]].values, axis=0) 
        _std = np.nanstd(expr[expr.columns[1:]].values, axis=0) + 1e-8

    # fill na - mean 
    expr = expr.fillna(np.nanmean(expr[expr.columns[1:]].values))

    # clip for good measure 
    expr[expr.columns[1:]] = np.clip(expr[expr.columns[1:]], 0, clip_val)

    # z-score 
    if zscore: 
        expr[expr.columns[1:]] = (expr[expr.columns[1:]] - _mean)/_std


    expr = expr.set_index('cell_iname').unstack().reset_index().rename({0:'expr', 'level_0':'gene_symbol'}, axis=1)

    # gene symbol -> uniprot 
    ccle_sym2uni = pd.read_csv(f'{extpath}/omnipath_uniprot2genesymb.tsv', sep='\t').rename({'To':'gene_symbol', 'From':'uniprot'}, axis=1)

    expr = expr.merge(ccle_sym2uni, on='gene_symbol')[['cell_iname', 'uniprot', 'expr']]

    # agg any duplicate mappings
    expr = expr.drop_duplicates().groupby(['cell_iname', 'uniprot']).mean().reset_index()

    expr = expr.pivot(index='cell_iname', columns='uniprot', values='expr')

    # agg duplicate columns
    expr = expr.T.groupby(level=0).mean().T

    return expr
