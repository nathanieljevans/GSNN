
import pandas as pd 
import numpy as np 
from src.proc import utils


def load_cnv(path, extpath):
    '''
    Get a single cell lines Copy Number Variation (CNV) values. Return genes in order of `genelist`. Genes in `genelist` that are missing in cnv will be assigned a value of 1 (prior to 0-1 normalization).  

    ~1e-3 % of entries are NA -> these values are assigned a value of 1. 

    Clip values between 0,5
        
    inputs
        cnv_path (str): path to `ccle_expression.txt` 
        genelist (list): gene symbols expression values to return
        cell_line_space (list): `cell_iname` to return from
    '''

    cnv = pd.read_csv(f'{path}/ccle_cnv.txt', sep=',', low_memory=False).rename({'Unnamed: 0': 'DepMap_ID'}, axis=1)
    cnv.columns = [x.split(' ')[0] for x in cnv.columns]

    # map depmap id to cell iname 
    dep2iname = utils.get_dep2iname(path).set_index('DepMap_ID').to_dict()['cell_iname']

    cnv = cnv.assign(cell_iname=[dep2iname[x] if x in dep2iname else None for x in cnv.DepMap_ID])
    cnv = cnv[lambda x: ~x.cell_iname.isna()]
    cnv = cnv.set_index('cell_iname').drop('DepMap_ID', axis=1)

    # gene name to uniprot 
    uni2symb = pd.read_csv(f'{extpath}/omnipath_uniprot2genesymb.tsv', sep='\t').set_index('To').to_dict()['From']
    cnv_genes = list(cnv.columns)
    cnv_genes_with_uniprot_id = list(set(cnv_genes).intersection(set(list(uni2symb.keys()))))
    cnv = cnv[cnv_genes_with_uniprot_id].rename(uni2symb, axis=1)

    # impute missing values with value of 1 
    cnv = cnv.fillna(value=1)

    # clip values between 0,7
    cnv = cnv.clip(0,7)

    # agg duplicate columns
    cnv = cnv.T.groupby(level=0).mean().T

    return cnv