import pandas as pd 
import numpy as np 
from src.proc import utils


def load_methyl(path, extpath): 

    methyl = pd.read_csv(f'{path}/ccle_methyl.txt', sep='\t', low_memory=False) 
    ccle2iname = utils.get_ccle2iname(path)
    ccle2iname = ccle2iname.set_index('ccle_name').to_dict()['cell_iname']

    methyl_sel = [True]*5 + [str(x) in ccle2iname for x in methyl.columns[5:]]
    methyl = methyl[methyl.columns[methyl_sel]]
    methyl.columns = list(methyl.columns[:5]) + [ccle2iname[x] for x in methyl.columns[5:]]

    methyl_lines = list(methyl.columns[5:])

    methyl = methyl[['gene_name'] + methyl_lines]

    for line in methyl_lines: 
        methyl[line] = pd.to_numeric(methyl[line], errors='coerce')

    methyl = methyl.groupby('gene_name').mean()

    methyl = methyl.T 

    methyl = methyl.reset_index().rename({'index':'cell_iname'}, axis=1)

    methyl_genes = list(methyl.columns[1:])

    uni2symb = pd.read_csv(f'{extpath}/omnipath_uniprot2genesymb.tsv', sep='\t').set_index('To').to_dict()['From']

    methyl_genes_with_uniprot_id = set(methyl_genes).intersection(set(list(uni2symb.keys())))

    methyl = methyl[['cell_iname'] + list(methyl_genes_with_uniprot_id)]
    methyl = methyl.rename(uni2symb, axis=1)

    methyl = methyl.set_index('cell_iname')

    # impute 
    fill = np.nanmean(methyl.values.ravel())
    methyl.fillna(fill, inplace=True)

    # agg duplicate columns
    methyl = methyl.T.groupby(level=0).mean().T

    return methyl