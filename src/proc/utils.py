import numpy as np
import pandas as pd

def get_dep2iname(path):
    clue_info = pd.read_csv(f'{path}/cellinfo_beta.txt', sep='\t')[['cell_iname', 'ccle_name']].dropna()
    ccle_info = pd.read_csv(f'{path}/ccle_info.txt', sep=',')[['DepMap_ID', 'CCLE_Name']].rename({'CCLE_Name':'ccle_name'}, axis=1).dropna()
    dep2iname = clue_info.merge(ccle_info)[['DepMap_ID', 'cell_iname']].drop_duplicates()
    return dep2iname

def get_ccle2iname(path):
    return pd.read_csv(f'{path}/cellinfo_beta.txt', sep='\t')[['cell_iname', 'ccle_name']].dropna()


def load_cellinfo(root, cellspace=None): 
    '''
    combine ccle and lincs cell line info
    input 
        cell_line_space (list): `cell_iname` identifiers of cell lines to return
    output: 
        (dataframe): merged dataframe 
        (dict): dep2iname 
        (dict): iname2dep
    '''
    cinfo = pd.read_csv(f'{root}/ccle_info.txt', low_memory=False)
    linfo = pd.read_csv(f'{root}/cellinfo_beta.txt', sep='\t', low_memory=False)

    cellinfo = cinfo.merge(linfo, left_on='CCLE_Name', right_on='ccle_name', how='inner')
    cellinfo = cellinfo[['DepMap_ID', 'cell_iname', 'sex', 'cell_lineage', 'primary_disease_x', 'subtype', 'growth_pattern', 'doubling_time', 'donor_age', 'primary_or_metastasis']]
    
    if cellspace is not None: cellinfo = cellinfo[lambda x: x.cell_iname.isin(cellspace)]

    dep2iname = cellinfo.set_index('DepMap_ID')['cell_iname'].to_dict() 
    iname2dep = cellinfo.set_index('cell_iname')['DepMap_ID'].to_dict()

    return cellinfo, dep2iname, iname2dep

def filter_to_common_cellspace(omics, cell_space=None): 
    '''
    takes a list of omics dataframes and filters to the common subset of cell_inames. Optionally filter to overlap with `cell_space` if not None. 
    '''
    if cell_space is None: 
        lines = None
    else: 
        lines = set(list(cell_space))

    for omic in omics: 
        if lines == None: 
            lines = set(omic.index.tolist())
        else: 
            lines = lines.intersection(set(omic.index.tolist()))

    cell_space = np.sort(list(lines))

    return [omic[lambda x: x.index.isin(cell_space)] for omic in omics], cell_space

def aggregate_duplicate_genes(omic): 
    '''
    non-unique mapping from gene symbol to uniprot id can result in multiple gene identifiers (columns)
    '''

def impute_missing_gene_ids(omic, gene_space, fill_value): 
    '''
    impute the missing gene identifiers with the given `fill_value`, then select `gene_space`
    '''

    omic_genes = omic.columns.tolist() 
    missing_genes = list( set(list(gene_space)) - set(omic_genes) )
    
    if len(missing_genes) > 0: 
        print('\t# missing genes:', len(missing_genes), f'[impute value: {fill_value}]')
        missing_df = pd.DataFrame({**{'cell_iname':omic.index}, **{g:[fill_value]*omic.shape[0] for g in missing_genes}}).set_index('cell_iname')
        omic = omic.merge(missing_df, left_index=True, right_index=True, how='inner', validate='1:1')

    return omic[list(gene_space)]