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

def load_prism(root, cellspace=None, drugspace=None, avg_replicates=True): 

    cellinfo, dep2iname, iname2dep = load_cellinfo(root=root)

    prism_primary = pd.read_csv(f'{root}/primary-screen-replicate-collapsed-logfold-change.csv')
    prism_primary = prism_primary.rename({'Unnamed: 0':'depmap_id'}, axis=1)

    prism_primary = prism_primary.assign(cell_iname = lambda x: [dep2iname[xx] if xx in dep2iname else None for xx in x.depmap_id])

    prism_primary = prism_primary.set_index(['depmap_id', 'cell_iname']).stack().reset_index().rename({'level_2':'meta', 0:'log_fold_change'}, axis=1)
    prism_primary[['pert_id_long', 'pert_dose','_']] = prism_primary.meta.str.split(pat='::', n=2, expand=True)
    prism_primary = prism_primary.assign(pert_id = lambda x: [xx[:13] for xx in x.pert_id_long])

    prism_primary = prism_primary.assign(screen_id = 'primary')

    prism_secondary = pd.read_csv(f'{root}/secondary-screen-replicate-collapsed-logfold-change.csv')
    prism_secondary = prism_secondary.rename({'Unnamed: 0':'depmap_id'}, axis=1)

    prism_secondary = prism_secondary.assign(cell_iname = lambda x: [dep2iname[xx] if xx in dep2iname else None for xx in x.depmap_id])

    prism_secondary = prism_secondary.set_index(['depmap_id', 'cell_iname']).stack().reset_index().rename({'level_2':'meta', 0:'log_fold_change'}, axis=1)
    prism_secondary[['pert_id_long', 'pert_dose','_']] = prism_secondary.meta.str.split(pat='::', n=2, expand=True)
    prism_secondary = prism_secondary.assign(pert_id = lambda x: [xx[:13] for xx in x.pert_id_long])

    prism_secondary = prism_secondary.assign(screen_id = 'secondary')

    prism = pd.concat((prism_primary, prism_secondary), axis=0)
    prism = prism[lambda x: ~x.cell_iname.isna()]

    # filter to drugspace + cellspace 
    if cellspace is not None: prism = prism[lambda x: x.cell_iname.isin(cellspace)]
    if drugspace is not None: prism = prism[lambda x: x.pert_id.isin(drugspace)]

    if avg_replicates: 
        prism = prism.groupby(['pert_id', 'depmap_id', 'cell_iname', 'pert_dose']).agg({'log_fold_change':np.mean, 'screen_id':list}).reset_index()
        prism = prism.assign(num_repl = lambda x: [len(xx) for xx in x.screen_id])

    # create aggregate id 
    prism['sig_id'] = prism[['cell_iname', 'pert_id', 'pert_dose']].agg('::'.join, axis=1)

    # add cell viability transformation
    # Calculate viability data as two to the power of replicate-level logfold change data
    prism = prism.assign(cell_viab = lambda x: 2**(x.log_fold_change))

    prism.pert_dose = prism.pert_dose.astype(float)

    return prism
