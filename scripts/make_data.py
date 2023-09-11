import pandas as pd 
import omnipath as op
import numpy as np
import networkx as nx 
import torch_geometric as pyg
import torch
from matplotlib import pyplot as plt 
import shutil
import h5py
import os 
import numpy as np
import argparse 


import sys 
sys.path.append('../')
from src.proc import utils
from src.proc.utils import filter_to_common_cellspace, impute_missing_gene_ids
from src.proc.load_methyl import load_methyl
from src.proc.load_expr import load_expr
from src.proc.load_cnv import load_cnv
from src.proc.load_mut import load_mut
from src.proc.load_prism import load_prism


def get_args(): 
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, default='../../data/',
                        help="path to data directory")
    
    parser.add_argument("--out", type=str, default='../processed_data/',
                        help="path to data directory")
    
    parser.add_argument("--exempl_sig_only", action='store_true',
                        help="whether to filter to only include 'exemplary signatures'")
    
    parser.add_argument('--feature_space', nargs='+', default=['landmark'],
                        help='lincs feature space [landmark, best-inferred, inferred]')
    
    parser.add_argument("--stitch_targets", action='store_true',
                        help="whether to include drug-targets from the STITCH database'")
    
    parser.add_argument("--full_grn", action='store_true',
                        help="whether to augment protein-space with downstream GRN transcription factors'")
    
    parser.add_argument("--min_drug_score", type=int, default=999,
                        help="STITCH drug-target confidence score required for inclusion [0,1000]")
     
    parser.add_argument("--targetome_targets", action='store_true',
                        help="whether to include drug-targets from The Cancer Targetome (Blucher 2017) database'")
    
    parser.add_argument("--min_prop_downstream_lincs_per_drug", type=float, default=0.5,
                        help="proportion of LINCS/output nodes that have a path from drug (e.g., predictable)")

    parser.add_argument("--time", type=float, default=24.,
                        help="the time point to predict expression changes for")
    
    parser.add_argument("--min_obs_per_drug", type=int, default=5,
                        help="drug inclusion criteria, must have at least X observations")
    
    parser.add_argument("--min_obs_per_cell", type=int, default=10,
                        help="cell line inclusion criteria, must have at least X observations")

    parser.add_argument("--test_prop", type=float, default=0.15,
                        help="proportion of cell lines to hold-out for the test set")
    
    parser.add_argument("--val_prop", type=float, default=0.15,
                        help="proportion of cell lines to hold-out for the validation set")
    
    parser.add_argument('--pathways', nargs='+', default=['R-HSA-9006934'],
                        help='Reactome pathway identifiers, gene-set will be used to create sub-graph; pass ["none"] to include full graph')
    
    parser.add_argument('--dose_trans_eps', type=float, default=1e-6, help='dose (uM) transformation to (log10(x + eps) - log10(eps))/-log10(eps). Pass -666 to use legacy "pseudo-count" style transformation.')
    
    return parser.parse_args()

def row2obs(row, dataset, dataset_row, uni2dataset_rowidx, sigid2idx, data, node2idx, meta, omics, eps): 
    ''' 
    Args: 
        row         (pandas.Series)             obs data 
        dataset     (h5py Dataset object)       contains lincs level 5 expression signatures 
        N           (int)                       # of nodes in graph (input + output + function) 

    Returns: 
        dict                                    one observation data 
    '''
    #A = time.time() 

    expr, methyl, mut, cnv = omics

    lincs_nodes, drug_nodes, expr_nodes, cnv_nodes, methyl_nodes, mut_nodes, N, input_nodes, output_nodes, func_nodes = meta

    obs = {}

    obs['pert_type']    = row.pert_type
    obs['conc_um']      = row.pert_dose
    obs['time_hr']      = row.pert_time 
    obs['sig_id']       = row.sig_id 
    obs['cell_iname']   = row.cell_iname
    obs['pert_id']      = row.pert_id 
    obs['y_idx']        = sigid2idx[row.sig_id]

    #B = time.time() 

    # only input nodes should be nonzero 
    # shape: (num_nodes, 1)
    x_dict = {**{n:[0.] for n in func_nodes}, **{n:[0.] for n in drug_nodes}, **{n:[0.] for n in lincs_nodes}}

    # DRUG CONCENTATION TRANSFORMATION 
    # DEPRECATED ("pseudo-count" style transformation): x_dict['DRUG__' + obs['pert_id']] = [np.log10(obs['conc_um'] + 1)]
    if eps == -666: 
        # for legacy reasons, we'll keep an option for the pseudo-count style transformation    
        x_dict['DRUG__' + obs['pert_id']] = [np.log10(obs['conc_um'] + 1)]
    else:
        # This transformation is linear in logspace between [np.log10(eps), inf] AND 0 uM = 0, 1 uM ~ 1. 
        # Recommended that `eps` should be at least a few orders of magnitude smaller than the smallest concentration in the dataset  
        x_dict['DRUG__' + obs['pert_id']] = [(np.log10(obs['conc_um'] + eps) - np.log10(eps))/(-np.log10(eps))]

    # add expr inputs 
    for node in expr_nodes: 
        val = expr.loc[obs['cell_iname'], node.split('__')[1]]
        x_dict[node] = [val]

    # methyl inputs 
    for node in methyl_nodes: 
        val = methyl.loc[obs['cell_iname'], node.split('__')[1]]
        x_dict[node] = [val]

    # cnv inputs 
    for node in cnv_nodes: 
        val = cnv.loc[obs['cell_iname'], node.split('__')[1]]
        x_dict[node] = [val]

    # mut inputs 
    for node in mut_nodes: 
        val = mut.loc[obs['cell_iname'], node.split('__')[1]]
        x_dict[node] = [val]
    
    x_df = pd.DataFrame(x_dict)
    obs['x'] = torch.tensor(x_df[data.node_names].values, dtype=torch.float32).reshape(-1,1)


    # only output nodes should be nonzero 
    # shape: (num_nodes, 1)
    y_dict = {**{n:[0.] for n in func_nodes}, 
              **{n:[0.] for n in drug_nodes}, 
              **{n:[0.] for n in expr_nodes}, 
              **{n:[0.] for n in mut_nodes}, 
              **{n:[0.] for n in methyl_nodes}, 
              **{n:[0.] for n in cnv_nodes}}
    # torch.tensor(dataset[obs['y_idx'], :][geneid_idxs], dtype=torch.float16) 

    y_lincs = dataset[obs['y_idx'], :]

    for lincs_node in lincs_nodes: 
        uniprot = lincs_node.split('__')[1]
        dataset_rowidx = uni2dataset_rowidx[uniprot]
        
        y_dict[lincs_node] = float(y_lincs[dataset_rowidx])
        
    y_df = pd.DataFrame(y_dict)
    #obs['y_dataframe'] = y_df
    obs['y'] = torch.tensor(y_df[data.node_names].values, dtype=torch.float32).reshape(-1,1)

    if (~torch.isfinite(obs['y'])).any(): 
        print(f'WARNING: non-finite value in lincs expr signature; converting non-finite value to 0 [sig_id={row.sig_id }]')
        obs['y'][~torch.isfinite(obs['y'])] = 0.
        obs['y'] = obs['y'].detach()

    #D = time.time() 

    #print(B-A,C-B,D-C)#; 3/0
        
    return obs 


def downstream_nodes(G, node, N, visited=None):
    # If visited is None, initialize it as an empty set
    if visited is None:
        visited = set()

    # Add the current node to the visited set
    visited.add(node)

    downstream = []
    if N > 0:
        for n in G.successors(node):
            if n not in visited:
                # Add this node and all its downstream nodes to the list
                downstream.append(n)
                downstream.extend(downstream_nodes(G, n, N-1, visited))

    return downstream


if __name__ == '__main__': 

    args = get_args()

    args.feature_space = [x.replace('-', ' ') for x in args.feature_space]
    print('feature space', args.feature_space)

    print('loading omnipath data...')
    dorothea = op.interactions.Dorothea().get()
    omnipath = op.interactions.OmniPath().get()
    pathways_extra = op.interactions.PathwayExtra().get()
    tf_mirna = op.interactions.TFmiRNA().get()
    mirna = op.interactions.miRNA().get()

    num_edges = sum([x.shape[0] for x in [dorothea, omnipath, pathways_extra, tf_mirna, mirna]])
    print('\ttotal # of edges:', num_edges)


    if 'none' != args.pathways[0]: 
        print('filtering omnipath nodes based on reactome pathways...')
        print(f'\tpathway id(s): {args.pathways}')

        # get pathway uniprot symbols 
        uni2rea = pd.read_csv(f'{args.data}/UniProt2Reactome_All_Levels.txt', sep='\t', header=None, low_memory=False, dtype=str)
        uni2rea.columns = ['uniprot', 'pathway', 'source', 'description', 'acc', 'species']
        uni2rea = uni2rea[lambda x: ~x.pathway.isna()]
        uni2rea.pathway = [str(p) for p in uni2rea.pathway]
        uni2rea = uni2rea[lambda x: (x.pathway.isin(args.pathways)) & (x.species == 'Homo sapiens')]

        include_uniprot = uni2rea.uniprot.unique()
        print('\tpathway(s) size (un-augmented):', len(include_uniprot))

        if args.full_grn: 
            print('\tincluding full GRN (augmenting protein-space with downstream TFs)...')
            include_uniprot = set(include_uniprot)
            # Add proteins that are involved in the GRN 
            # e.g., TF(in) -> mRNA -> TF(out) -> ...
            # or    TF(in) -> miRNA -> mRNA -> TF(out) -> ...
            # step 1: for tfs in pathway, identify all downstream TFs (via mRNA or miRNA)
            tfs_all = set(dorothea.source.unique())
            tfs_in = tfs_all.intersection(set(include_uniprot))
            GRN = nx.DiGraph()
            for src,dst in dorothea[['source', 'target']].values: 
                GRN.add_edge(src,dst)
            for src,dst in tf_mirna[['source', 'target']].values: 
                GRN.add_edge(src,dst)
            for src,dst in mirna[['source', 'target']].values: 
                GRN.add_edge(src,dst)
            for tf in tfs_in: 
                ds = set(downstream_nodes(GRN, tf, 10, None))
                ds_tfs = ds.intersection(tfs_all)
                include_uniprot = include_uniprot.union(ds_tfs)

            include_uniprot = np.array(list(include_uniprot))
            print('\t\tpathway(s) size (GRN-augmented):', len(include_uniprot))
        

        # filter `dorothea` proteins (source)
        dorothea = dorothea[lambda x: x.source.isin(include_uniprot)]

        # filter `omnipath` proiteins (source, target)
        omnipath = omnipath[lambda x: x.source.isin(include_uniprot) & x.target.isin(include_uniprot)]

        # filter `pathways_extra` (source, target)
        pathways_extra = pathways_extra[lambda x: x.source.isin(include_uniprot) & x.target.isin(include_uniprot)]

        # filter `tf_mirna` (source)
        tf_mirna = tf_mirna[lambda x: x.source.isin(include_uniprot)]

    print('loading gene identifier mappings...')
    uni2id = pd.read_csv('../extdata/omnipath_uniprot2geneid.tsv', sep='\t').rename({'From':'uniprot', 'To':'gene_id'}, axis=1)
    uni2symb = pd.read_csv('../extdata/omnipath_uniprot2genesymb.tsv', sep='\t').rename({'From':'uniprot', 'To':'gene_symbol'}, axis=1)
    gene_map = uni2id.merge(uni2symb, on='uniprot', how='outer')
    geneinfo = pd.read_csv(f'{args.data}/geneinfo_beta.txt', sep='\t')[lambda x: x.feature_space.isin(args.feature_space)].merge(gene_map, on=['gene_id'], how='left')

    # lincs genes with uniport ids 
    __LINCS_GENES_UNIPROT__ = geneinfo.uniprot.dropna().unique()

    lincs_uni2id = {row.uniprot: row.gene_id for i,row in geneinfo[['uniprot', 'gene_id']].dropna().iterrows()}
    lincs_id2uni = {row.gene_id:row.uniprot for i,row in geneinfo[['uniprot', 'gene_id']].dropna().iterrows()}

    print('loading LINCS observation metadata...')
    siginfo = pd.read_csv(f'{args.data}/siginfo_beta.txt', sep='\t', low_memory=False)

    siginfo = siginfo[lambda x: (x.pert_type.isin(['trt_cp'])) 
                        & (x.qc_pass == 1.)                         # passes QC
                        & (np.isfinite(x.pert_time.values))         # ensure time is finite
                        & (x.pert_time.values > 0)                  # some values are -666 
                        & (x.pert_id != None)
                        & (x.cell_iname != None)
                        & (np.isfinite(x.pert_dose.values))
                        & (x.pert_time == args.time)]

    if args.exempl_sig_only: siginfo = siginfo[lambda x: x.is_exemplar_sig == 1]



    print('creating protein- and rna-spaces...')

    # all TF dataset `target`s. 
    # all rna nodes included must have at least one regulatory edge from a protein in our protein space or from a MIRNA 
    #                       # TF targets                    TF targets (miRNA)               
    rna_space = np.unique(dorothea.target.values.tolist() + tf_mirna.target.values.tolist()).tolist()
    ntfrna = len(rna_space)
    print('\t# TF regulated RNAs:', ntfrna)

    # add RNAs that are in the miRNA regulatory network 
    # e.g., must have miRNA edge from rna_space -> new_rna
    for i in range(3): 
        new_rnas = list(set(mirna[lambda x: (x.source.isin(rna_space))].target.tolist()) - set(rna_space))
        rna_space += new_rnas 
        print(i, '\t# added rnas', len(new_rnas))

    rna_space = np.unique(rna_space)
    print('\t# miRNA regulated RNAs:', len(rna_space) - ntfrna)
    _mirnas = mirna[lambda x: (x.source.isin(rna_space))].source.unique()
    print('\t# of miRNA regulators:', len(_mirnas))

    # TF `source` and PPI `source` + `target`
    protein_space   = np.unique(dorothea.source.values.tolist() 
                            + omnipath.source.values.tolist() + omnipath.target.values.tolist() 
                            + pathways_extra.source.values.tolist() + pathways_extra.target.values.tolist()
                            + tf_mirna.source.values.tolist())
    
    _tfs = np.unique(tf_mirna.source.values.tolist() + dorothea.source.values.tolist())
    print('\t# of TFs:', len(_tfs))

    print('\tFinal RNA-space size:', len(rna_space))
    print('\tFinal PROTEIN-space size:', len(protein_space))

    rna_in_prot = set(rna_space.tolist()).intersection(set(protein_space.tolist()))

    print('\toverlap between rna-prot', len(rna_in_prot))

    # NOTE: not all proteins have gene/rna nodes - but that's okay bc it means we have no transcriptional regulation information 
    prots_missing_rna = set(protein_space.tolist()) - set(rna_space.tolist())
    print('\t# proteins without rna/gene nodes:', len(prots_missing_rna))

    # NOTE: not all rna molecules will have a protein product 
    rna_without_prot = set(rna_space.tolist()) - set(protein_space.tolist())
    print('\t# rna molecules without protein products:', len(rna_without_prot))


    print('creating lincs-space...')
    lincs_in_rna_space = set(rna_space.tolist()).intersection(set(__LINCS_GENES_UNIPROT__.tolist()))

    print('\t# lincs (best inferred + landmark) genes in rna-space:', len(lincs_in_rna_space))
    print(f'\tproportion of all rna nodes with LINCS coverage (feature-space: {args.feature_space}):', len(lincs_in_rna_space)/len(rna_space))

    # are any of the genes non-uniquely mapped? 
    # NOTE: these mappings will be dropped and only one unique mapping used. 
    lincs2uniprot = geneinfo[lambda x: x.uniprot.isin(lincs_in_rna_space)].set_index('gene_id')['uniprot'].drop_duplicates().to_dict()

    lincs_space_gene_ids = np.array(list(lincs2uniprot.keys()))

    lincs_space = np.sort(list(lincs2uniprot.values()))
    len(lincs_space)


    print('processing drug targets...')

    druginfo = pd.read_csv(f'{args.data}/compoundinfo_beta.txt', sep='\t')

    clue_drug_target_symbols = druginfo.target.unique()

    gene2uni = gene_map[['uniprot', 'gene_symbol']]

    druginfo = druginfo.merge(gene2uni, left_on='target', right_on='gene_symbol', how='inner') 

    druginfo = druginfo[['pert_id', 'uniprot', 'moa']].rename({'uniprot':'target'}, axis=1)

    druginfo = druginfo.assign(combined_score=1000, source='clue')

    if args.stitch_targets: 
        stitch = pd.read_csv('../extdata/processed_stitch_targets.csv')
        stitch = stitch.assign(source='stitch', moa='')
        druginfo = pd.concat((druginfo, stitch), axis=0)

    if args.targetome_targets: 
        targ = pd.read_csv('../extdata/targetome_with_broad_ids.csv')
        targ = targ.rename({'Target_UniProt':'target'}, axis=1)
        targ = targ.assign(combined_score = 1000, source='targetome', moa='')
        targ = targ[['pert_id','target','combined_score','moa','source']]
        druginfo = pd.concat((druginfo, targ), axis=0)

    druginfo = druginfo.groupby(['pert_id', 'target']).agg({'combined_score' : np.mean, "moa" : lambda x: ','.join(x), 'source':lambda x: '+'.join(x)}).reset_index()

    # filter to protein-space
    druginfo = druginfo[lambda x: x.target.isin(protein_space)]

    # apply score filter 
    druginfo = druginfo[lambda x: x.combined_score >= args.min_drug_score]

    # druginfo.groupby('source').count()[['pert_id']].sort_values('pert_id', ascending=False).head(5)

    drug_space = siginfo[lambda x: x.pert_type == 'trt_cp'].pert_id.unique()

    drug_space_with_targets = druginfo.pert_id.unique()

    num_obs_with_targets = siginfo[lambda x: x.pert_id.isin(drug_space_with_targets)].shape[0]

    print('\t# obs from drugs with targets:', num_obs_with_targets)

    drug_space = drug_space_with_targets

    # filter low cnt drugs 
    sigcnts = siginfo.groupby('pert_id').count()['sig_id'].reset_index().rename({'sig_id':'cnts'}, axis=1)
    sigcnts = sigcnts[lambda x: x.cnts >= args.min_obs_per_drug]

    # remove any drugs that don't have observations in LINCS
    print('\tdrug space before obs-count filter', len(drug_space))
    drug_space = np.sort(list(set(sigcnts.pert_id.values.tolist()).intersection(drug_space)))
    print('\tdrug space after', len(drug_space))

    druginfo2 = druginfo.assign(source                  = lambda x: x.pert_id,
                            is_stimulation          = False, 
                            is_inhibition           = False,
                            is_directed             = True,
                            curation_effort         = 0, 
                            consensus_stimulation   = False, 
                            consensus_inhibition    = False, 
                            n_references            = 0, 
                            n_sources               = 0, 
                            n_primary_sources       = 0,
                            edge_type               = 'drug-target') 

    print('\tbefore filter to protein- and drug-space', druginfo2.shape)
    druginfo2 = druginfo2[lambda x: x.target.isin(protein_space)]
    druginfo2 = druginfo2[lambda x: x.pert_id.isin(drug_space)]
    print('\tafter', druginfo2.shape)

    drug_space = np.sort(druginfo2.pert_id.unique())
    print('\tdrug space size:', len(drug_space))

    print('filtering drugs based on proportion of downstream LINCS nodes...')
    # create a homogenous graph with one edge type to use for infering "upstream" genes 
    all_edges = pd.concat([dorothea, omnipath, pathways_extra, tf_mirna, mirna], axis=0)
    G = nx.DiGraph()
    for i,edge in all_edges.iterrows(): 
        G.add_edge(edge.source, edge.target)

    _drug_res = {'pert_id':[], 'num_downstream_lincs':[]}
    for i, pert_id in enumerate(drug_space):
        print(f'progress {i}/{len(drug_space)}', end='\r')
        targs = druginfo2[lambda x: x.pert_id == pert_id]
        downstream = [] 
        for t in targs.target.values:
            downstream += downstream_nodes(G, t, N=10)
        downstream = np.unique(downstream)
        lincs_downstream = set(downstream).intersection(set(lincs_space))

        _drug_res['pert_id'].append(pert_id)
        _drug_res['num_downstream_lincs'].append(len(lincs_downstream))
    print()

    _drug_res = pd.DataFrame(_drug_res)
    _drug_res = _drug_res.assign(p_downstream_lincs = lambda x: x.num_downstream_lincs / len(lincs_space))

    drug_space = np.sort(_drug_res[lambda x: x.p_downstream_lincs >= args.min_prop_downstream_lincs_per_drug].pert_id.values)
    druginfo2 = druginfo2[lambda x: x.pert_id.isin(drug_space)]
    print('\tfinal drug-space size:', len(drug_space))

    # get valid observation sig_id 
    drug_sig_ids = siginfo[lambda x: x.pert_id.isin(drug_space)].sig_id.unique()

    print('\t# of obs. in drug-space:', len(drug_sig_ids))

    # double check all drugs have at least one target
    assert druginfo2.pert_id.unique().shape[0] == len(drug_space), 'drugspace does not match available target infomation in druginfo2'


    #####################################################################################################################
    #####################################################################################################################
    # OMICS
    #####################################################################################################################
    #####################################################################################################################
    print('creating cell-space and loading omics...')
    siginfo = pd.read_csv(f'{args.data}/siginfo_beta.txt', sep='\t', low_memory=False)

    siginfo2 = siginfo[lambda x: x.sig_id.isin(drug_sig_ids)]
    print('# drugspace obs', siginfo2.shape)

    # get initial cell space defined by having at least `__MIN_OBS_PER_CELL_LINE__` observations per cell line for inclusion
    cell_cnts = siginfo2.groupby('cell_iname').count()[['sig_id']].sort_values('sig_id', ascending=False).reset_index()
    cell_cnts = cell_cnts[lambda x: x.sig_id >= args.min_obs_per_cell]
    cell_space = cell_cnts.cell_iname.unique().astype(str)

    print('\tloading omics...')
    methyl  = load_methyl(path=args.data, extpath='../extdata/') ; print('\t\tmethyl loaded.')
    expr    = load_expr(path=args.data, extpath='../extdata/', zscore=False, clip_val=10) ; print('\t\texpr loaded.')
    cnv     = load_cnv(path=args.data, extpath='../extdata/') ; print('\t\tcnv loaded.')
    mut     = load_mut(path=args.data, extpath='../extdata/') ; print('\t\tmut loaded.')

    [expr, methyl, cnv, mut], cell_space = filter_to_common_cellspace(omics=[expr, methyl, cnv, mut], cell_space=cell_space)

    print('\t\tcell space size:', len(cell_space))

    required_nodes = np.unique(rna_space.tolist() + protein_space.tolist())

    # NOTE: We don't need to include an omic input for every node 
    #       we can only include omic inputs for those nodes that we have information. 
    #       therefore we don't need to impute missing omics

    #expr = impute_missing_gene_ids(omic=expr, gene_space=required_nodes, fill_value=0)
    #cnv = impute_missing_gene_ids(omic=cnv, gene_space=required_nodes, fill_value=0)
    #methyl = impute_missing_gene_ids(omic=methyl, gene_space=required_nodes, fill_value=0)
    #mut = impute_missing_gene_ids(omic=mut, gene_space=required_nodes, fill_value=0)

    # NOTE: Because we don't need to include an omic for every node, we can similarly filter to omics that have good variance
    # Remove the bottom 10% of genes with lowest variance
    expr_genes = list(expr.columns[expr.std(axis=0) >= np.quantile(expr.std(axis=0), q=0.1)])
    methyl_genes = list(methyl.columns[methyl.std(axis=0) >= np.quantile(methyl.std(axis=0), q=0.1)])
    mut_genes = list(mut.columns[mut.std(axis=0) >= np.quantile(mut.std(axis=0), q=0.1)])
    cnv_genes = list(cnv.columns[cnv.std(axis=0) >= np.quantile(cnv.std(axis=0), q=0.1)])

    # only include omics that are in rna space or protein space
    expr_genes = [x for x in expr_genes if (x in protein_space) | (x in rna_space)]
    methyl_genes = [x for x in methyl_genes if (x in protein_space) | (x in rna_space)]
    mut_genes = [x for x in mut_genes if (x in protein_space) | (x in rna_space)]
    cnv_genes = [x for x in cnv_genes if (x in protein_space) | (x in rna_space)]

    expr = expr[expr_genes]
    mut = mut[mut_genes]
    methyl = methyl[methyl_genes]
    cnv = cnv[cnv_genes]

    print('\t# expr genes', len(expr_genes))
    print('\t# mut genes', len(mut_genes))
    print('\t# methyl genes', len(methyl_genes))
    print('\t# cnv genes', len(cnv_genes))
    
    # NOTE: We can also normalize across cell line (within a given omic), since it relative now. 
    # omics are (cell x gene)
    expr = (expr - expr.mean(axis=0)) / (expr.std(axis=0) + 1e-8)
    methyl = (methyl - methyl.mean(axis=0)) / (methyl.std(axis=0) + 1e-8)
    mut = (mut - mut.mean(axis=0)) / (mut.std(axis=0) + 1e-8)
    cnv = (cnv - cnv.mean(axis=0)) / (cnv.std(axis=0) + 1e-8)

    # filter siginfo to cell space 
    siginfo2 = siginfo2[lambda x: x.cell_iname.isin(cell_space)]

    # double check we have obs for each drug 
    if siginfo2.pert_id.unique().shape[0] != len(drug_space): 
        print('\tWARNING: some drugs in drug-space do not have observations (after cell-space filter)')

    #####################################################################################################################
    #####################################################################################################################
    # GRAPH
    #####################################################################################################################
    #####################################################################################################################
    print('creating pytorch-geometric graph/data object...')

    # omnipath.interactions.Dorothea        :: protein (TF) -> gene 
    # omnipath.interactions.OmniPath        :: protein -> protein           :: literature curated and are either activity flow, enzyme-PTM or undirected interaction resources. We also added network databases with high-throughput data. Then we added further directions and effect signs from resources without literature references.
    # omnipath.interactions.PathwayExtra    :: protein -> protein           :: activity flow resources without literature references. However, they are manually curated and many have effect signs.
    # omnipath.interactions.TFmiRNA         :: protein (TF) -> gene         :: Transcriptional regulation of miRNA (“tf_mirna”) from 2 literature curated resources.
    # omnipath.interactions.miRNA           :: RNA -> RNA                   :: contains miRNA-mRNA interactions. 

    doro = dorothea[lambda x: x.source.isin(protein_space) & x.target.isin(rna_space)].assign(source = lambda x: ['PROTEIN__' + y for y in x.source],
                        target = lambda x: ['RNA__' + y for y in x.target], 
                        edge_type = 'dorothea',
                        input_edge = False, 
                        output_edge = False)[['source', 'target', 'edge_type', 'input_edge', 'output_edge']]

    omni = omnipath[lambda x: x.source.isin(protein_space) & x.target.isin(protein_space)].assign(source = lambda x: ['PROTEIN__' + y for y in x.source],
                        target = lambda x: ['PROTEIN__' + y for y in x.target], 
                        edge_type = 'omnipath',
                        input_edge = False, 
                        output_edge = False)[['source', 'target', 'edge_type', 'input_edge', 'output_edge']]

    path = pathways_extra[lambda x: x.source.isin(protein_space) & x.target.isin(protein_space)].assign(source = lambda x: ['PROTEIN__' + y for y in x.source],
                        target = lambda x: ['PROTEIN__' + y for y in x.target], 
                        edge_type = 'pathways_extra',
                        input_edge = False, 
                        output_edge = False)[['source', 'target', 'edge_type', 'input_edge', 'output_edge']]     

    tfmirna = tf_mirna[lambda x: x.source.isin(protein_space) & x.target.isin(rna_space)].assign(source = lambda x: ['PROTEIN__' + y for y in x.source],
                        target = lambda x: ['RNA__' + y for y in x.target], 
                        edge_type = 'tf_mirna',
                        input_edge = False, 
                        output_edge = False)[['source', 'target', 'edge_type', 'input_edge', 'output_edge']]

    mirna_ = mirna[lambda x: x.source.isin(rna_space) & x.target.isin(rna_space)].assign(source = lambda x: ['RNA__' + y for y in x.source],
                        target = lambda x: ['RNA__' + y for y in x.target], 
                        edge_type = 'mirna',
                        input_edge = False, 
                        output_edge = False)[['source', 'target', 'edge_type', 'input_edge', 'output_edge']]  

    drug = druginfo2[lambda x: x.source.isin(drug_space) & x.target.isin(protein_space)].assign(source = lambda x: ['DRUG__' + y for y in x.source],
                        target = lambda x: ['PROTEIN__' + y for y in x.target], 
                        edge_type = 'drug_input',
                        input_edge = True, 
                        output_edge = False)[['source', 'target', 'edge_type', 'input_edge', 'output_edge']]          

    lincs = pd.DataFrame({'source':['RNA__' + x for x in lincs_space], 
                        'target':['LINCS__' + x for x in lincs_space], 
                        'edge_type':'lincs', 
                        'input_edge' : False, 
                        'output_edge' : True})

    RNA_PROT_OVERLAP = list(set(rna_space).intersection(set(protein_space)))
    trans = pd.DataFrame({'source': ['RNA__' + x for x in RNA_PROT_OVERLAP],
                        'target': ['PROTEIN__' + x for x in RNA_PROT_OVERLAP],
                        'edge_type':'translation', 
                        'input_edge' : False, 
                        'output_edge' : False})

    # NOTE: 
    # expr omics input values
    expr_ = pd.DataFrame({'source': ['EXPR__' + x for x in expr_genes if x in rna_space] + ['EXPR__' + x for x in expr_genes if x in protein_space],
                        'target': ['RNA__' + x for x in expr_genes if x in rna_space] + ['PROTEIN__' + x for x in expr_genes if x in protein_space],
                        'edge_type':'expr_input', 
                        'input_edge' : True, 
                        'output_edge' : False})
    
    # methylation
    methyl_ = pd.DataFrame({'source': ['METHYL__' + x for x in methyl_genes if x in rna_space] + ['METHYL__' + x for x in methyl_genes if x in protein_space],
                        'target': ['RNA__' + x for x in methyl_genes if x in rna_space] + ['PROTEIN__' + x for x in methyl_genes if x in protein_space],
                        'edge_type':'methyl_input', 
                        'input_edge' : True, 
                        'output_edge' : False})
    
    # mut
    mut_ = pd.DataFrame({'source': ['MUT__' + x for x in mut_genes if x in rna_space] + ['MUT__' + x for x in mut_genes if x in protein_space],
                        'target': ['RNA__' + x for x in mut_genes if x in rna_space] + ['PROTEIN__' + x for x in mut_genes if x in protein_space],
                        'edge_type':'mut_input', 
                        'input_edge' : True, 
                        'output_edge' : False})
    
    # cnv
    cnv_ = pd.DataFrame({'source': ['CNV__' + x for x in cnv_genes if x in rna_space] + ['CNV__' + x for x in cnv_genes if x in protein_space],
                        'target': ['RNA__' + x for x in cnv_genes if x in rna_space] + ['PROTEIN__' + x for x in cnv_genes if x in protein_space],
                        'edge_type':'cnv_input', 
                        'input_edge' : True, 
                        'output_edge' : False})

    edgelist = pd.concat([doro, omni, path, tfmirna, mirna_, drug, lincs, trans, expr_, methyl_, cnv_, mut_], axis=0)

    node_space = np.sort(np.unique(edgelist.source.values.tolist() + edgelist.target.values.tolist()))
    node2idx = {n:i for i,n in enumerate(node_space)}

    print('\tTotal # of nodes (before stub filter):', len(node_space))
    print('\tTotal # of edges (before stub filter):', len(edgelist))


    # create nx DiGraph and remove any nodes that don't have a LINCS gene downstream. 
    LINCS_nodes = [x for x in node_space if 'LINCS__' in x]
    print('\t# lincs nodes', len(LINCS_nodes))

    G = nx.DiGraph()
    for i,row in edgelist.iterrows(): 
        G.add_edge(row.source, row.target)

    nodes_to_remove = [] 

    for i,node in enumerate(G.nodes()): 
        print('\t\tprogress', i, end='\r')
        has_lincs_downstream = False

        if node in LINCS_nodes: 
            has_lincs_downstream = True 
            continue 
        
        for lincs in LINCS_nodes: 
            if nx.has_path(G, node, lincs): 
                has_lincs_downstream = True 
                break 

        if not has_lincs_downstream: 
            nodes_to_remove.append(node)
    print()
    print('\t# of nodes to remove (no downstream LINCS nodes)', len(nodes_to_remove))

    edgelist = edgelist[lambda x: (~x.source.isin(nodes_to_remove)) & (~x.target.isin(nodes_to_remove))]

    node_space = np.sort(np.unique(edgelist.source.values.tolist() + edgelist.target.values.tolist()))
    node2idx = {n:i for i,n in enumerate(node_space)}

    print('\tTotal # of nodes:', len(node_space))
    print('\tTotal # of edges:', len(edgelist))


    data = pyg.data.Data()

    data.node_names = node_space

    src = torch.tensor([node2idx[x] for x in edgelist.source.values])
    dst = torch.tensor([node2idx[x] for x in edgelist.target.values])
    data.edge_index =  torch.stack((src,dst), dim=0).type(torch.long)

    data.input_edge_mask = torch.tensor(edgelist.input_edge.values, dtype=torch.bool)
    data.output_edge_mask = torch.tensor(edgelist.output_edge.values, dtype=torch.bool)

    data.input_node_ixs = torch.unique(src[data.input_edge_mask])
    data.output_node_ixs = torch.unique(dst[data.output_edge_mask])

    input_node_mask = torch.zeros((len(node_space)), dtype=torch.bool)
    output_node_mask = torch.zeros((len(node_space)), dtype=torch.bool)
    input_node_mask[data.input_node_ixs] = True
    output_node_mask[data.output_node_ixs] = True
    data.input_node_mask = input_node_mask 
    data.output_node_mask = output_node_mask

    print('\t# input nodes:', data.input_node_mask.sum())
    print('\t# output nodes:', data.output_node_mask.sum())
    print('\t# function nodes:', (~(data.output_node_mask | data.input_node_mask)).sum())



    print('processing and saving observations...')

    hdf_cp            = h5py.File(args.data + '/level5_beta_trt_cp_n720216x12328.gctx')
    dataset_cp        = hdf_cp['0']['DATA']['0']['matrix']
    col_cp            = np.array(hdf_cp['0']['META']['COL']['id'][...].astype('str'))       # lincs sample ids 
    row_cp            = hdf_cp['0']['META']['ROW']['id'][...].astype(int)                   # gene ids 

    uni2id_dict = uni2id[lambda x: x.gene_id.isin(row_cp)].set_index('uniprot').to_dict()['gene_id']
    uni2dataset_rowidx = {}
    row_cp = row_cp.tolist()

    for uniprot in [x.split('__')[1] for x in data.node_names if 'LINCS__' in x]: 
        gene_id = uni2id_dict[uniprot]
        uni2dataset_rowidx[uniprot] = row_cp.index(gene_id)

    if not os.path.exists(f'{args.out}'): 
        os.mkdir(f'{args.out}')

    if os.path.exists(f'{args.out}/obs/'): 
        print('\tdeleting current /obs/ folder...')
        shutil.rmtree(f'{args.out}/obs/')

    os.mkdir(f'{args.out}/obs/')


    sigid2idx_cp    = {sid:i for i,sid in enumerate(col_cp)}

    lincs_nodes = [nn for nn in data.node_names if 'LINCS__' in nn]
    drug_nodes = [nn for nn in data.node_names if ('DRUG__' in nn) & ((nn in data.node_names[data.input_node_mask.view(-1).detach().numpy()]))]
    expr_nodes = [nn for nn in data.node_names if ('EXPR__' in nn) & ((nn in data.node_names[data.input_node_mask.view(-1).detach().numpy()]))]
    cnv_nodes = [nn for nn in data.node_names if ('CNV__' in nn) & ((nn in data.node_names[data.input_node_mask.view(-1).detach().numpy()]))]
    methyl_nodes = [nn for nn in data.node_names if ('METHYL__' in nn) & ((nn in data.node_names[data.input_node_mask.view(-1).detach().numpy()]))]
    mut_nodes = [nn for nn in data.node_names if ('MUT__' in nn) & ((nn in data.node_names[data.input_node_mask.view(-1).detach().numpy()]))]
    #expr_nodes = ['EXPR__' + x for x in expr_genes]
    #mut_nodes = ['MUT__' + x for x in mut_genes]
    #methyl_nodes = ['METHYL__' + x for x in methyl_genes]
    #cnv_nodes = ['CNV__' + x for x in cnv_genes]
    
    N = len(data.node_names)

    input_nodes = data.node_names[data.input_node_mask]
    output_nodes = data.node_names[data.output_node_mask]
    func_nodes =  data.node_names[~(data.output_node_mask | data.input_node_mask)]

    omics = [expr, methyl, mut, cnv]
    meta = [lincs_nodes, drug_nodes, expr_nodes, cnv_nodes, methyl_nodes, mut_nodes, N, input_nodes, output_nodes, func_nodes]

    # create obs
    for i,row in siginfo2.reset_index().iterrows(): 
        if (i%10)==0: 
            print(f'\tprogress: {i}/{len(siginfo2)}', end='\r')

        obs = row2obs(row=row, 
                    dataset=dataset_cp, 
                    dataset_row=row_cp, 
                    uni2dataset_rowidx=uni2dataset_rowidx, 
                    sigid2idx=sigid2idx_cp, 
                    data=data, 
                    node2idx=node2idx,
                    meta=meta,
                    omics=omics,
                    eps=args.dose_trans_eps)

        torch.save(obs, f'{args.out}/obs/{obs["sig_id"]}.pt')

    print()
    print('creating train/test/val splits...')

    # hold-out cell lines 
    data.cellspace = cell_space

    #train_mask = np.ones((len(data.cellspace),), dtype=bool)
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

    train_obs = siginfo2[lambda x: x.cell_iname.isin(train_cells)].sig_id.values
    test_obs = siginfo2[lambda x: x.cell_iname.isin(test_cells)].sig_id.values
    val_obs = siginfo2[lambda x: x.cell_iname.isin(val_cells)].sig_id.values

    print(f'# train cell lines (# obs): {len(train_cells)} ({len(train_obs)})')
    print(f'# test cell lines (# obs): {len(test_cells)} ({len(test_obs)})')
    print(f'# val cell lines (# obs): {len(val_cells)} ({len(val_obs)})')

    assert len(set(list(train_cells)).intersection(set(list(test_cells)))) == 0, 'train/test set share cell lines'
    assert len(set(list(train_cells)).intersection(set(list(val_cells)))) == 0, 'train/val set share cell lines'
    assert len(set(list(test_cells)).intersection(set(list(val_cells)))) == 0, 'val/test set share cell lines'

    assert len(set(list(train_obs)).intersection(set(list(test_obs)))) == 0, 'train/test set share observations'
    assert len(set(list(train_obs)).intersection(set(list(val_obs)))) == 0, 'train/val set share observations'
    assert len(set(list(test_obs)).intersection(set(list(val_obs)))) == 0, 'val/test set share observations'

    print('saving data...')
    np.save(f'{args.out}/val_obs', val_obs)
    np.save(f'{args.out}/test_obs', test_obs)
    np.save(f'{args.out}/train_obs', train_obs)

    np.save(f'{args.out}/val_cells', val_cells)
    np.save(f'{args.out}/test_cells', test_cells)
    np.save(f'{args.out}/train_cells', train_cells)

    # save data obj to disk
    torch.save(data, f'{args.out}/Data.pt')
    print('done.')

    with open(f'{args.out}/make_data_completed_successfully.flag', 'w') as f: 
         f.write(':)')


    
