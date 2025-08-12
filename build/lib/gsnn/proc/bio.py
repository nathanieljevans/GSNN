
import omnipath as op 
import pandas as pd 
import numpy as np
from pypath.utils import mapping
import pandas as pd


def get_bio_interactions(undirected=False, include_mirna=False, include_extra=False, dorothea_levels=['A', 'B'], gene_symbol=True, verbose=True): 
    r"""
    Retrieve and standardise directed biological interactions from the
    OmniPath knowledge base suite.

    The function downloads, harmonises and concatenates several curated
    interaction resources that are exposed through the *omnipath* Python
    package and converts them into a single DataFrame with unified node
    identifiers.  Each identifier is prefixed with the molecular entity
    type so that the downstream GSNN pipeline can easily distinguish
    between RNA and protein nodes:

        * ``PROTEIN__<gene_symbol>``
        * ``RNA__<gene_symbol>``

    In addition, an explicit *translation* edge (``RNA → PROTEIN``) is
    created for every gene that is found in both the RNA and the protein
    namespace.

    Parameters
    ----------
    undirected : bool, optional (default=False)
        If ``True``, the graph is made undirected by adding a reverse edge
        for every existing interaction.
    include_mirna : bool, optional (default=False)
        Whether to augment the graph with miRNA-target and TF-miRNA
        interactions.
    include_extra : bool, optional (default=False)
        Whether to include additional kinase and pathway interactions that
        lack direct literature support.
    dorothea_levels : list[str], optional (default=['A', 'B'])
        Confidence levels to retain from the DoRothEA transcription-factor
        regulon resource. Valid levels are ``['A', 'B', 'C', 'D']``.
    gene_symbol : bool, optional (default=True)
        If ``True`` the identifiers are returned as HGNC gene symbols.
        Otherwise uniprot gene identifiers are used.
    verbose : bool, optional (default=True)
        Whether to print progress updates.

    Returns
    -------
    list[str]
        Unique node identifiers in homogeneous ordering.
    pandas.DataFrame
        DataFrame with the columns ``['source', 'target', 'edge_type']``
        describing the directed interaction graph.

    Notes
    -----
    The function prints the number of automatically generated translation
    edges.  Depending on the local cache state, the first call may take a
    few seconds because the interaction tables are lazily downloaded from
    the OmniPath server.

    Examples
    --------
    >>> from gsnn.proc.bio import get_bio_interactions
    >>> nodes, edges = get_bio_interactions(undirected=True, include_mirna=True)
    >>> len(nodes), edges.shape
    """

    if verbose: print('loading omnipath interactions...')

    if verbose: print('\tdorothea...')
    dorothea        = op.interactions.Dorothea().get(organism = 'human', dorothea_levels=dorothea_levels, genesymbol=gene_symbol)

    if verbose: print('\tomnipath...')
    omnipath        = op.interactions.OmniPath().get(organism = 'human', genesymbol=gene_symbol)

    if include_extra:
        if verbose: print('\tpathways_extra...')
        pathways_extra  = op.interactions.PathwayExtra().get(organism = 'human', genesymbol=gene_symbol)

        if verbose: print('\tkinase_extra...')
        kin_extra       = op.interactions.KinaseExtra().get(organism = 'human', genesymbol=gene_symbol)

    if include_mirna: 
        if verbose: print('\tTF-miRNA...')
        tf_mirna        = op.interactions.TFmiRNA().get(organism = 'human', genesymbol=gene_symbol)

        if verbose: print('\tmiRNA...')
        mirna           = op.interactions.miRNA().get(organism = 'human', genesymbol=gene_symbol)

    if gene_symbol: 
        src_name = 'source_genesymbol'
        tgt_name = 'target_genesymbol'
    else:
        src_name = 'source'
        tgt_name = 'target'

    doro = dorothea.assign(source = lambda x: ['PROTEIN__' + y for y in x[src_name]],
                        target = lambda x: ['RNA__' + y for y in x[tgt_name]], 
                        edge_type = 'dorothea')[['source', 'target', 'edge_type']]

    omni = omnipath.assign(source = lambda x: ['PROTEIN__' + y for y in x[src_name]],
                        target = lambda x: ['PROTEIN__' + y for y in x[tgt_name]], 
                        edge_type = 'omnipath')[['source', 'target', 'edge_type']]

    interactions = [doro, omni]
    
    if include_extra:
        # interactions without literature reference 
        path = pathways_extra.assign(source = lambda x: ['PROTEIN__' + y for y in x[src_name]],
                            target = lambda x: ['PROTEIN__' + y for y in x[tgt_name]], 
                            edge_type = 'pathways_extra')[['source', 'target', 'edge_type']]   

        kin = kin_extra.assign(source = lambda x: ['PROTEIN__' + y for y in x[src_name]],
                            target = lambda x: ['PROTEIN__' + y for y in x[tgt_name]], 
                            edge_type = 'kinase_extra')[['source', 'target', 'edge_type']]

        interactions += [path, kin]  

    if include_mirna:
        tfmirna = tf_mirna.assign(source = lambda x: ['PROTEIN__' + y for y in x[src_name]],
                            target = lambda x: ['RNA__' + y for y in x[tgt_name]], 
                            edge_type = 'tf_mirna')[['source', 'target', 'edge_type']]

        mirna_ = mirna.assign(source = lambda x: ['RNA__' + y for y in x[src_name]],
                            target = lambda x: ['RNA__' + y for y in x[tgt_name]], 
                            edge_type = 'mirna')[['source', 'target', 'edge_type']]

        interactions += [tfmirna, mirna_]
    
    # get translation interactions 
    _fdf = pd.concat(interactions, axis=0, ignore_index=True)

    _fnames = _fdf['source'].values.tolist() + _fdf['target'].values.tolist()
    rna_space = [x.split('__')[1] for x in _fnames if x.split('__')[0] == 'RNA']
    protein_space = [x.split('__')[1] for x in _fnames if x.split('__')[0] == 'PROTEIN']
    RNA_PROT_OVERLAP = list(set(rna_space).intersection(set(protein_space)))
    trans = pd.DataFrame({'source': ['RNA__' + x for x in RNA_PROT_OVERLAP],
                        'target': ['PROTEIN__' + x for x in RNA_PROT_OVERLAP],
                        'edge_type':'translation'})

    if verbose: print('# of translation edges (RNA->PROTEIN):', len(trans))

    # combine all edges 
    func_df = pd.concat(interactions + [trans], axis=0, ignore_index=True)

    if undirected:
        print('transforming to undirected (adding reverse edges)')
        # swap the direction of each edge to obtain an undirected graph
        # (doing the swap via column selection avoids potential duplicate column
        # names that can arise with a direct ``rename`` using the same targets)
        func_df2 = func_df[['target', 'source', 'edge_type']].copy()
        func_df2.columns = ['source', 'target', 'edge_type']
        func_df = pd.concat((func_df, func_df2), ignore_index=True, axis=0)
        func_df = func_df.drop_duplicates()
        func_df = func_df.dropna()

    func_names = np.unique(func_df['source'].tolist() + func_df['target'].tolist()).tolist()

    # rename source to src and target to dst
    func_df = func_df.rename(columns={'source': 'src', 'target': 'dst'})

    return func_names, func_df




def uniprot2symbol(uniprot_ids, allow='1:m', drop_na=True):
    r"""Map UniProt accession IDs to HGNC gene symbols using PyPath.

    A convenience wrapper around :pyfunc:`pypath.utils.mapping.map_name` that
    translates protein accessions into their corresponding gene symbols.

    Two mapping strategies are available (`allow`):

        1. ``'1:m'`` - keep **all** gene symbols associated with a UniProt ID
           (one-to-many, default).
        2. ``'1:1'`` - keep only the **first** gene symbol returned by PyPath for
           each UniProt ID (one-to-one).

    Args:
        uniprot_ids (Sequence[str] or pandas.Series): Iterable of UniProt
            accession IDs.  Duplicate IDs are collapsed to the unique set for
            the lookup, but the returned :class:`~pandas.DataFrame` contains one
            row per *combination* of accession and gene symbol.
        allow (str, optional): Mapping strategy; must be either ``'1:m'`` or
            ``'1:1'``.  Defaults to ``'1:m'``.

    Returns:
        pandas.DataFrame: A two-column DataFrame with

            * ``'uniprot_id'`` - UniProt accession (str)
            * ``'gene_symbol'`` - Gene symbol (str) or *None* if the accession
              could not be mapped.

    Example:
        >>> from gsnn.proc.map import uniprot2symbol
        >>> ids = pd.Series(['P38398', 'Q9Y243', 'INVALID'])
        >>> uniprot2symbol(ids, mapping='1:m').head()
           uniprot_id gene_symbol
        0      P38398       MAPK1
        1      Q9Y243        PTEN
        2     INVALID        None
    """

    assert allow in ['1:m', '1:1'], 'allow must be one of ["1:m", "1:1"]'

    u2s  = {'uniprot_id': [], 'gene_symbol': []}

    for u in np.unique(uniprot_ids): 
        
        s = mapping.map_name(u, 'uniprot', 'genesymbol')

        if len(s) > 0: 
            for g in s:  
                u2s['uniprot_id'].append(u)
                u2s['gene_symbol'].append(g)
                if allow == '1:1': break # only map one gene symbol per uniprot id; first in set
        else: 
            u2s['uniprot_id'].append(u)
            u2s['gene_symbol'].append(None)

    u2s = pd.DataFrame(u2s)

    if drop_na: 
        u2s = u2s.dropna()

    return u2s.drop_duplicates()



def symbol2uniprot(gene_symbols, allow='1:m', drop_na=True):
    r"""Map gene symbols to UniProt accession IDs using PyPath.

    A convenience wrapper around :pyfunc:`pypath.utils.mapping.map_name` that
    translates gene symbols into their corresponding UniProt accession IDs.

    Two mapping strategies are available (`allow`):

        1. ``'1:m'`` - keep **all** gene symbols associated with a UniProt ID
           (one-to-many, default).
        2. ``'1:1'`` - keep only the **first** gene symbol returned by PyPath for
           each UniProt ID (one-to-one).

    Args:
        gene_symbols (Sequence[str] or pandas.Series): Iterable of gene symbols.
            Duplicate symbols are collapsed to the unique set for the lookup,
            but the returned :class:`~pandas.DataFrame` contains one row per
            *combination* of symbol and UniProt ID.
        allow (str, optional): Mapping strategy; must be either ``'1:m'`` or
            ``'1:1'``.  Defaults to ``'1:m'``.

    Returns:
        pandas.DataFrame: A two-column DataFrame with

            * ``'gene_symbol'`` - Gene symbol (str)
            * ``'uniprot_id'`` - UniProt accession (str) or *None* if the symbol
              could not be mapped.

    Example:
        >>> from gsnn.proc.map import symbol2uniprot
        >>> symbols = pd.Series(['MAPK1', 'PTEN', 'INVALID'])
        >>> symbol2uniprot(symbols, mapping='1:m').head()
           gene_symbol uniprot_id
        0      MAPK1       P38398
        1      PTEN        Q9Y243
        2     INVALID        None
    """

    assert allow in ['1:m', '1:1'], 'allow must be one of ["1:m", "1:1"]'

    s2u  = {'uniprot_id': [], 'gene_symbol': []}

    for s in np.unique(gene_symbols): 
        
        u = mapping.map_name(s, 'genesymbol', 'uniprot')

        if len(u) > 0: 
            for u_ in u:  
                s2u['uniprot_id'].append(u_)
                s2u['gene_symbol'].append(s)
                if allow == '1:1': break # only map one gene symbol per uniprot id; first in set
        else: 
            s2u['uniprot_id'].append(None)
            s2u['gene_symbol'].append(s)

    s2u = pd.DataFrame(s2u)

    if drop_na: 
        s2u = s2u.dropna()

    return s2u.drop_duplicates()