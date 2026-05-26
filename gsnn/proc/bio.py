
# NOTE: ``omnipath`` probes ``omnipathdb.org`` at package-import time, and
# ``pypath`` pulls in a similarly heavy graph of resources. Both are only
# needed inside the public helpers below, so we import them lazily (i.e.
# inside the function bodies). That way ``import gsnn.proc.bio`` (and the
# transitive ``import gsnn`` triggered by ``gsnn/__init__.py``) stays cheap
# and -- crucially -- silent on hosts with no egress to omnipathdb.org.
import re

import pandas as pd 
import numpy as np


_UNIPROT_RE = re.compile(
    r"^[OPQ][0-9][A-Z0-9]{3}[0-9]|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$"
)

_PROTEIN_PREFIX = 'PROTEIN__'
_COMPLEX_PREFIX = 'COMPLEX__'
_EDGE_COLS = ('source', 'target', 'edge_type', 'source_uniprot', 'target_uniprot')
_DEDUP_COLS = ('source', 'target', 'edge_type')


def _is_complex_node(node):
    """Return True if ``node`` encodes a multi-protein complex.

    OmniPath represents a protein complex by underscore-concatenating the
    HGNC gene symbols of its members (matching the
    ``components_genesymbols`` field of the OmniPath ``complexes`` endpoint),
    e.g. ``PROTEIN__AEBP2_EED_EZH2_RBBP4_SUZ12`` for PRC2.
    """
    if not isinstance(node, str):
        return False
    if not node.startswith(_PROTEIN_PREFIX):
        return False
    return '_' in node[len(_PROTEIN_PREFIX):]


def _expand_complex_members(node):
    """Return the constituent ``PROTEIN__<member>`` nodes of a complex.

    Non-complex nodes are returned unchanged in a single-element list.
    """
    if _is_complex_node(node):
        members = node[len(_PROTEIN_PREFIX):].split('_')
        return [_PROTEIN_PREFIX + m for m in members]
    return [node]


def _node_uniprot_lookup(df, prefix):
    """Last-seen accession per prefixed node from either endpoint."""
    as_src = df.loc[df['source'].str.startswith(prefix, na=False)].groupby('source')['source_uniprot'].last()
    as_tgt = df.loc[df['target'].str.startswith(prefix, na=False)].groupby('target')['target_uniprot'].last()
    return as_src.combine_first(as_tgt)


def _build_func_nodes_df(func_df):
    """Build per-node metadata from a standardized edge table."""
    nodes = np.unique(func_df['src'].tolist() + func_df['dst'].tolist())
    from_src = func_df.groupby('src', sort=False)['source_uniprot'].last()
    from_dst = func_df.groupby('dst', sort=False)['target_uniprot'].last()
    uniprot = from_src.combine_first(from_dst)
    return pd.DataFrame({
        'func_name': nodes,
        'uniprot': [uniprot.get(n, pd.NA) for n in nodes],
        'gene_symbol': [n.split('__', 1)[1] if '__' in n else n for n in nodes],
    })


def build_uniprot_symbol_map(func_edges):
    """All (uniprot, gene_symbol, func_name) triples observed on any edge endpoint.

    Parameters
    ----------
    func_edges : pd.DataFrame
        The second return value of :func:`get_bio_interactions`. Must contain
        columns ``src``, ``dst``, ``source_uniprot``, ``target_uniprot``.

    Returns
    -------
    pd.DataFrame
        Columns (in order): ``uniprot``, ``gene_symbol``, ``func_name``,
        ``node_kind`` (``'PROTEIN'`` or ``'RNA'``). One row per unique
        ``(uniprot, func_name)`` pair. Many-to-many: a single uniprot may
        appear with multiple func_names, and a single func_name may carry
        multiple uniprots.

    Notes
    -----
    * Excludes synthetic ``COMPLEX:...`` accessions emitted by the OmniPath
      complexes endpoint (they are not real UniProt IDs).
    * Excludes miRBase-style identifiers carried on miRNA edges (anything
      that does not look like a UniProt accession; see ``_UNIPROT_RE``).
    * Pure pandas; no network I/O.
    * The unique uniprots here are a strict superset of the well-formed
      UniProt accessions in ``func_nodes['uniprot']`` (i.e. those matching
      ``_UNIPROT_RE``), because ``func_nodes`` keeps only one accession per
      node via ``.last()`` while this table retains every distinct
      ``(uniprot, func_name)`` pair seen on any edge endpoint.
    """
    cols_src = ['src', 'source_uniprot']
    cols_dst = ['dst', 'target_uniprot']
    src = (
        func_edges.loc[:, cols_src]
        .rename(columns={'src': 'func_name', 'source_uniprot': 'uniprot'})
    )
    dst = (
        func_edges.loc[:, cols_dst]
        .rename(columns={'dst': 'func_name', 'target_uniprot': 'uniprot'})
    )
    pairs = pd.concat([src, dst], ignore_index=True)
    pairs = pairs.dropna(subset=['func_name', 'uniprot'])
    pairs['func_name'] = pairs['func_name'].astype(str)
    pairs['uniprot'] = pairs['uniprot'].astype(str).str.strip()
    is_prot = pairs['func_name'].str.startswith('PROTEIN__')
    is_rna = pairs['func_name'].str.startswith('RNA__')
    pairs = pairs.loc[is_prot | is_rna].copy()
    pairs['node_kind'] = np.where(
        pairs['func_name'].str.startswith('PROTEIN__'), 'PROTEIN', 'RNA',
    )
    looks_uniprot = pairs['uniprot'].str.match(_UNIPROT_RE)
    pairs = pairs.loc[looks_uniprot]
    pairs['gene_symbol'] = pairs['func_name'].str.split('__', n=1).str[1]
    out = (
        pairs[['uniprot', 'gene_symbol', 'func_name', 'node_kind']]
        .drop_duplicates()
        .sort_values(['func_name', 'uniprot'])
        .reset_index(drop=True)
    )
    return out


def _strip_alias_suffix(name):
    """Drop semicolon-separated alias names and keep only the first entry.

    Several miRNA-resource tables exposed through OmniPath carry the
    miRBase legacy "previous IDs" string in the gene-symbol columns, with
    multiple historical names of the same mature miRNA joined by
    ``';'`` (e.g. ``HSA-MIR-675B;HSA-MIR-675*``).  These are aliases of a
    single entity, not distinct molecules, so we collapse them down to
    the first listed name.
    """
    if not isinstance(name, str) or ';' not in name:
        return name
    if '__' in name:
        prefix, _, rest = name.partition('__')
        return f'{prefix}__{rest.split(";")[0]}'
    return name.split(';')[0]


def _filter_curation_quality(df, min_n_references=None, min_curation_effort=None,
                             dataset='', verbose=True):
    """Drop OmniPath edges below literature-support thresholds.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw interaction table returned by ``omnipath``.
    min_n_references : int or None
        Minimum number of PubMed references supporting an edge.
    min_curation_effort : int or None
        Minimum OmniPath curation-effort score for an edge.
    dataset : str
        Label used in verbose progress messages.
    verbose : bool
        Whether to print summary statistics.

    Returns
    -------
    pandas.DataFrame
    """
    if min_n_references is None and min_curation_effort is None:
        return df

    n_before = len(df)
    out = df

    if min_n_references is not None:
        if 'n_references' not in out.columns:
            if verbose:
                print(
                    f'\t{dataset}: no n_references column; dropping all {n_before} edges '
                    f'(min_n_references={min_n_references})'
                )
            return out.iloc[0:0].copy()
        out = out.loc[out['n_references'].fillna(0).ge(min_n_references)]

    if min_curation_effort is not None:
        if 'curation_effort' not in out.columns:
            if verbose:
                print(
                    f'\t{dataset}: no curation_effort column; dropping all {len(out)} edges '
                    f'(min_curation_effort={min_curation_effort})'
                )
            return out.iloc[0:0].copy()
        out = out.loc[out['curation_effort'].fillna(0).ge(min_curation_effort)]

    if verbose and len(out) < n_before:
        print(
            f'\t{dataset}: curation filter kept {len(out)}/{n_before} edges '
            f'(min_n_references={min_n_references}, min_curation_effort={min_curation_effort})'
        )

    return out.reset_index(drop=True)


def _standardize_interactions(df, src_name, tgt_name, src_prefix, tgt_prefix, edge_type,
                              min_n_references=None, min_curation_effort=None,
                              dataset='', verbose=True):
    """Filter, prefix and return a standard GSNN edge dataframe."""
    df = _filter_curation_quality(
        df, min_n_references, min_curation_effort, dataset=dataset, verbose=verbose,
    )
    if df.empty:
        return pd.DataFrame(columns=list(_EDGE_COLS))
    df = df.rename(columns={'source': 'source_uniprot', 'target': 'target_uniprot'})
    if src_name == 'source':
        src_name = 'source_uniprot'
    if tgt_name == 'target':
        tgt_name = 'target_uniprot'
    return df.assign(
        source=lambda x: [src_prefix + y for y in x[src_name]],
        target=lambda x: [tgt_prefix + y for y in x[tgt_name]],
        edge_type=edge_type,
    )[list(_EDGE_COLS)]


def _apply_complex_handling(df, mode, verbose=True):
    """Resolve protein-complex nodes in an interaction dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Edge dataframe with columns ``_EDGE_COLS``.
    mode : str
        One of ``{'none', 'remove', 'expand', 'link'}``:

            * ``'none'``   - leave complex nodes untouched (backwards
              compatible with the legacy behaviour).
            * ``'remove'`` - drop every edge that involves a complex node.
            * ``'expand'`` - replace each complex with its constituent
              members, fanning the original edge out to one edge per
              member (cartesian product when both endpoints are
              complexes).  Self-loops introduced by overlapping
              membership are removed.
            * ``'link'``   - rename complex nodes into the
              ``COMPLEX__`` namespace and add explicit
              ``PROTEIN__<member> -> COMPLEX__<...>`` "assembly" edges
              so the model can learn complex activity from member
              activity.
    verbose : bool, optional
        Whether to print summary statistics.

    Returns
    -------
    pandas.DataFrame
    """
    if mode == 'none':
        return df

    src_is_cplx = df['source'].apply(_is_complex_node)
    dst_is_cplx = df['target'].apply(_is_complex_node)
    has_cplx = src_is_cplx | dst_is_cplx
    n_cplx_edges = int(has_cplx.sum())

    if mode == 'remove':
        if verbose:
            print(f'\tcomplex_handling=remove: dropping {n_cplx_edges} edges that involve a complex')
        return df.loc[~has_cplx].reset_index(drop=True)

    if mode == 'expand':
        keep = df.loc[~has_cplx]
        cplx_rows = df.loc[has_cplx]
        expanded = []
        for _, row in cplx_rows.iterrows():
            srcs = _expand_complex_members(row['source'])
            dsts = _expand_complex_members(row['target'])
            for s in srcs:
                for d in dsts:
                    if s == d:
                        continue
                    expanded.append({
                        'source': s,
                        'target': d,
                        'edge_type': row['edge_type'],
                        'source_uniprot': row['source_uniprot'],
                        'target_uniprot': row['target_uniprot'],
                    })
        expanded_df = pd.DataFrame(expanded, columns=list(_EDGE_COLS))
        out = pd.concat([keep, expanded_df], axis=0, ignore_index=True)
        out = out.drop_duplicates(subset=list(_DEDUP_COLS)).reset_index(drop=True)
        if verbose:
            print(f'\tcomplex_handling=expand: {n_cplx_edges} complex-level edges fanned out into {len(expanded_df)} member-level edges')
        return out

    if mode == 'link':
        # OmniPath complex accession keyed by underscore-joined member symbols
        cplx_acc_by_members = {}
        for _, row in df.iterrows():
            if _is_complex_node(row['source']):
                cplx_acc_by_members[row['source'][len(_PROTEIN_PREFIX):]] = row['source_uniprot']
            if _is_complex_node(row['target']):
                cplx_acc_by_members[row['target'][len(_PROTEIN_PREFIX):]] = row['target_uniprot']

        def _rename(node):
            if _is_complex_node(node):
                return _COMPLEX_PREFIX + node[len(_PROTEIN_PREFIX):]
            return node

        out = df.copy()
        out['source'] = out['source'].apply(_rename)
        out['target'] = out['target'].apply(_rename)
        cplx_nodes = sorted(set(
            out.loc[out['source'].str.startswith(_COMPLEX_PREFIX), 'source'].tolist()
            + out.loc[out['target'].str.startswith(_COMPLEX_PREFIX), 'target'].tolist()
        ))
        protein_uniprot = _node_uniprot_lookup(out, _PROTEIN_PREFIX)
        assembly_edges = []
        for cnode in cplx_nodes:
            member_key = cnode[len(_COMPLEX_PREFIX):]
            cplx_acc = cplx_acc_by_members[member_key]
            members = member_key.split('_')
            accs = cplx_acc[len('COMPLEX:'):].split('_') if isinstance(cplx_acc, str) and cplx_acc.startswith('COMPLEX:') else []
            acc_by_member = dict(zip(members, accs)) if len(members) == len(accs) else {}
            for m in members:
                src_acc = protein_uniprot.get(_PROTEIN_PREFIX + m)
                if src_acc is None:
                    src_acc = acc_by_member.get(m)
                assembly_edges.append({
                    'source': _PROTEIN_PREFIX + m,
                    'target': cnode,
                    'edge_type': 'assembly',
                    'source_uniprot': src_acc,
                    'target_uniprot': cplx_acc,
                })
        asm_df = pd.DataFrame(assembly_edges, columns=list(_EDGE_COLS))
        out = pd.concat([out, asm_df], axis=0, ignore_index=True)
        out = out.drop_duplicates(subset=list(_DEDUP_COLS)).reset_index(drop=True)
        if verbose:
            print(f'\tcomplex_handling=link: {len(cplx_nodes)} complexes linked via {len(asm_df)} PROTEIN->COMPLEX assembly edges')
        return out

    raise ValueError(
        f"complex_handling must be one of ['none', 'remove', 'expand', 'link'], got {mode!r}"
    )


def get_bio_interactions(undirected=False, 
                         include_tf_mirna=False, 
                         include_pathway_extra=False, 
                         include_kinase_extra=False, 
                         include_ligrec_extra=False,
                         include_collecTRI=False,
                         include_dorothea=True,
                         include_omnipath=True,
                         dorothea_levels=['A', 'B'], 
                         gene_symbol=True, 
                         complex_handling='link', 
                         min_n_references=None, 
                         min_curation_effort=None,
                         return_uniprot_map=False,
                         verbose=True): 
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
    include_tf_mirna : bool, optional (default=False)
        Whether to augment the graph with TF-miRNA and miRNA-target
        interactions.
    include_pathway_extra : bool, optional (default=False)
        Whether to include additional pathway interactions that lack direct
        literature support.
    include_kinase_extra : bool, optional (default=False)
        Whether to include additional kinase-substrate interactions that
        lack direct literature support.
    include_ligrec_extra : bool, optional (default=False)
        Whether to include additional ligand-receptor interactions that
        lack direct literature support.
    include_collecTRI : bool, optional (default=False)
        Whether to include CollecTRI transcription-factor regulon
        interactions.
    include_dorothea : bool, optional (default=True)
        Whether to include DoRothEA transcription-factor regulon
        interactions.
    include_omnipath : bool, optional (default=True)
        Whether to include curated OmniPath protein-protein interactions.
    dorothea_levels : list[str], optional (default=['A', 'B'])
        Confidence levels to retain from the DoRothEA transcription-factor
        regulon resource. Valid levels are ``['A', 'B', 'C', 'D']``.
    gene_symbol : bool, optional (default=True)
        If ``True`` the identifiers are returned as HGNC gene symbols.
        Otherwise uniprot gene identifiers are used.
    complex_handling : {'none', 'remove', 'expand', 'link'}, optional
        How to deal with protein-complex entities (OmniPath encodes
        complexes as underscore-concatenated member gene symbols, e.g.
        ``PROTEIN__AEBP2_EED_EZH2_RBBP4_SUZ12`` for PRC2):

            * ``'none'``   - leave complex nodes untouched (legacy
              behaviour, kept for backwards compatibility).
            * ``'remove'`` - drop every edge that involves a complex.
            * ``'expand'`` - replace each complex with its constituent
              members, fanning out one edge per member.  This recovers
              gene-level coverage at the cost of introducing approximate
              member-level edges that were not literally curated.
            * ``'link'``   - rename complex nodes into a dedicated
              ``COMPLEX__`` namespace and add explicit
              ``PROTEIN__<member> -> COMPLEX__<...>`` "assembly" edges,
              so the GSNN can learn complex activity from member activity
              while preserving the unit-level semantics of the curated
              interaction.
    min_n_references : int or None, optional (default=None)
        If set, retain only edges supported by at least this many PubMed
        references (OmniPath ``n_references`` field).  Datasets that do
        not expose the column are dropped entirely when this filter is
        active.
    min_curation_effort : int or None, optional (default=None)
        If set, retain only edges whose OmniPath curation-effort score is
        at least this value.  Datasets that do not expose the column are
        dropped entirely when this filter is active.
    return_uniprot_map : bool, optional (default=False)
        If ``True``, return a third element: the many-to-many UniProt
        mapping table produced by :func:`build_uniprot_symbol_map`.
    verbose : bool, optional (default=True)
        Whether to print progress updates.

    Returns
    -------
    pandas.DataFrame
        One row per function-graph node with columns ``['func_name', 'uniprot',
        'gene_symbol']``.  ``func_name`` is the prefixed node id (e.g.
        ``PROTEIN__TP53``); ``gene_symbol`` is the suffix after ``__``;
        ``uniprot`` is the last-seen OmniPath accession for that node.
    pandas.DataFrame
        DataFrame with columns ``['src', 'dst', 'edge_type', 'source_uniprot',
        'target_uniprot']`` describing the directed interaction graph.
        ``source_uniprot`` and ``target_uniprot`` are the OmniPath ``source`` /
        ``target`` accession columns (typically UniProt; complexes may use
        ``COMPLEX:...``; miRNA resources may use miRBase IDs).  A handful of
        synthetic ``assembly`` edges may lack a member-level ``source_uniprot``.
    pandas.DataFrame, optional
        Returned only when ``return_uniprot_map=True``.  The many-to-many
        UniProt ↔ gene-symbol ↔ func_name mapping table from
        :func:`build_uniprot_symbol_map`.

    Notes
    -----
    The function prints the number of automatically generated translation
    edges.  Depending on the local cache state, the first call may take a
    few seconds because the interaction tables are lazily downloaded from
    the OmniPath server.

    Duplicate edges are collapsed on ``(src, dst, edge_type)`` only; if
    multiple accessions map to the same symbol after alias stripping, the
    first row is kept.

    Several miRNA-related tables surface miRBase legacy alias strings of
    the form ``"HSA-MIR-675B;HSA-MIR-675*"`` -- multiple historical names
    of the *same* mature miRNA joined by ``';'``.  Only the first alias
    is retained as the canonical node identifier.

    Examples
    --------
    >>> from gsnn.proc.bio import get_bio_interactions
    >>> nodes, edges = get_bio_interactions(undirected=True, include_tf_mirna=True)
    >>> nodes.columns.tolist(), edges.shape
    """

    import omnipath as op  # lazy: avoids network probes at gsnn import time

    if verbose: print('loading omnipath interactions...')
    if verbose and (min_n_references is not None or min_curation_effort is not None):
        print(
            f'\tapplying curation filters: min_n_references={min_n_references}, '
            f'min_curation_effort={min_curation_effort}'
        )

    if include_dorothea:
        if verbose: print('\tdorothea...')
        dorothea = op.interactions.Dorothea().get(
            organism='human', dorothea_levels=dorothea_levels, genesymbol=gene_symbol,
        )

    if include_omnipath:
        if verbose: print('\tomnipath...')
        omnipath = op.interactions.OmniPath().get(organism='human', genesymbol=gene_symbol)

    if include_collecTRI:
        if verbose: print('\tcollectri...')
        collectri = op.interactions.CollecTRI().get(organism='human', genesymbol=gene_symbol)

    if include_pathway_extra:
        if verbose: print('\tpathways_extra...')
        pathways_extra = op.interactions.PathwayExtra().get(organism='human', genesymbol=gene_symbol)

    if include_kinase_extra:
        if verbose: print('\tkinase_extra...')
        kin_extra = op.interactions.KinaseExtra().get(organism='human', genesymbol=gene_symbol)

    if include_ligrec_extra:
        if verbose: print('\tligrec_extra...')
        ligrec_extra = op.interactions.LigRecExtra().get(organism='human', genesymbol=gene_symbol)

    if include_tf_mirna:
        if verbose: print('\tTF-miRNA...')
        tf_mirna = op.interactions.TFmiRNA().get(organism='human', genesymbol=gene_symbol)

        if verbose: print('\tmiRNA...')
        mirna = op.interactions.miRNA().get(organism='human', genesymbol=gene_symbol)

    if gene_symbol:
        src_name = 'source_genesymbol'
        tgt_name = 'target_genesymbol'
    else:
        src_name = 'source'
        tgt_name = 'target'

    _std = lambda df, sp, tp, et, name: _standardize_interactions(
        df, src_name, tgt_name, sp, tp, et,
        min_n_references=min_n_references,
        min_curation_effort=min_curation_effort,
        dataset=name,
        verbose=verbose,
    )

    interactions = []

    if include_dorothea:
        interactions.append(_std(dorothea, 'PROTEIN__', 'RNA__', 'dorothea', 'dorothea'))

    if include_omnipath:
        interactions.append(_std(omnipath, 'PROTEIN__', 'PROTEIN__', 'omnipath', 'omnipath'))

    if include_collecTRI:
        interactions.append(_std(collectri, 'PROTEIN__', 'RNA__', 'collectri', 'collectri'))

    if include_pathway_extra:
        interactions.append(_std(pathways_extra, 'PROTEIN__', 'PROTEIN__', 'pathways_extra', 'pathways_extra'))

    if include_kinase_extra:
        interactions.append(_std(kin_extra, 'PROTEIN__', 'PROTEIN__', 'kinase_extra', 'kinase_extra'))

    if include_ligrec_extra:
        interactions.append(_std(ligrec_extra, 'PROTEIN__', 'PROTEIN__', 'ligrec_extra', 'ligrec_extra'))

    if include_tf_mirna:
        interactions.append(_std(tf_mirna, 'PROTEIN__', 'RNA__', 'tf_mirna', 'tf_mirna'))
        interactions.append(_std(mirna, 'RNA__', 'RNA__', 'mirna', 'mirna'))

    if not interactions:
        base_df = pd.DataFrame(columns=list(_EDGE_COLS))
    else:
        base_df = pd.concat(interactions, axis=0, ignore_index=True)

    # Collapse miRBase legacy alias strings (semicolon-separated) down to
    # a single canonical name so e.g. "RNA__HSA-MIR-675B;HSA-MIR-675*" and
    # "RNA__HSA-MIR-675B" are treated as the same node.
    base_df['source'] = base_df['source'].apply(_strip_alias_suffix)
    base_df['target'] = base_df['target'].apply(_strip_alias_suffix)
    base_df = base_df.drop_duplicates(subset=list(_DEDUP_COLS)).reset_index(drop=True)

    base_df = _apply_complex_handling(base_df, complex_handling, verbose=verbose)

    rna_uniprot = _node_uniprot_lookup(base_df, 'RNA__')
    protein_uniprot = _node_uniprot_lookup(base_df, 'PROTEIN__')

    # get translation interactions
    _fnames = base_df['source'].values.tolist() + base_df['target'].values.tolist()
    rna_space = [x.split('__')[1] for x in _fnames if x.split('__')[0] == 'RNA']
    protein_space = [x.split('__')[1] for x in _fnames if x.split('__')[0] == 'PROTEIN']
    RNA_PROT_OVERLAP = list(set(rna_space).intersection(set(protein_space)))
    trans_sources = ['RNA__' + x for x in RNA_PROT_OVERLAP]
    trans_targets = ['PROTEIN__' + x for x in RNA_PROT_OVERLAP]
    trans = pd.DataFrame({
        'source': trans_sources,
        'target': trans_targets,
        'edge_type': 'translation',
        'source_uniprot': rna_uniprot.loc[trans_sources].values,
        'target_uniprot': protein_uniprot.loc[trans_targets].values,
    })

    if verbose: print('# of translation edges (RNA->PROTEIN):', len(trans))

    # combine all edges
    func_df = pd.concat([base_df, trans], axis=0, ignore_index=True)

    if undirected:
        print('transforming to undirected (adding reverse edges)')
        # swap the direction of each edge to obtain an undirected graph
        # (doing the swap via column selection avoids potential duplicate column
        # names that can arise with a direct ``rename`` using the same targets)
        func_df2 = func_df[['target', 'source', 'target_uniprot', 'source_uniprot', 'edge_type']].copy()
        func_df2.columns = list(_EDGE_COLS)
        func_df = pd.concat((func_df, func_df2), ignore_index=True, axis=0)
        func_df = func_df.drop_duplicates()
        func_df = func_df.dropna()

    # rename source to src and target to dst
    func_df = func_df.rename(columns={'source': 'src', 'target': 'dst'})

    func_nodes = _build_func_nodes_df(func_df)

    if return_uniprot_map:
        return func_nodes, func_df, build_uniprot_symbol_map(func_df)
    return func_nodes, func_df




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

    from pypath.utils import mapping  # lazy: avoids network probes at gsnn import time

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

    from pypath.utils import mapping  # lazy: avoids network probes at gsnn import time

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


def ensg2symbol(ensg_ids, allow='1:m', drop_na=True):
    r"""Map Ensembl gene IDs (ENSG) to HGNC gene symbols using PyPath.

    A convenience wrapper around :pyfunc:`pypath.utils.mapping.map_name` that
    translates Ensembl gene IDs into their corresponding HGNC gene symbols.

    Two mapping strategies are available (`allow`):

        1. ``'1:m'`` - keep **all** gene symbols associated with an Ensembl ID
           (one-to-many, default).
        2. ``'1:1'`` - keep only the **first** gene symbol returned by PyPath for
           each Ensembl ID (one-to-one).

    Args:
        ensg_ids (Sequence[str] or pandas.Series): Iterable of Ensembl gene
            IDs. Duplicate IDs are collapsed to the unique set for the lookup,
            but the returned :class:`~pandas.DataFrame` contains one row per
            *combination* of Ensembl ID and gene symbol.
        allow (str, optional): Mapping strategy; must be either ``'1:m'`` or
            ``'1:1'``.  Defaults to ``'1:m'``.
        drop_na (bool, optional): If ``True``, drop rows where the Ensembl ID
            could not be mapped to a gene symbol. Defaults to ``True``.

    Returns:
        pandas.DataFrame: A two-column DataFrame with

            * ``'ensg_id'`` - Ensembl gene ID (str)
            * ``'gene_symbol'`` - HGNC gene symbol (str) or *None* if the
              Ensembl ID could not be mapped.

    Example:
        >>> from gsnn.proc.bio import ensg2symbol
        >>> ensgs = pd.Series(['ENSG00000100030', 'ENSG00000171862', 'INVALID'])
        >>> ensg2symbol(ensgs, allow='1:m').head()
                   ensg_id gene_symbol
        0  ENSG00000100030       MAPK1
        1  ENSG00000171862        PTEN
    """

    from pypath.utils import mapping  # lazy: avoids network probes at gsnn import time

    assert allow in ['1:m', '1:1'], 'allow must be one of ["1:m", "1:1"]'

    e2s = {'ensg_id': [], 'gene_symbol': []}

    for e in np.unique(ensg_ids): 
        
        s = mapping.map_name(e, 'ensg', 'genesymbol')

        if len(s) > 0: 
            for s_ in s:  
                e2s['ensg_id'].append(e)
                e2s['gene_symbol'].append(s_)
                if allow == '1:1': break # only map one gene symbol per ensg id; first in set
        else: 
            e2s['ensg_id'].append(e)
            e2s['gene_symbol'].append(None)

    e2s = pd.DataFrame(e2s)

    if drop_na: 
        e2s = e2s.dropna()

    return e2s.drop_duplicates()