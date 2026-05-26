import inspect

import pandas as pd
import pandas.testing as pdt

from gsnn.proc.bio import build_uniprot_symbol_map, get_bio_interactions


def _edges(rows):
    """Build a minimal func_edges DataFrame from row dicts."""
    return pd.DataFrame(rows, columns=[
        'src', 'dst', 'edge_type', 'source_uniprot', 'target_uniprot',
    ])


def test_many_uniprots_per_symbol():
    edges = _edges([
        ('PROTEIN__TP53', 'RNA__FOO', 'dorothea', 'P04637', 'Q00987'),
        ('PROTEIN__TP53', 'RNA__BAR', 'omnipath', 'Q53GA5', 'Q00988'),
        ('RNA__BAZ', 'PROTEIN__TP53', 'translation', 'Q00989', 'P38398'),
    ])
    out = build_uniprot_symbol_map(edges)
    tp53 = out.loc[out['func_name'] == 'PROTEIN__TP53']
    assert len(tp53) == 3
    assert set(tp53['uniprot']) == {'P04637', 'Q53GA5', 'P38398'}
    assert (tp53['gene_symbol'] == 'TP53').all()


def test_many_symbols_per_uniprot():
    edges = _edges([
        ('PROTEIN__SYM1', 'RNA__FOO', 'omnipath', 'P38398', 'Q00987'),
        ('PROTEIN__SYM2', 'RNA__BAR', 'omnipath', 'P38398', 'Q00988'),
    ])
    out = build_uniprot_symbol_map(edges)
    p38398 = out.loc[out['uniprot'] == 'P38398']
    assert set(p38398['func_name']) == {'PROTEIN__SYM1', 'PROTEIN__SYM2'}


def test_endpoint_coverage_dst_only():
    edges = _edges([
        ('RNA__FOO', 'PROTEIN__BAR', 'translation', 'Q00987', 'P38398'),
    ])
    out = build_uniprot_symbol_map(edges)
    bar = out.loc[out['func_name'] == 'PROTEIN__BAR']
    assert len(bar) == 1
    assert bar.iloc[0]['uniprot'] == 'P38398'
    assert bar.iloc[0]['gene_symbol'] == 'BAR'


def test_filter_complex_mirbase_nan():
    edges = _edges([
        ('PROTEIN__TP53', 'COMPLEX__A_B', 'assembly', 'P04637', 'COMPLEX:P04637_Q00987'),
        ('PROTEIN__MYC', 'RNA__MIR21', 'tf_mirna', 'MIMAT0000062', 'Q00987'),
        ('PROTEIN__EGFR', 'RNA__FOO', 'dorothea', 'P00533', pd.NA),
        ('PROTEIN__KRAS', 'RNA__BAR', 'omnipath', 'P01116', 'Q00988'),
    ])
    out = build_uniprot_symbol_map(edges)
    assert 'COMPLEX:P04637_Q00987' not in out['uniprot'].values
    assert 'MIMAT0000062' not in out['uniprot'].values
    # NaN target_uniprot on the EGFR row must not create a spurious entry
    assert out.loc[out['func_name'] == 'PROTEIN__EGFR', 'uniprot'].tolist() == ['P00533']
    assert 'PROTEIN__TP53' in out['func_name'].values
    assert 'PROTEIN__KRAS' in out['func_name'].values
    assert out.loc[out['func_name'] == 'PROTEIN__TP53', 'uniprot'].iloc[0] == 'P04637'
    assert out.loc[out['func_name'] == 'PROTEIN__KRAS', 'uniprot'].iloc[0] == 'P01116'


def test_determinism():
    edges = _edges([
        ('PROTEIN__B', 'RNA__Y', 'omnipath', 'P38398', 'Q00987'),
        ('PROTEIN__A', 'RNA__X', 'dorothea', 'P04637', 'Q00988'),
        ('RNA__Z', 'PROTEIN__C', 'translation', 'Q00989', 'P00533'),
    ])
    out1 = build_uniprot_symbol_map(edges)
    out2 = build_uniprot_symbol_map(edges)
    pdt.assert_frame_equal(out1, out2)

    shuffled = edges.sample(frac=1, random_state=42).reset_index(drop=True)
    out3 = build_uniprot_symbol_map(shuffled)
    pdt.assert_frame_equal(out1, out3)


def test_node_kind_tag():
    edges = _edges([
        ('PROTEIN__FOO', 'RNA__MIR21', 'dorothea', 'P38398', 'Q00987'),
        ('RNA__MIR21', 'PROTEIN__BAR', 'mirna', 'Q00988', 'P04637'),
    ])
    out = build_uniprot_symbol_map(edges)
    prot = out.loc[out['func_name'].str.startswith('PROTEIN__')]
    rna = out.loc[out['func_name'].str.startswith('RNA__')]
    assert (prot['node_kind'] == 'PROTEIN').all()
    assert (rna['node_kind'] == 'RNA').all()


def test_get_bio_interactions_signature():
    sig = inspect.signature(get_bio_interactions)
    assert 'return_uniprot_map' in sig.parameters
    assert sig.parameters['return_uniprot_map'].default is False
