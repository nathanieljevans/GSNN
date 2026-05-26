"""Tests for gsnn.proc.bio (ID mapping helpers)."""

import pandas as pd
import pytest

from gsnn.proc.bio import build_uniprot_symbol_map


def test_build_uniprot_symbol_map():
    func_df = pd.DataFrame(
        {
            "src": ["PROTEIN__BRCA1", "PROTEIN__TP53"],
            "dst": ["PROTEIN__RAD51", "PROTEIN__MDM2"],
            "source_uniprot": ["P38398", "P04637"],
            "target_uniprot": ["Q06609", "Q00987"],
        }
    )
    mapping = build_uniprot_symbol_map(func_df)
    assert isinstance(mapping, pd.DataFrame)
    assert "uniprot" in mapping.columns
    assert "gene_symbol" in mapping.columns
    assert len(mapping) >= 2


@pytest.mark.integration
def test_uniprot2symbol_smoke():
    pypath = pytest.importorskip("pypath")
    from gsnn.proc.bio import uniprot2symbol

    df = uniprot2symbol(["P38398"], allow="1:1")
    assert "gene_symbol" in df.columns


@pytest.mark.integration
def test_symbol2uniprot_smoke():
    pytest.importorskip("pypath")
    from gsnn.proc.bio import symbol2uniprot

    df = symbol2uniprot(["TP53"], allow="1:1")
    assert "uniprot_id" in df.columns


@pytest.mark.integration
def test_ensg2symbol_smoke():
    pytest.importorskip("pypath")
    from gsnn.proc.bio import ensg2symbol

    df = ensg2symbol(["ENSG00000141510"], allow="1:1")
    assert "gene_symbol" in df.columns
