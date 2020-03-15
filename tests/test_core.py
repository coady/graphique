import pytest
from graphique.core import Array as A, Table as T


def test_dictionary(table):
    array = table['state'].dictionary_encode()
    values, counts = A.unique(array, counts=True)
    assert len(values) == len(counts) == 52
    assert set(A.unique(array)) == set(values)
    assert A.min(array) == 'AK'
    assert A.max(array) == 'WY'
    with pytest.raises(ValueError):
        A.min(array[:0])
    with pytest.raises(ValueError):
        A.max(array[:0])


def test_filter(table):
    tbl = T.filter(table, city=lambda a: a == 'Mountain View')
    assert len(tbl) == 11
    assert len(tbl['state'].unique()) == 6
    tbl = T.filter(table, state=lambda a: a == 'CA', city=lambda a: a == 'Mountain View')
    assert len(tbl) == 6
    assert set(tbl['state']) == {'CA'}
