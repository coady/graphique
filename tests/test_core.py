from functools import partial
import pytest
import numpy as np
import pyarrow as pa
from graphique.core import Column as C, Table as T


def test_dictionary(table):
    array = table['state'].dictionary_encode()
    values, counts = C.value_counts(array)
    assert len(values) == len(counts) == 52
    assert set(C.unique(array)) == set(values)
    assert array[C.argmin(array)] == C.min(array) == 'AK'
    assert array[C.argmax(array)] == C.max(array) == 'WY'
    with pytest.raises(ValueError):
        C.min(array[:0])
    with pytest.raises(ValueError):
        C.max(array[:0])


def test_chunks():
    array = pa.chunked_array([list('aba'), list('bcb')])
    assert C.argmin(array) == 0
    assert C.argmax(array) == 4
    with pytest.raises(ValueError):
        C.argmin(array[:0])
    with pytest.raises(ValueError):
        C.argmax(array[:0])
    values, counts = C.value_counts(array.dictionary_encode())
    pair = C.value_counts(array)
    assert values.equals(pair[0])
    assert counts.equals(pair[1])
    groups = {key: list(value) for key, value in C.arggroupby(array).items()}
    assert groups == {'a': [0, 2], 'b': [1, 0, 2], 'c': [1]}


def test_membership():
    array = pa.chunked_array([[0]])
    assert not C.any(array) and not C.all(array) and C.count(array, True) == 0
    array = pa.chunked_array([[0, 1]])
    assert C.any(array) and not C.all(array) and C.count(array, True) == 1
    array = pa.chunked_array([[1, 1]])
    assert C.any(array) and C.all(array) and C.count(array, True) == 2
    assert C.contains(array, 1) and not C.contains(array, 0)
    assert C.count(array, 0) == C.count(array, None) == 0


def test_filter(table):
    array = table['state'].dictionary_encode()
    mask = C.equal(array, 'CA')
    assert len(array.filter(mask)) == 2647

    tbl = T.filter(table, city=partial(np.equal, 'Mountain View'))
    assert len(tbl) == 11
    assert len(tbl['state'].unique()) == 6
    tbl = T.filter(table, state=partial(np.equal, 'CA'), city=partial(np.equal, 'Mountain View'))
    assert len(tbl) == 6
    assert set(tbl['state']) == {'CA'}


def test_groupby(table):
    groups = C.arggroupby(table['zipcode'])
    assert len(groups) == 41700
    assert set(map(len, groups.values())) == {1}
    groups = C.arggroupby(table['state'])
    assert len(groups) == 52
    indices = groups['CA']
    assert set(C.take(table['state'], indices)) == {'CA'}
    groups = C.arggroupby(table['latitude'])
    assert max(map(len, groups.values())) == 6
