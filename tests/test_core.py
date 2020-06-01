import pytest
import numpy as np
import pyarrow as pa
from graphique.core import Column as C, Table as T


def test_dictionary(table):
    array = table['state'].dictionary_encode()
    values, counts = C.value_counts(array).flatten()
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
    assert array.value_counts().equals(C.value_counts(array.dictionary_encode()))
    groups = {key: list(value) for key, value in C.arggroupby(array).items()}
    assert groups == {'a': [0, 2], 'b': [1, 3, 5], 'c': [4]}
    chunk = array.chunk(0)
    assert list(C.predicate()(chunk)) == list(chunk)
    assert list(C.predicate(equal="a", less="c")(chunk)) == [True, False, True]
    assert list(C.predicate(not_equal="a")(chunk)) == [False, True, False]


def test_membership():
    array = pa.chunked_array([[0]])
    assert not C.any(array) and not C.all(array) and C.count(array, True) == 0
    array = pa.chunked_array([[0, 1]])
    assert C.any(array) and not C.all(array) and C.count(array, True) == 1
    array = pa.chunked_array([[1, 1]])
    assert C.any(array) and C.all(array) and C.count(array, True) == 2
    assert C.contains(array, 1) and not C.contains(array, 0)
    assert C.count(array, False) == C.count(array, None) == 0
    assert C.count(array, 0) == 0 and C.count(array, 1) == 2


def test_functional(table):
    array = table['state'].dictionary_encode()
    mask = C.equal(array, 'CA')
    assert mask == C.isin(array, ['CA']) == C.isin(table['state'], ['CA'])
    assert C.not_equal(array, 'CA') == C.isin(array, ['CA'], invert=True)
    assert len(array.filter(mask)) == 2647
    assert T.apply(table, len) == dict.fromkeys(table.column_names, 41700)
    assert T.apply(table, zipcode=len) == {'zipcode': 41700}
    mask = T.mask(table, state=lambda ch: np.equal(ch, 'CA'))
    assert len(table.filter(mask)) == 2647


def test_groupby(table):
    groups = C.arggroupby(table['zipcode'])
    assert len(groups) == 41700
    assert set(map(len, groups.values())) == {1}
    groups = C.arggroupby(table['state'])
    assert len(groups) == 52
    assert set(table['state'].chunks[0].take(groups['CA'])) == {'CA'}
    groups = C.arggroupby(table['latitude'])
    assert max(map(len, groups.values())) == 6


def test_sort(table):
    indices = C.argsort(table['state']).tolist()
    states = C.sort(table['state'])
    assert states[0] == table['state'][indices[0]] == 'AK'
    assert states[-1] == table['state'][indices[-1]] == 'WY'
    indices = C.argsort(table['state'], reverse=True).tolist()
    states = C.sort(table['state'], reverse=True)
    assert states[0] == table['state'][indices[0]] == 'WY'
    assert states[-1] == table['state'][indices[-1]] == 'AK'
    indices = C.argsort(table['state'], length=1).tolist()
    states = C.sort(table['state'], length=1)
    assert list(states) == [table['state'][i] for i in indices] == ['AK']
    indices = C.argsort(table['state'], reverse=True, length=1).tolist()
    states = C.sort(table['state'], reverse=True, length=1)
    assert list(states) == [table['state'][i] for i in indices] == ['WY']
