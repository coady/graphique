from functools import partial
import pytest
import numpy as np
import pyarrow as pa
from graphique.core import Array as A, Table as T


def test_dictionary(table):
    array = table['state'].dictionary_encode()
    values, counts = A.value_counts(array)
    assert len(values) == len(counts) == 52
    assert set(A.unique(array)) == set(values)
    assert array[A.argmin(array)] == A.min(array) == 'AK'
    assert array[A.argmax(array)] == A.max(array) == 'WY'
    with pytest.raises(ValueError):
        A.min(array[:0])
    with pytest.raises(ValueError):
        A.max(array[:0])


def test_chunks():
    array = pa.chunked_array([list('aba'), list('bcb')])
    assert A.argmin(array) == 0
    assert A.argmax(array) == 4
    with pytest.raises(ValueError):
        A.argmin(array[:0])
    with pytest.raises(ValueError):
        A.argmax(array[:0])
    values, counts = A.value_counts(array.dictionary_encode())
    pair = A.value_counts(array)
    assert values.equals(pair[0])
    assert counts.equals(pair[1])
    groups = {key: list(value) for key, value in A.arggroupby(array).items()}
    assert groups == {'a': [0, 2], 'b': [1, 0, 2], 'c': [1]}


def test_boolean():
    array = pa.chunked_array([[0]])
    assert not A.any(array) and not A.all(array) and A.count(array) == 0
    array = pa.chunked_array([[0, 1]])
    assert A.any(array) and not A.all(array) and A.count(array) == 1
    array = pa.chunked_array([[1, 1]])
    assert A.any(array) and A.all(array) and A.count(array) == 2


def test_filter(table):
    array = table['state'].dictionary_encode()
    mask = A.equal(array, 'CA')
    assert len(array.filter(mask)) == 2647

    tbl = T.filter(table, city=partial(np.equal, 'Mountain View'))
    assert len(tbl) == 11
    assert len(tbl['state'].unique()) == 6
    tbl = T.filter(table, state=partial(np.equal, 'CA'), city=partial(np.equal, 'Mountain View'))
    assert len(tbl) == 6
    assert set(tbl['state']) == {'CA'}


def test_groupby(table):
    groups = A.arggroupby(table['zipcode'])
    assert len(groups) == 41700
    assert set(map(len, groups.values())) == {1}
    groups = A.arggroupby(table['state'])
    assert len(groups) == 52
    indices = groups['CA']
    assert set(A.take(table['state'], indices)) == {'CA'}
    groups = A.arggroupby(table['latitude'])
    assert max(map(len, groups.values())) == 6
