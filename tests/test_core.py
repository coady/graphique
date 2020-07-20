import pytest
import numpy as np
import pyarrow as pa
from graphique.core import Column as C, Table as T


def eq(left, right):
    return left == right and type(left) is type(right)


def test_dictionary(table):
    array = table['state'].dictionary_encode()
    values, counts = C.value_counts(array).flatten()
    assert len(values) == len(counts) == 52
    assert set(C.unique(array)) == set(values)
    assert array[C.argmin(array)].as_py() == C.min(array) == 'AK'
    assert array[C.argmax(array)].as_py() == C.max(array) == 'WY'
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
    groups = {key: value.to_pylist() for key, value in C.arggroupby(array).items()}
    assert groups == {'a': [0, 2], 'b': [1, 3, 5], 'c': [4]}
    chunk = array.chunk(0)
    assert list(C.predicate()(chunk)) == chunk.to_pylist()
    assert list(C.predicate(equal="a", less="c")(chunk)) == [True, False, True]
    assert list(C.predicate(not_equal="a")(chunk)) == [False, True, False]
    assert C.sort(array, length=1).to_pylist() == ["a"]
    assert C.argsort(array, length=1).to_pylist() == [0]
    assert set(C.argunique(array).to_pylist()) == {0, 1, 4}
    assert set(C.argunique(array, reverse=True).to_pylist()) == {5, 4, 2}
    arr = pa.chunked_array([[0.0, 1.0, 0.0]])
    assert C.argunique(arr, reverse=True).equals(pa.array([2, 1]))


def test_reduce():
    array = pa.chunked_array([[0, 1], [2, 3]])
    assert eq(C.min(array), 0)
    assert eq(C.max(array), 3)
    assert eq(C.sum(array), 6)
    assert eq(C.sum(array, exp=2), 14)


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
    for key, group in T.arggroupby(table, 'state').items():
        assert groups[key].equals(group)
    assert len(groups) == 52
    assert set(table['state'].chunk(0).take(groups['CA'])) == {pa.scalar('CA')}
    groups = C.arggroupby(table['latitude'])
    assert max(map(len, groups.values())) == 6
    groups = T.arggroupby(table, 'state', 'county')
    group = groups['CA']['Santa Clara']
    assert len(group) == 108
    assert set(table['county'].chunk(0).take(group)) == {pa.scalar('Santa Clara')}
    groups = T.arggroupby(table, 'state', 'county', 'city')
    group = groups['CA']['Santa Clara']['Mountain View']
    assert len(group) == 6
    assert set(table['city'].chunk(0).take(group)) == {pa.scalar('Mountain View')}


def test_unique(table):
    indices = C.argunique(table['zipcode'])
    assert len(indices) == 41700
    indices = C.argunique(table['state'])
    assert T.argunique(table, 'state').equals(indices)
    states = table['state'].chunk(0)
    assert C.argunique(table['state'].dictionary_encode()).equals(indices)
    assert len(indices) == 52
    assert set(states.take(indices)) == set(states)
    first, last = C.argmin(table['zipcode']), C.argmax(table['zipcode'])
    assert first in indices.to_pylist() and last not in indices.to_pylist()
    indices = C.argunique(table['state'], reverse=True)
    assert T.argunique(table, 'state', reverse=True).equals(indices)
    assert first not in indices.to_pylist() and last in indices.to_pylist()
    indices = C.argunique(table['latitude'])
    assert len(indices) < 41700
    assert not C.argunique(table['latitude'], reverse=True).equals(indices)
    indices = T.argunique(table, 'state', 'county')
    keys = zip(states.take(indices), table['county'].chunk(0).take(indices))
    assert len(indices) == len(set(keys)) == 3216


def test_sort(table):
    indices = C.argsort(table['state']).to_pylist()
    states = C.sort(table['state'])
    assert states[0] == table['state'][indices[0]] == pa.scalar('AK')
    assert states[-1] == table['state'][indices[-1]] == pa.scalar('WY')
    indices = C.argsort(table['state'], reverse=True).to_pylist()
    states = C.sort(table['state'], reverse=True)
    assert states[0] == table['state'][indices[0]] == pa.scalar('WY')
    assert states[-1] == table['state'][indices[-1]] == pa.scalar('AK')
    indices = C.argsort(table['state'], length=1).to_pylist()
    states = C.sort(table['state'], length=1)
    assert states.to_pylist() == table['state'].take(indices).to_pylist() == ['AK']
    indices = C.argsort(table['state'], reverse=True, length=1).to_pylist()
    states = C.sort(table['state'], reverse=True, length=1)
    assert states.to_pylist() == table['state'].take(indices).to_pylist() == ['WY']
