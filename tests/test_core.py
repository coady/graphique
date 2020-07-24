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
    assert groups == {'a': [0, 2], 'b': [1, 0, 2], 'c': [1]}
    table = pa.Table.from_pydict({'col': array})
    tables = list(T.group(table, 'col'))
    assert list(map(len, tables)) == [2, 3, 1]
    assert table['col'].unique() == pa.array('abc')
    chunk = array.chunk(0)
    assert list(C.predicate()(chunk)) == chunk.to_pylist()
    assert list(C.predicate(equal='a', less='c')(chunk)) == [True, False, True]
    assert list(C.predicate(not_equal='a')(chunk)) == [False, True, False]
    assert C.sort(array, length=1).to_pylist() == ['a']
    assert C.sort(array, reverse=True).to_pylist() == list('cbbbaa')
    groups = [''.join(tbl['col'].to_pylist()) for tbl in T.group(table, 'col')]
    assert groups == ['aa', 'bbb', 'c']
    assert ''.join(T.unique(table, 'col')['col'].to_pylist()) == 'abc'
    table = pa.Table.from_pydict({'col': array.dictionary_encode()})
    groups = [''.join(tbl['col'].to_pylist()) for tbl in T.group(table, 'col', reverse=True)]
    assert groups == ['c', 'bbb', 'aa']
    assert ''.join(T.unique(table, 'col', reverse=True)['col'].to_pylist()) == 'bca'
    table = pa.Table.from_pydict({'col': array, 'other': range(6)})
    assert len(list(T.group(table, 'col'))) == 3
    assert len(T.unique(table, 'col')) == 3


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


def test_group(table):
    groups = C.arggroupby(table['zipcode'])
    assert len(groups) == 41700
    assert set(map(len, groups.values())) == {1}
    groups = C.arggroupby(table['state'])
    tables = list(T.group(table, 'state'))
    assert len(groups) == len(tables) == 52
    assert list(map(len, groups.values())) == list(map(len, tables))
    assert set(table['state'].chunk(0).take(groups['CA'])) == {pa.scalar('CA')}
    groups = C.arggroupby(table['latitude'])
    assert max(map(len, groups.values())) == 6


def test_unique(table):
    assert len(T.unique(table, 'zipcode')) == 41700
    zipcodes = T.unique(table, 'state')['zipcode'].to_pylist()
    assert len(zipcodes) == 52
    assert zipcodes[0] == 501
    assert zipcodes[-1] == 99501
    zipcodes = T.unique(table, 'state', reverse=True)['zipcode'].to_pylist()
    assert len(zipcodes) == 52
    assert zipcodes[0] == 99950
    assert zipcodes[-1] == 988


def test_sort(table):
    states = C.sort(table['state']).to_pylist()
    assert (states[0], states[-1]) == ('AK', 'WY')
    states = C.sort(table['state'], reverse=True).to_pylist()
    assert (states[0], states[-1]) == ('WY', 'AK')
    assert C.sort(table['state'], length=1).to_pylist() == ['AK']
    assert C.sort(table['state'], reverse=True, length=1).to_pylist() == ['WY']
    data = T.sort(table, 'state').to_pydict()
    assert (data['state'][0], data['county'][0]) == ('AK', 'Anchorage')
    data = T.sort(table, 'state', 'county', length=1).to_pydict()
    assert (data['state'], data['county']) == (['AK'], ['Aleutians East'])
    data = T.sort(table, 'state', 'county', 'city', reverse=True, length=2).to_pydict()
    assert data['state'] == ['WY', 'WY']
    assert data['county'] == ['Weston', 'Weston']
    assert data['city'] == ['Upton', 'Osage']
