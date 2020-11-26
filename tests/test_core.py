from datetime import date
import pytest
import pyarrow as pa
import pyarrow.compute as pc
from graphique.core import Chunk, ListChunk, Column as C, Table as T


def eq(left, right):
    return left == right and type(left) is type(right)


def test_dictionary(table):
    array = table['state'].dictionary_encode()
    values, counts = C.value_counts(array).flatten()
    assert len(values) == len(counts) == 52
    assert set(C.unique(array)) == set(values)
    assert C.min(array) == 'AK'
    assert C.max(array) == 'WY'
    assert C.min(array[:0]) is None
    assert C.max(array[:0]) is None
    assert sum(C.mask(array, match_substring="CA").to_pylist()) == 2647
    assert sum(C.mask(array, is_in=["CA"]).to_pylist()) == 2647
    assert "ca" in C.call(array, pc.utf8_lower).unique().dictionary.to_pylist()
    assert C.sort(array)[0].as_py() == "AK"
    table = pa.Table.from_pydict({'state': array})
    assert T.sort(table, 'state')['state'][0].as_py() == 'AK'


def test_chunks():
    array = pa.chunked_array([list('aba'), list('bcb')])
    assert C.minimum(array, array).equals(array)
    assert C.maximum(array, array).equals(array)
    table = pa.Table.from_pydict({'col': array})
    assert T.unique(table, 'col', count=True)[1].to_pylist() == [2, 3, 1]
    groups, counts = T.group(table, 'col')
    assert groups['col'].to_pylist() == list('abc')
    assert counts.to_pylist() == [2, 3, 1]
    assert table['col'].unique() == pa.array('abc')
    assert C.sort(array, length=1).to_pylist() == ['a']
    assert C.sort(array, reverse=True).to_pylist() == list('cbbbaa')
    assert ''.join(T.unique(table, 'col')[0]['col'].to_pylist()) == 'abc'
    table = pa.Table.from_pydict({'col': array.dictionary_encode()})
    assert ''.join(T.unique(table, 'col', reverse=True)[0]['col'].to_pylist()) == 'bca'
    table = pa.Table.from_pydict({'col': array, 'other': range(6)})
    assert len(T.unique(table, 'col')[0]) == 3


def test_lists():
    array = pa.array([[2, 1], [0, 0], [None], [], None])
    assert ListChunk.first(array).to_pylist() == [2, 0, None, None, None]
    assert ListChunk.last(array).to_pylist() == [1, 0, None, None, None]
    assert ListChunk.unique(array).to_pylist() == [[2, 1], [0], [None], [], []]
    assert ListChunk.min(array).to_pylist() == [1, 0, None, None, None]
    assert ListChunk.max(array).to_pylist() == [2, 0, None, None, None]
    assert ListChunk.sum(array).to_pylist() == [3, 0, None, None, None]
    assert ListChunk.mean(array).to_pylist() == [1.5, 0.0, None, None, None]
    assert ListChunk.mode(array).to_pylist() == [1, 0, None, None, None]
    assert ListChunk.stddev(array).to_pylist() == [0.5, 0.0, None, None, None]
    assert ListChunk.variance(array).to_pylist() == [0.25, 0.0, None, None, None]
    array = pa.array([["a", "b"], [None]])
    assert ListChunk.min(array).to_pylist() == ["a", None]
    assert ListChunk.max(array).to_pylist() == ["b", None]


def test_reduce():
    array = pa.chunked_array([[0, 1], [2, 3]])
    assert eq(C.min(array), 0)
    assert eq(C.max(array), 3)
    assert eq(C.sum(array), 6)
    assert eq(C.mean(array), 1.5)
    assert C.quantile(array, 0.5) == [1.5]
    array = pa.chunked_array([[None, 1]])
    assert eq(C.min(array), 1)
    assert eq(C.max(array), 1)
    assert C.quantile(array, 0.5) == [1.0]
    assert C.quantile(array[:1], 0.5) == [None]
    array = pa.chunked_array([[0, 3, 0]])
    assert C.mode(array) == 0
    assert C.stddev(array) == pytest.approx(2.0 ** 0.5)
    assert C.variance(array) == 2.0
    array = pa.chunked_array([[date.today(), None]])
    assert C.min(array) == C.max(array[:1])
    assert C.min(array[1:]) is C.max(array[1:]) is None


def test_membership():
    array = pa.chunked_array([[0]])
    assert C.count(array, True) == 0
    array = pa.chunked_array([[0, 1]])
    assert C.count(array, True) == 1
    array = pa.chunked_array([[1, 1]])
    assert C.count(array, True) == 2
    assert C.count(array, False) == C.count(array, None) == 0
    assert C.count(array, 0) == 0 and C.count(array, 1) == 2


def test_functional(table):
    array = table['state'].dictionary_encode()
    assert set(C.mask(array).to_pylist()) == {True}
    assert set(C.equal(array, None).to_pylist()) == {False}
    assert set(C.not_equal(array, None).to_pylist()) == {True}
    mask = C.equal(array, 'CA')
    assert mask == C.is_in(array, ['CA']) == C.is_in(table['state'], ['CA'])
    assert C.not_equal(array, 'CA') == pc.invert(mask)
    assert len(array.filter(mask)) == 2647
    assert sum(C.mask(array, less_equal='CA', greater_equal='CA').to_pylist()) == 2647
    assert sum(C.mask(array, binary_length={'equal': 2}).to_pylist()) == 41700
    assert sum(C.mask(array, utf8_is_upper=True).to_pylist()) == 41700
    assert sum(C.mask(array, utf8_is_upper=False).to_pylist()) == 41700
    assert sum(C.mask(table['longitude'], absolute={'less': 0}).to_pylist()) == 0
    mask = T.mask(table, 'state', equal='CA', apply={'minimum': 'state'})
    assert sum(mask.to_pylist()) == 2647
    mask = T.mask(table, 'city', apply={'equal': 'county'})
    assert sum(mask.to_pylist()) == 2805
    table = T.apply(table, 'latitude', alias='diff', subtract='longitude')
    assert table['latitude'][0].as_py() - table['longitude'][0].as_py() == table['diff'][0].as_py()
    table = T.apply(table, 'zipcode', fill_null=0)
    assert not table['zipcode'].null_count
    table = T.apply(table, 'state', utf8_lower=True, utf8_upper=False)
    assert table['state'][0].as_py() == 'ny'


def test_group(table):
    groups, counts = T.group(table, 'state')
    assert len(groups) == len(counts) == 52
    assert groups['state'][0].as_py() == 'NY'
    groups, counts = T.group(table, 'state', reverse=True, length=2)
    assert groups['state'].to_pylist() == ['AK', 'WA']
    assert counts.to_pylist() == [273, 732]


def test_unique(table):
    assert len(T.unique(table, 'zipcode')[0]) == 41700
    zipcodes = T.unique(table, 'state')[0]['zipcode'].to_pylist()
    assert len(zipcodes) == 52
    assert zipcodes[0] == 501
    assert zipcodes[-1] == 99501
    zipcodes = T.unique(table, 'state', reverse=True)[0]['zipcode'].to_pylist()
    assert len(zipcodes) == 52
    assert zipcodes[0] == 99950
    assert zipcodes[-1] == 988
    indices, _ = Chunk.unique_indices(pa.array([1, None, 1]))
    assert indices.to_pylist() == [0, 1]
    table, counts = T.unique(table, 'state', 'county', 'city', reverse=True, count=True)
    assert len(table) == 29865
    assert table['zipcode'][0].as_py() == 99927
    assert counts[0].as_py() == 1


def test_sort(table):
    states = C.sort(table['state']).to_pylist()
    assert (states[0], states[-1]) == ('AK', 'WY')
    states = C.sort(table.combine_chunks()['state'], reverse=True).to_pylist()
    assert (states[0], states[-1]) == ('WY', 'AK')
    assert C.sort(table['state'], length=1).to_pylist() == ['AK']
    assert C.sort(table['state'], reverse=True, length=1).to_pylist() == ['WY']
    len(C.sort(table['state'], length=10 ** 5)) == 41700
    assert C.sort(table['state'], length=0).to_pylist() == []
    data = T.sort(table, 'state').to_pydict()
    assert (data['state'][0], data['county'][0]) == ('AK', 'Anchorage')
    data = T.sort(table, 'state', 'county', length=1).to_pydict()
    assert (data['state'], data['county']) == (['AK'], ['Aleutians East'])
    data = T.sort(table, 'state', 'county', 'city', reverse=True, length=2).to_pydict()
    assert data['state'] == ['WY', 'WY']
    assert data['county'] == ['Weston', 'Weston']
    assert data['city'] == ['Upton', 'Osage']
