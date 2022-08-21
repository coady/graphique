from datetime import time
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from graphique.core import Agg, ListChunk, Column as C, Table as T


def test_dictionary(table):
    array = table['state'].dictionary_encode()
    assert C.min(array) == 'AK'
    assert C.max(array) == 'WY'
    assert C.min_max(array[:0]) == {'min': None, 'max': None}
    assert sum(C.mask(array, match_substring="CA").to_pylist()) == 2647
    assert sum(C.call(array, pc.match_substring, "ca", ignore_case=True).to_pylist()) == 2647
    assert sum(C.mask(array, match_substring='CA', regex=True).to_pylist()) == 2647
    assert sum(C.mask(array, is_in=['CA']).to_pylist()) == 2647
    assert "ca" in C.call(array, pc.utf8_lower).unique().to_pylist()
    table = pa.table({'state': array})
    assert T.sort(table, 'state')['state'][0].as_py() == 'AK'
    array = C.call(pa.chunked_array([[0, 0]]).dictionary_encode(), pc.add, 1)
    assert array.to_pylist() == [1, 1]
    array = pa.chunked_array([['a', 'b'], ['a', 'b', None]]).dictionary_encode()
    assert C.fill_null(array, "c").to_pylist() == C.fill_null(C.decode(array), "c").to_pylist()
    assert C.fill_null(array[3:], "c").to_pylist() == list('bc')
    assert C.fill_null(array[:3], "c").to_pylist() == list('aba')
    assert not C.mask(pa.chunked_array([], 'string').dictionary_encode(), utf8_is_upper=True)


def test_chunks():
    array = pa.chunked_array([list('aba'), list('bcb')])
    table = pa.table({'col': array})
    groups = T.group(table, 'col', counts='counts')
    assert groups['col'].to_pylist() == list('abc')
    assert groups['counts'].to_pylist() == [2, 3, 1]
    assert table['col'].unique() == pa.array('abc')
    array = pa.chunked_array([pa.array(list(chunk)).dictionary_encode() for chunk in ('aba', 'ca')])
    assert pa.Array.equals(*(chunk.dictionary for chunk in C.unify_dictionaries(array).chunks))
    assert C.equal(array, 'a').to_pylist() == [True, False, True, False, True]
    assert C.index(array, 'a') == 0
    assert C.index(array, 'c') == 3
    assert C.index(array, 'a', start=3) == 4
    assert C.index(array, 'b', start=2) == -1


def test_lists():
    array = pa.array([[2, 1], [0, 0], [None], [], None])
    assert ListChunk.first(array).to_pylist() == [2, 0, None, None, None]
    assert ListChunk.element(array, -2).to_pylist() == [2, 0, None, None, None]
    assert ListChunk.last(array).to_pylist() == [1, 0, None, None, None]
    assert ListChunk.one(array).to_pylist()[1:] == [0, None]
    assert ListChunk.element(array, 1).to_pylist() == [1, 0, None, None, None]
    assert ListChunk.unique(array).to_pylist() == [[2, 1], [0], [None], [], []]
    assert ListChunk.distinct(array).to_pylist() == [[2, 1], [0], []]
    assert ListChunk.distinct(array, mode='all').to_pylist() == [[2, 1], [0], [None]]
    assert ListChunk.min(array).to_pylist() == [1, 0, None, None, None]
    assert ListChunk.max(array).to_pylist() == [2, 0, None, None, None]
    assert ListChunk.sum(array).to_pylist() == [3, 0, None, None, None]
    assert ListChunk.product(array).to_pylist() == [2, 0, None, None, None]
    assert ListChunk.mean(array).to_pylist() == [1.5, 0.0, None, None, None]
    assert ListChunk.mode(array).to_pylist() == [1, 0, None, None, None]
    assert ListChunk.mode(array, n=1).to_pylist() == [[1], [0], [], [], []]
    assert ListChunk.quantile(array).to_pylist() == [1.5, 0.0, None, None, None]
    quantile = ListChunk.quantile(array, q=[0.75])
    assert quantile.to_pylist() == [[1.75], [0.0], [None], [None], [None]]
    assert ListChunk.tdigest(array).to_pylist() == [1.0, 0.0, None, None, None]
    tdigest = ListChunk.tdigest(array, q=[0.75])
    assert tdigest.to_pylist() == [[2.0], [0.0], [None], [None], [None]]
    assert ListChunk.stddev(array).to_pylist() == [0.5, 0.0, None, None, None]
    assert ListChunk.variance(array).to_pylist() == [0.25, 0.0, None, None, None]
    array = pa.array([[True, True], [False, False], [None], [], None])
    assert ListChunk.any(array).to_pylist() == [True, False, None, None, None]
    assert ListChunk.all(array).to_pylist() == [True, False, None, None, None]
    array = pa.ListArray.from_arrays([0, 2, 3], pa.array(["a", "b", None]).dictionary_encode())
    assert ListChunk.count_distinct(array).to_pylist() == [2, 0]
    assert ListChunk.min(array).to_pylist() == ["a", None]
    assert ListChunk.max(array).to_pylist() == ["b", None]


def test_membership():
    array = pa.chunked_array([[0]])
    assert C.count(C.mask(array, is_nan=False), True) == 0
    array = pa.chunked_array([[0, 1]])
    assert C.count(array, 1) == 1
    array = pa.chunked_array([[1, 1]])
    assert C.count(array, 0) == C.count(array, None) == 0
    assert C.count(array, 1) == 2
    assert C.index(array, 1) == C.index(array, 1, end=1) == 0
    assert C.index(array, 1, start=1) == 1
    assert C.index(array, 1, start=2) == -1


def test_functional(table):
    array = table['state'].dictionary_encode()
    assert set(C.equal(array, None).to_pylist()) == {False}
    assert set(C.not_equal(array, None).to_pylist()) == {True}
    mask = C.equal(array, 'CA')
    assert C.count(table['city'], table['county'].dictionary_encode()) == 2805
    assert mask == C.is_in(array, ['CA']) == C.is_in(table['state'], ['CA'])
    assert C.not_equal(array, 'CA') == pc.invert(mask)
    assert len(array.filter(mask)) == 2647
    assert sum(C.mask(array, less_equal='CA', greater_equal='CA').to_pylist()) == 2647
    assert sum(C.mask(array, utf8_is_upper=True).to_pylist()) == 41700
    mask = T.mask(table, 'city', apply={'equal': 'county'})
    assert sum(mask.to_pylist()) == 2805
    table = T.apply(table, 'zipcode', fill_null=0)
    assert not table['zipcode'].null_count
    table = T.apply(table, 'state', utf8_lower=True, utf8_upper=False)
    assert table['state'][0].as_py() == 'ny'


def test_group(table):
    groups = T.group(table, 'state', list=[Agg('county'), Agg('city')])
    assert len(groups) == 52
    assert groups['state'][0].as_py() == 'NY'
    assert sum(T.mask(groups, 'county', apply={'equal': 'city'}).to_pylist()) == 2805
    assert sum(T.mask(groups, 'county', apply={'equal': 'state'}).to_pylist()) == 0
    mins = T.matched(groups, C.min, 'state', 'county')
    assert mins['state'].to_pylist() == ['AK']
    assert mins['county'].to_pylist() == [['Aleutians East'] * 5]
    assert mins['city'][0].values[0].as_py() == 'Akutan'
    groups = T.sort_list(groups, 'county')
    assert groups['county'][0].values[0].as_py() == 'Albany'
    groups = T.sort_list(groups, '-county', '-city', length=1)
    assert groups['county'][0].values.to_pylist() == ['Yates']
    assert groups['city'][0].values.to_pylist() == ['Rushville']
    groups = groups.append_column('other', pa.array([[0]] * len(groups)))
    with pytest.raises(ValueError):
        T.sort_list(groups, 'county')


def test_aggregate(table):
    tbl = T.union(table, table.select([0]).rename_columns(['test']))
    assert tbl.column_names == table.column_names + ['test']
    groups = T.group(table, 'state', 'county')
    assert len(groups) == 3216
    assert groups.column_names == ['state', 'county']
    groups = T.group(table, 'state', counts='counts', first=[Agg('county')])
    assert len(groups) == 52
    assert groups['state'][0].as_py() == 'NY'
    assert groups['counts'][0].as_py() == 2205
    assert groups['county'][0].as_py() == 'Suffolk'
    groups = T.group(table, 'state', last=[Agg('city', 'last')], min=[Agg('zipcode')])
    assert groups['last'][0].as_py() == 'Elmira'
    assert groups['zipcode'][0].as_py() == 501
    groups = T.group(table, 'state', max=[Agg('zipcode', 'max', skip_nulls=False)])
    assert groups['max'][0].as_py() == 14925
    groups = T.group(table, 'state', tdigest=[Agg('longitude'), Agg('latitude', q=[0.5])])
    assert groups['longitude'][0].as_py() == pytest.approx(-74.25370)
    assert groups['latitude'][0].as_py() == [pytest.approx(42.34672)]


def test_partition(table):
    groups, counts = T.partition(table, 'state')
    assert len(groups) == len(counts) == 66
    assert pc.sum(counts).as_py() == 41700
    assert groups['state'][0].as_py() == 'NY'
    assert C.count(groups['state'], 'NY') == 3
    assert groups['county'][0].values.to_pylist() == ['Suffolk', 'Suffolk']
    groups, counts = T.partition(table, 'state', 'county')
    assert len(groups) == len(counts) == 22751
    groups, counts = T.partition(table, zipcode=(pc.greater, 100))
    assert len(groups) == len(counts) == 59
    tbl = T.sort(table, 'state', 'longitude')
    groups, counts = T.partition(tbl, 'state', longitude=(pc.greater, 1.0))
    assert len(groups) == len(counts) == 62
    assert groups['state'].value_counts()[0].as_py() == {'values': 'AK', 'counts': 7}
    assert groups['longitude'][:2].to_pylist() == [[-174.213333], [-171.701685]]
    groups, counts = T.partition(tbl, 'state', longitude=(pc.less,))
    assert len(groups) == len(counts) == 52


def test_sort(table):
    data = T.sort(table, 'state').to_pydict()
    assert (data['state'][0], data['county'][0]) == ('AK', 'Anchorage')
    data = T.sort(table, 'state', 'county', length=1).to_pydict()
    assert (data['state'], data['county']) == (['AK'], ['Aleutians East'])
    data = T.sort(table, 'state', '-county', length=2).to_pydict()
    assert data['state'] == ['AK'] * 2
    assert data['county'] == ['Yukon Koyukuk'] * 2
    mask = [False] * len(table)
    assert not T.sort(table.filter(mask), 'state')


def test_time():
    array = pa.array([time(), None], pa.time32('s'))
    assert C.equal(array, time()).to_pylist() == [True, None]


def test_numeric():
    array = pa.chunked_array([range(5)])
    assert C.digitize(array, range(0, 5, 2)).to_pylist() == [1, 1, 2, 2, 3]
    assert C.digitize(array, np.arange(0, 5, 2), right=True).to_pylist() == [0, 1, 1, 2, 2]


def test_not_implemented():
    dictionary = pa.array(['']).dictionary_encode()
    with pytest.raises(NotImplementedError):
        pc.sort_indices(dictionary)
    with pytest.raises(NotImplementedError):
        dictionary.index('')
    with pytest.raises(NotImplementedError):
        pc.min_max(dictionary)
    with pytest.raises(NotImplementedError):
        pc.count_distinct(dictionary)
    with pytest.raises(NotImplementedError):
        pa.StructArray.from_arrays([], []).dictionary_encode()
    for index in (-1, 1):
        with pytest.raises(ValueError):
            pc.list_element(pa.array([[0]]), index)
    assert pc.equal([0, None], None).to_pylist() == [None] * 2
    with pytest.raises(NotImplementedError):
        pc.any([0])
    dictionary = pa.array(['']).dictionary_encode()
    with pytest.raises(ValueError, match="string vs dictionary"):
        pc.index_in(dictionary.unique(), value_set=dictionary)
    array = pa.array(list('aba'))
    with pytest.raises(NotImplementedError):
        pc._group_by([array.dictionary_encode()], [array], [('hash_min', None)])
    with pytest.raises(NotImplementedError):
        pc._group_by([array], [array], [('hash_any', None)])
