import pytest
from graphique.core import Nodes, Table as T


@pytest.mark.benchmark
def test_group(table):
    Nodes('table_source', table).group('state', 'county', 'city')
    T.runs(table, 'state', 'county', 'city')


@pytest.mark.benchmark
def test_rank(table):
    T.rank(table, 1, 'state', 'county', 'city')
    T.rank(table, 10, 'state', 'county', 'city')


@pytest.mark.benchmark
def test_sort(table):
    T.sort(table, 'state', 'county', 'city', length=1)
    T.sort(table, 'state', 'county', 'city', length=10)
    T.sort(table, 'state', 'county', 'city')
