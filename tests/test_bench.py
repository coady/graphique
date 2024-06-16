import pytest
from graphique.core import Table as T


@pytest.mark.benchmark
def test_group(table):
    T.group(table, 'state', 'county', 'city')
    T.runs(table, 'state', 'county', 'city')


@pytest.mark.benchmark
def test_rank(table):
    T.ranked(table, 1, 'state', 'county', 'city')
    T.ranked(table, 10, 'state', 'county', 'city')


@pytest.mark.benchmark
def test_sort(table):
    T.sort(table, 'state', 'county', 'city', length=1)
    T.sort(table, 'state', 'county', 'city', length=10)
    T.sort(table, 'state', 'county', 'city')
