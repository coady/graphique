import pytest


def test_slice(client):
    data = client.execute(query='{ slice(stop: 3) { zipcode city state } }')
    assert data == {
        'slice': {
            'zipcode': [501, 544, 601],
            'city': ['Holtsville', 'Holtsville', 'Adjuntas'],
            'state': ['NY', 'NY', 'PR'],
        }
    }


def test_unique(client):
    data = client.execute(query='{ unique { state } }')
    states = data['unique']['state']
    assert len(states) == 52
    assert min(states) == 'AK'


def test_counts(client):
    data = client.execute(query='{ count }')
    assert data == {'count': 41700}
    data = client.execute(query='{ uniqueCount { state } }')
    assert data == {'uniqueCount': {'state': 52}}
    data = client.execute(query='{ nullCount { state } }')
    assert data == {'nullCount': {'state': 0}}


def test_sum(client):
    data = client.execute(query='{ sum { zipcode } }')
    assert data == {'sum': {'zipcode': 2066562337}}


def test_search(client, monkeypatch):
    data = client.execute(query='{ search { zipcode } }')
    assert len(data['search']['zipcode']) == 41700
    data = client.execute(query='{ search(ranges: [{}]) { zipcode } }')
    assert len(data['search']['zipcode']) == 41700
    assert len(data['search']['zipcode']) == 41700
    data = client.execute(query='{ search(ranges: [{lower: 501, upper: 501}]) { zipcode } }')
    assert data == {'search': {'zipcode': []}}
    data = client.execute(query='{ search(ranges: [{lower: 501, upper: 601}]) { zipcode } }')
    assert data == {'search': {'zipcode': [501, 544]}}
    data = client.execute(
        query='''{ search(ranges:
        [{lower: 501, upper: 601, includeLower: false, includeUpper: true}]) { zipcode } }'''
    )
    assert data == {'search': {'zipcode': [544, 601]}}
    with pytest.raises(ValueError, match="too many"):
        client.execute(query='{ search(ranges: [{}, {}]) { zipcode } }')
    from graphique import service

    monkeypatch.setattr(service, 'index', ['zipcode', 'state'])
    with pytest.raises(ValueError, match="after a multi-valued"):
        client.execute(query='{ search(ranges: [{}, {}]) { zipcode } }')
