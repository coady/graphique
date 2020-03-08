import pytest


def test_slice(client):
    data = client.execute(query='{ slice(length: 3) { zipcode city state } }')
    assert data == {
        'slice': {
            'zipcode': [501, 544, 601],
            'city': ['Holtsville', 'Holtsville', 'Adjuntas'],
            'state': ['NY', 'NY', 'PR'],
        }
    }
    data = client.execute(query='{ slice(offset: 1) { zipcode } }')
    zipcodes = data['slice']['zipcode']
    assert zipcodes[0] == 544
    assert len(zipcodes) == 41699


def test_unique(client):
    data = client.execute(query='{ unique { values { state } } }')
    states = data['unique']['values']['state']
    assert len(states) == 52
    assert min(states) == 'AK'
    data = client.execute(query='{ unique { values { state } counts { state } } }')
    assert data['unique']['values']['state'] == states
    counts = data['unique']['counts']['state']
    assert len(counts) == len(states)
    assert min(counts) == 91


def test_counts(client):
    data = client.execute(query='{ count }')
    assert data == {'count': 41700}
    data = client.execute(query='{ nullCount { state } }')
    assert data == {'nullCount': {'state': 0}}


def test_sum(client):
    data = client.execute(query='{ sum { zipcode } }')
    assert data == {'sum': {'zipcode': 2066562337}}


def test_search(client):
    data = client.execute(query='{ search { zipcode } }')
    assert len(data['search']['zipcode']) == 41700
    data = client.execute(query='{ search(equals: {}) { zipcode } }')
    assert len(data['search']['zipcode']) == 41700
    data = client.execute(query='{ search(equals: {zipcode: 501}) { zipcode } }')
    assert data == {'search': {'zipcode': [501]}}

    data = client.execute(query='{ search(range: {}) { zipcode } }')
    assert len(data['search']['zipcode']) == 41700
    data = client.execute(query='{ search(range: {zipcode: {lower: 99929}}) { zipcode } }')
    assert data == {'search': {'zipcode': [99929, 99950]}}
    data = client.execute(query='{ search(range: {zipcode: {upper: 601}}) { zipcode } }')
    assert data == {'search': {'zipcode': [501, 544]}}
    data = client.execute(
        query='''{ search(range: {zipcode:
        {lower: 501, upper: 601, includeLower: false, includeUpper: true}}) { zipcode } }'''
    )
    assert data == {'search': {'zipcode': [544, 601]}}

    data = client.execute(query='{ search(isin: {zipcode: []}) { zipcode } }')
    assert data == {'search': {'zipcode': []}}
    data = client.execute(query='{ search(isin: {zipcode: [0]}) { zipcode } }')
    assert data == {'search': {'zipcode': []}}
    data = client.execute(query='{ search(isin: {zipcode: [501, 601]}) { zipcode } }')
    assert data == {'search': {'zipcode': [501, 601]}}

    with pytest.raises(ValueError, match="not a prefix"):
        client.execute(query='{ search(equals: {zipcode: 0}, range: {zipcode: {}}) { zipcode } }')
    with pytest.raises(ValueError, match="not a prefix"):
        client.execute(query='{ search(equals: {zipcode: 0}, isin: {zipcode: []}) { zipcode } }')
    with pytest.raises(ValueError, match="only one"):
        client.execute(query='{ search(range: {zipcode: {}}, isin: {zipcode: []}) { zipcode } }')
