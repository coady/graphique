import pytest


def test_slice(client):
    data = client.execute('{ slice(length: 3) { zipcode city state } }')
    assert data == {
        'slice': {
            'zipcode': [501, 544, 601],
            'city': ['Holtsville', 'Holtsville', 'Adjuntas'],
            'state': ['NY', 'NY', 'PR'],
        }
    }
    data = client.execute('{ slice(offset: 1) { zipcode } }')
    zipcodes = data['slice']['zipcode']
    assert zipcodes[0] == 544
    assert len(zipcodes) == 41699


def test_unique(client):
    data = client.execute('{ unique { values { state } } }')
    states = data['unique']['values']['state']
    assert len(states) == 52
    assert min(states) == 'AK'
    data = client.execute('{ unique { values { state } counts { state } } }')
    assert data['unique']['values']['state'] == states
    counts = data['unique']['counts']['state']
    assert len(counts) == len(states)
    assert min(counts) == 91


def test_counts(client):
    data = client.execute('{ count }')
    assert data == {'count': 41700}
    data = client.execute('{ nullCount { state } }')
    assert data == {'nullCount': {'state': 0}}


def test_reduce(client):
    data = client.execute('{ sum { zipcode } }')
    assert data == {'sum': {'zipcode': 2066562337}}
    data = client.execute('{ min { latitude } }')
    assert data['min']['latitude'] == pytest.approx(17.963333)
    data = client.execute('{ max { longitude } }')
    assert data['max']['longitude'] == pytest.approx(-65.301389)


def test_search(client):
    data = client.execute('{ search { zipcode } }')
    assert len(data['search']['zipcode']) == 41700
    data = client.execute('{ search(equals: {}) { zipcode } }')
    assert len(data['search']['zipcode']) == 41700
    data = client.execute('{ search(equals: {zipcode: 501}) { zipcode } }')
    assert data == {'search': {'zipcode': [501]}}

    data = client.execute('{ search(range: {}) { zipcode } }')
    assert len(data['search']['zipcode']) == 41700
    data = client.execute('{ search(range: {zipcode: {lower: 99929}}) { zipcode } }')
    assert data == {'search': {'zipcode': [99929, 99950]}}
    data = client.execute('{ search(range: {zipcode: {upper: 601}}) { zipcode } }')
    assert data == {'search': {'zipcode': [501, 544]}}
    data = client.execute(
        '''{ search(range: {zipcode:
        {lower: 501, upper: 601, includeLower: false, includeUpper: true}}) { zipcode } }'''
    )
    assert data == {'search': {'zipcode': [544, 601]}}

    data = client.execute('{ search(isin: {zipcode: []}) { zipcode } }')
    assert data == {'search': {'zipcode': []}}
    data = client.execute('{ search(isin: {zipcode: [0]}) { zipcode } }')
    assert data == {'search': {'zipcode': []}}
    data = client.execute('{ search(isin: {zipcode: [501, 601]}) { zipcode } }')
    assert data == {'search': {'zipcode': [501, 601]}}

    with pytest.raises(ValueError, match="not a prefix"):
        client.execute('{ search(equals: {zipcode: 0}, range: {zipcode: {}}) { zipcode } }')
    with pytest.raises(ValueError, match="not a prefix"):
        client.execute('{ search(equals: {zipcode: 0}, isin: {zipcode: []}) { zipcode } }')
    with pytest.raises(ValueError, match="only one"):
        client.execute('{ search(range: {zipcode: {}}, isin: {zipcode: []}) { zipcode } }')
