import pytest


def test_slice(client):
    data = client.execute('{ length slice(length: 3) { zipcode { values } } }')
    assert data == {'length': 41700, 'slice': {'zipcode': {'values': [501, 544, 601]}}}
    data = client.execute('{ slice(offset: 1) { zipcode { values } } }')
    zipcodes = data['slice']['zipcode']['values']
    assert zipcodes[0] == 544
    assert len(zipcodes) == 41699


def test_ints(client):
    data = client.execute('{ slice { zipcode { values sum min max unique { values counts } } } }')
    zipcodes = data['slice']['zipcode']
    assert len(zipcodes['values']) == 41700
    assert zipcodes['sum'] == 2066562337
    assert zipcodes['min'] == 501
    assert zipcodes['max'] == 99950
    assert len(zipcodes['unique']['values']) == 41700
    assert set(zipcodes['unique']['counts']) == {1}


def test_floats(client):
    data = client.execute('{ slice { latitude { values sum } longitude { min max } } }')
    latitudes = data['slice']['latitude']
    assert len(latitudes['values']) == 41700
    assert latitudes['sum'] == pytest.approx(1606220.07592)
    longitudes = data['slice']['longitude']
    assert longitudes['min'] == pytest.approx(-174.21333)
    assert longitudes['max'] == pytest.approx(-65.301389)


def test_strings(client):
    data = client.execute(
        '''{ slice {
        state { values unique { values counts } }
        county { unique { length values } }
        city { min max }
    } }'''
    )
    states = data['slice']['state']
    assert len(states['values']) == 41700
    assert len(states['unique']['values']) == 52
    assert sum(states['unique']['counts']) == 41700
    counties = data['slice']['county']
    assert len(counties['unique']['values']) == counties['unique']['length'] == 1920
    cities = data['slice']['city']
    assert cities['min'] == 'Aaronsburg'
    assert cities['max'] == 'Zwolle'


def test_search(client):
    data = client.execute('{ index search { length } }')
    assert data['index'] == ['zipcode']
    assert data['search']['length'] == 41700
    data = client.execute('{ search(equal: {}) { length } }')
    assert data['search']['length'] == 41700
    data = client.execute('{ search(equal: {zipcode: 501}) { slice { zipcode { values } } } }')
    assert data == {'search': {'slice': {'zipcode': {'values': [501]}}}}

    data = client.execute('{ search(range: {}) { length } }')
    assert data['search']['length'] == 41700
    data = client.execute(
        '''{ search(range: {zipcode: {lower: 99929}})
        { slice { zipcode { values } } } }'''
    )
    assert data == {'search': {'slice': {'zipcode': {'values': [99929, 99950]}}}}
    data = client.execute(
        '''{ search(range: {zipcode: {upper: 601}})
        { slice { zipcode { values } } } }'''
    )
    assert data == {'search': {'slice': {'zipcode': {'values': [501, 544]}}}}
    data = client.execute(
        '''{ search(range: {zipcode:
            {lower: 501, upper: 601, includeLower: false, includeUpper: true}})
        { slice { zipcode { values } } } }'''
    )
    assert data == {'search': {'slice': {'zipcode': {'values': [544, 601]}}}}

    data = client.execute('{ search(isin: {}) { length } }')
    assert data['search']['length'] == 41700
    data = client.execute('{ search(isin: {zipcode: []}) { length } }')
    assert data == {'search': {'length': 0}}
    data = client.execute('{ search(isin: {zipcode: [0]}) { length } }')
    assert data == {'search': {'length': 0}}
    data = client.execute(
        '''{ search(isin: {zipcode: [501, 601]})
        { slice { zipcode { values } } } }'''
    )
    assert data == {'search': {'slice': {'zipcode': {'values': [501, 601]}}}}

    with pytest.raises(ValueError, match="not a prefix"):
        client.execute('{ search(equal: {zipcode: 0}, range: {zipcode: {}}) { length } }')
    with pytest.raises(ValueError, match="not a prefix"):
        client.execute('{ search(equal: {zipcode: 0}, isin: {zipcode: []}) { length } }')
    with pytest.raises(ValueError, match="only one"):
        client.execute('{ search(range: {zipcode: {}}, isin: {zipcode: []}) { length } }')
