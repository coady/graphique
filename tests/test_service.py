import pytest


def test_slice(client):
    data = client.execute('{ length slice(length: 3) { zipcode { values } } }')
    assert data == {'length': 41700, 'slice': {'zipcode': {'values': [501, 544, 601]}}}
    data = client.execute('{ slice(offset: 1) { zipcode { values } } }')
    zipcodes = data['slice']['zipcode']['values']
    assert zipcodes[0] == 544
    assert len(zipcodes) == 41699
    data = client.execute('{ slice { zipcode { any(equal: null) count(notEqual: null) } } }')
    zipcodes = data['slice']['zipcode']
    assert not zipcodes['any'] and zipcodes['count'] == 41700
    data = client.execute('{ slice { zipcode { all(notEqual: null) count(equal: null) } } }')
    zipcodes = data['slice']['zipcode']
    assert zipcodes['all'] and zipcodes['count'] == 0


def test_ints(client):
    data = client.execute('{ slice { zipcode { values sum min max unique { values counts } } } }')
    zipcodes = data['slice']['zipcode']
    assert len(zipcodes['values']) == 41700
    assert zipcodes['sum'] == 2066562337
    assert zipcodes['min'] == 501
    assert zipcodes['max'] == 99950
    assert len(zipcodes['unique']['values']) == 41700
    assert set(zipcodes['unique']['counts']) == {1}
    data = client.execute('{ slice { zipcode { truthy: count count(equal: 0) } } }')
    zipcodes = data['slice']['zipcode']
    assert zipcodes['truthy'] == 41700
    assert zipcodes['count'] == 0
    data = client.execute('{ slice { zipcode { count(equal: 501) } } }')
    zipcodes = data['slice']['zipcode']
    assert zipcodes['count'] == 1
    data = client.execute('{ slice { zipcode { any all } } }')
    zipcodes = data['slice']['zipcode']
    assert zipcodes['any'] and zipcodes['all']
    data = client.execute('{ slice { zipcode { any(greater: 1000) all(greater: 1000) } } }')
    zipcodes = data['slice']['zipcode']
    assert zipcodes['any'] and not zipcodes['all']


def test_floats(client):
    data = client.execute('{ slice { latitude { values sum } longitude { min max } } }')
    latitudes = data['slice']['latitude']
    assert len(latitudes['values']) == 41700
    assert latitudes['sum'] == pytest.approx(1606220.07592)
    longitudes = data['slice']['longitude']
    assert longitudes['min'] == pytest.approx(-174.21333)
    assert longitudes['max'] == pytest.approx(-65.301389)
    data = client.execute('{ slice { latitude { truthy: count count(equal: 0.0) } } }')
    latitudes = data['slice']['latitude']
    assert latitudes['truthy'] == 41700
    assert latitudes['count'] == 0
    data = client.execute('{ slice { latitude { any all } } }')
    latitudes = data['slice']['latitude']
    assert latitudes['any'] and latitudes['all']
    data = client.execute('{ slice { latitude { any(greater: 45.0) all(greater: 45.0) } } }')
    latitudes = data['slice']['latitude']
    assert latitudes['any'] and not latitudes['all']


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
    data = client.execute('{ slice { state { truthy: count count(equal: "") } } }')
    states = data['slice']['state']
    assert states['truthy'] == 41700
    assert states['count'] == 0
    data = client.execute('{ slice { state { count(equal: "CA") } } }')
    states = data['slice']['state']
    assert states['count'] == 2647
    data = client.execute('{ slice { state { any all } } }')
    states = data['slice']['state']
    assert states['any'] and states['all']
    data = client.execute('{ slice { state { any(greater: "CA") all(greater: "CA") } } }')
    states = data['slice']['state']
    assert states['any'] and not states['all']


def test_search(client):
    data = client.execute('{ index search { length } }')
    assert data['index'] == ['zipcode']
    assert data['search']['length'] == 41700
    data = client.execute('{ search(zipcode: {equal: 501}) { slice { zipcode { values } } } }')
    assert data == {'search': {'slice': {'zipcode': {'values': [501]}}}}
    data = client.execute('{ search(zipcode: {notEqual: 501}) { length } }')
    assert data['search']['length'] == 41699

    data = client.execute(
        '''{ search(zipcode: {greaterEqual: 99929})
        { slice { zipcode { values } } } }'''
    )
    assert data == {'search': {'slice': {'zipcode': {'values': [99929, 99950]}}}}
    data = client.execute(
        '''{ search(zipcode: {less: 601})
        { slice { zipcode { values } } } }'''
    )
    assert data == {'search': {'slice': {'zipcode': {'values': [501, 544]}}}}
    data = client.execute(
        '''{ search(zipcode: {greater: 501, lessEqual: 601})
        { slice { zipcode { values } } } }'''
    )
    assert data == {'search': {'slice': {'zipcode': {'values': [544, 601]}}}}

    data = client.execute('{ search(zipcode: {isin: []}) { length } }')
    assert data == {'search': {'length': 0}}
    data = client.execute('{ search(zipcode: {isin: [0]}) { length } }')
    assert data == {'search': {'length': 0}}
    data = client.execute(
        '''{ search(zipcode: {isin: [501, 601]})
        { slice { zipcode { values } } } }'''
    )
    assert data == {'search': {'slice': {'zipcode': {'values': [501, 601]}}}}

    with pytest.raises(ValueError, match="Unknown argument"):
        client.execute('{ search(state: "") { length } }')


def test_filter(client):
    data = client.execute('{ filter { length } }')
    assert data['filter']['length'] == 41700
    data = client.execute('{ filter(city: {equal: "Mountain View"}) { length } }')
    assert data['filter']['length'] == 11
    data = client.execute('{ filter(state: {notEqual: "CA"}) { length } }')
    assert data['filter']['length'] == 39053
    data = client.execute(
        '{ filter(city: {equal: "Mountain View"}, state: {lessEqual: "CA"}) { length } }'
    )
    assert data['filter']['length'] == 7
    data = client.execute('{ filter(state: {equal: null}) {slice {state { values } } } }')
