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
    data = client.execute('{ slice { zipcode { sort desc: sort(reverse: true)} } }')
    zipcodes = data['slice']['zipcode']
    assert zipcodes['sort'][0] == zipcodes['desc'][-1] == 501
    assert zipcodes['sort'][-1] == zipcodes['desc'][0] == 99950
    data = client.execute('{ slice { zipcode { item last: item(index: -1) } } }')
    assert data['slice']['zipcode'] == {'item': 501, 'last': 99950}


def test_floats(client):
    data = client.execute('{ slice { latitude { values sum(exp: 2) } longitude { min max } } }')
    latitudes = data['slice']['latitude']
    assert len(latitudes['values']) == 41700
    assert latitudes['sum'] == pytest.approx(63075443.42831)
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
    data = client.execute('{ slice { latitude { quantile(q: [0.5]) } } }')
    (quantile,) = data['slice']['latitude']['quantile']
    assert quantile == pytest.approx(39.12054)


def test_strings(client):
    data = client.execute(
        '''{ slice {
        state { values unique { values counts } }
        county { unique { length values } }
        city { min max sort(length: 1), desc: sort(reverse: true, length: 1) }
    } }'''
    )
    states = data['slice']['state']
    assert len(states['values']) == 41700
    assert len(states['unique']['values']) == 52
    assert sum(states['unique']['counts']) == 41700
    counties = data['slice']['county']
    assert len(counties['unique']['values']) == counties['unique']['length'] == 1920
    cities = data['slice']['city']
    assert [cities['min']] == cities['sort'] == ['Aaronsburg']
    assert [cities['max']] == cities['desc'] == ['Zwolle']
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
    data = client.execute('{ slice { state { item last: item(index: -1) } } }')
    assert data['slice']['state'] == {'item': 'NY', 'last': "AK"}


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
    data = client.execute('{ filter(state: {equal: null}) { slice {state { values } } } }')
    assert data['filter']['slice']['state']['values'] == []
    data = client.execute('{ exclude { length } }')
    assert data['exclude']['length'] == 41700
    data = client.execute('{ exclude(state: {equal: "CA"}) { length } }')
    assert data['exclude']['length'] == 39053


def test_sort(client):
    with pytest.raises(ValueError, match="is required"):
        client.execute('{ sort { state { values } } }')
    with pytest.raises(ValueError, match="sequence of keys"):
        client.execute('{ sort(names: []) { state { values } } }')
    data = client.execute('{ sort(names: ["state"]) { state { values } } }')
    assert data['sort']['state']['values'][0] == 'AK'
    data = client.execute('{ sort(names: ["state"], reverse: true) { state { values } } }')
    assert data['sort']['state']['values'][0] == 'WY'
    data = client.execute('{ sort(names: ["state"], length: 1) { state { values } } }')
    assert data['sort']['state']['values'] == ['AK']
    data = client.execute(
        '{ sort(names: ["state"], reverse: true, length: 1) { state { values } } }'
    )
    assert data['sort']['state']['values'] == ['WY']
    data = client.execute('{ sort(names: ["state", "county"]) { county { values } } }')
    assert data['sort']['county']['values'][0] == 'Aleutians East'
    data = client.execute(
        '{ sort(names: ["state", "county"], reverse: true, length: 1) { county { values } } }'
    )
    assert data['sort']['county']['values'] == ['Weston']
    data = client.execute('{ sort(names: ["state"], length: 2) { state { values } } }')
    assert data['sort']['state']['values'] == ['AK', 'AK']


def test_groupby(client):
    with pytest.raises(ValueError, match="is required"):
        client.execute('{ groupby { length } }')
    with pytest.raises(ValueError, match="out of range"):
        client.execute('{ groupby(names: []) { length } }')
    data = client.execute('{ groupby(names: ["state"]) { length slice { state { min max } } } }')
    assert len(data['groupby']) == 52
    assert data['groupby'][0]['length'] == 273
    states = data['groupby'][0]['slice']['state']
    assert states['min'] == states['max'] == 'AK'
    data = client.execute(
        '''{ groupby(names: ["state", "county"], reverse: true, length: 3)
        { length slice { state { item } county { item } } } }'''
    )
    assert [group['length'] for group in data['groupby']] == [4, 2, 7]
    tables = [group['slice'] for group in data['groupby']]
    assert [table['state']['item'] for table in tables] == ['WY'] * 3
    assert [table['county']['item'] for table in tables] == ['Weston', 'Washakie', 'Uinta']
