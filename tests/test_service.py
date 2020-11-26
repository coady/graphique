import pytest


def test_slice(client):
    data = client.execute('{ length slice(length: 3) { columns { zipcode { values } } } }')
    assert data == {'length': 41700, 'slice': {'columns': {'zipcode': {'values': [501, 544, 601]}}}}
    data = client.execute('{ slice(offset: 1) { columns { zipcode { values } } }}')
    zipcodes = data['slice']['columns']['zipcode']['values']
    assert zipcodes[0] == 544
    assert len(zipcodes) == 41699
    data = client.execute('{ columns { zipcode { count(notEqual: null) } } }')
    assert data['columns']['zipcode']['count'] == 41700
    data = client.execute('{ columns { zipcode { count(equal: null) } } }')
    assert data['columns']['zipcode']['count'] == 0


def test_ints(client):
    data = client.execute('{ columns { zipcode { values sum mean } } }')
    zipcodes = data['columns']['zipcode']
    assert len(zipcodes['values']) == 41700
    assert zipcodes['sum'] == 2066562337
    assert zipcodes['mean'] == pytest.approx(49557.849808)
    data = client.execute('{ columns { zipcode { values min max unique { values counts } } } }')
    zipcodes = data['columns']['zipcode']
    assert len(zipcodes['values']) == 41700
    assert zipcodes['min'] == 501
    assert zipcodes['max'] == 99950
    assert len(zipcodes['unique']['values']) == 41700
    assert set(zipcodes['unique']['counts']) == {1}
    data = client.execute('{ columns { zipcode { truthy: count count(equal: 0) } } }')
    zipcodes = data['columns']['zipcode']
    assert zipcodes['truthy'] == 41700
    assert zipcodes['count'] == 0
    data = client.execute('{ columns { zipcode { count(equal: 501) } } }')
    zipcodes = data['columns']['zipcode']
    assert zipcodes['count'] == 1
    data = client.execute('{ columns { zipcode { sort desc: sort(reverse: true)} } }')
    zipcodes = data['columns']['zipcode']
    assert zipcodes['sort'][0] == zipcodes['desc'][-1] == 501
    assert zipcodes['sort'][-1] == zipcodes['desc'][0] == 99950
    data = client.execute('{ columns { zipcode(add: "zipcode") { min } } }')
    assert data == {'columns': {'zipcode': {'min': 1002}}}
    data = client.execute('{ columns { zipcode(subtract: "zipcode") { unique { values } } } }')
    assert data == {'columns': {'zipcode': {'unique': {'values': [0]}}}}


def test_floats(client):
    data = client.execute('{ columns { latitude { values sum mean } } }')
    latitudes = data['columns']['latitude']
    assert len(latitudes['values']) == 41700
    assert latitudes['sum'] == pytest.approx(1606220.07592)
    assert latitudes['mean'] == pytest.approx(38.518467)
    data = client.execute('{ columns { longitude { min max } } }')
    longitudes = data['columns']['longitude']
    assert longitudes['min'] == pytest.approx(-174.21333)
    assert longitudes['max'] == pytest.approx(-65.301389)
    data = client.execute('{ columns { latitude { truthy: count count(equal: 0.0) } } }')
    latitudes = data['columns']['latitude']
    assert latitudes['truthy'] == 41700
    assert latitudes['count'] == 0
    data = client.execute('{ columns { latitude { quantile(q: [0.5]) } } }')
    (quantile,) = data['columns']['latitude']['quantile']
    assert quantile == pytest.approx(39.12054)
    data = client.execute('{ columns { latitude(multiply: "longitude") { min } } }')
    assert data['columns']['latitude']['min'] == pytest.approx(-11389.408478)
    data = client.execute('{ columns { latitude(minimum: "longitude") { min } } }')
    assert data['columns']['latitude']['min'] == pytest.approx(-174.213333)
    data = client.execute('{ columns { latitude(maximum: "longitude") { max } } }')
    assert data['columns']['latitude']['max'] == pytest.approx(71.290556)
    data = client.execute('{ columns { longitude { absolute { min } } } }')
    assert data['columns']['longitude']['absolute']['min'] > 65


def test_strings(client):
    data = client.execute(
        '''{ columns {
        state { values unique { values counts } }
        county { unique { length values } }
        city { min max sort(length: 1), desc: sort(reverse: true, length: 1) }
    } }'''
    )
    states = data['columns']['state']
    assert len(states['values']) == 41700
    assert len(states['unique']['values']) == 52
    assert sum(states['unique']['counts']) == 41700
    counties = data['columns']['county']
    assert len(counties['unique']['values']) == counties['unique']['length'] == 1920
    cities = data['columns']['city']
    assert [cities['min']] == cities['sort'] == ['Aaronsburg']
    assert [cities['max']] == cities['desc'] == ['Zwolle']
    data = client.execute('{ columns { state { truthy: count count(equal: "") } } }')
    states = data['columns']['state']
    assert states['truthy'] == 41700
    assert states['count'] == 0
    data = client.execute('{ columns { state { count(equal: "CA") } } }')
    assert data == {'columns': {'state': {'count': 2647}}}
    data = client.execute('{ columns { state { count(utf8Lower: {equal: "ca"}) } } }')
    assert data == {'columns': {'state': {'count': 2647}}}
    data = client.execute('{ columns { state { binaryLength { unique { values } } } } }')
    assert data['columns']['state']['binaryLength']['unique']['values'] == [2]
    data = client.execute('{ columns { state { utf8Lower { values } } } }')
    assert 'ca' in data['columns']['state']['utf8Lower']['values']
    data = client.execute('{ columns { city { utf8Upper { values } } } }')
    assert 'MOUNTAIN VIEW' in data['columns']['city']['utf8Upper']['values']
    data = client.execute(
        '''{ filter(query: {state: {equal: "CA"}}) {
        apply(city: {binaryLength: true, alias: "size"}) {
        column(name: "size") { ... on IntColumn { max } } } } }'''
    )
    assert data == {'filter': {'apply': {'column': {'max': 24}}}}


def test_search(client):
    data = client.execute('{ index names search { length } }')
    assert data['index'] == ['zipcode']
    assert set(data['names']) == {'latitude', 'longitude', 'state', 'city', 'county', 'zipcode'}
    assert data['search']['length'] == 41700
    data = client.execute('{ search(zipcode: {equal: 501}) { columns { zipcode { values } } } }')
    assert data == {'search': {'columns': {'zipcode': {'values': [501]}}}}
    data = client.execute('{ search(zipcode: {notEqual: 501}) { length } }')
    assert data['search']['length'] == 41699

    data = client.execute(
        '{ search(zipcode: {greaterEqual: 99929}) { columns { zipcode { values } } } }'
    )
    assert data == {'search': {'columns': {'zipcode': {'values': [99929, 99950]}}}}
    data = client.execute('{ search(zipcode: {less: 601}) { columns { zipcode { values } } } }')
    assert data == {'search': {'columns': {'zipcode': {'values': [501, 544]}}}}
    data = client.execute(
        '{ search(zipcode: {greater: 501, lessEqual: 601}) { columns { zipcode { values } } } }'
    )
    assert data == {'search': {'columns': {'zipcode': {'values': [544, 601]}}}}

    data = client.execute('{ search(zipcode: {isIn: []}) { length } }')
    assert data == {'search': {'length': 0}}
    data = client.execute('{ search(zipcode: {isIn: [0]}) { length } }')
    assert data == {'search': {'length': 0}}
    data = client.execute(
        '{ search(zipcode: {isIn: [501, 601]}) { columns { zipcode { values } } } }'
    )
    assert data == {'search': {'columns': {'zipcode': {'values': [501, 601]}}}}


def test_filter(client):
    data = client.execute('{ filter(query: {}) { length } }')
    assert data['filter']['length'] == 41700
    data = client.execute('{ filter(query: {city: {equal: "Mountain View"}}) { length } }')
    assert data['filter']['length'] == 11
    data = client.execute('{ filter(query: {state: {notEqual: "CA"}}) { length } }')
    assert data['filter']['length'] == 39053
    data = client.execute(
        '{ filter(query: {city: {equal: "Mountain View"}, state: {lessEqual: "CA"}}) { length } }'
    )
    assert data['filter']['length'] == 7
    data = client.execute(
        '{ filter(query: {state: {equal: null}}) { columns { state { values } } } }'
    )
    assert data['filter']['columns']['state']['values'] == []
    data = client.execute('{ filter(query: {}, invert: true) { length } }')
    assert data['filter']['length'] == 41700
    data = client.execute('{ filter(query: {state: {equal: "CA"}}, invert: true) { length } }')
    assert data['filter']['length'] == 39053
    data = client.execute('{ filter(query: {city: {matchSubstring: "Mountain"}}) { length } }')
    assert data['filter']['length'] == 88
    data = client.execute(
        '{ filter(query: {city: {utf8Lower: {matchSubstring: "mountain"}}}) { length } }'
    )
    assert data['filter']['length'] == 88
    data = client.execute(
        '{ filter(query: {city: {utf8Upper: {matchSubstring: "MOUNTAIN"}}}) { length } }'
    )
    assert data['filter']['length'] == 88
    data = client.execute('{ filter(query: {county: {apply: {equal: "city"}}}) { length } }')
    assert data['filter']['length'] == 2805
    data = client.execute('{ filter(query: {city: {utf8IsLower: true}}) { length } }')
    assert data['filter']['length'] == 0
    data = client.execute('{ filter(query: {city: {utf8IsTitle: true}}) { length } }')
    assert data['filter']['length'] == 41700
    data = client.execute('{ filter(query: {city: {}}) { length } }')
    assert data['filter']['length'] == 41700
    data = client.execute(
        '{ filter(query: {zipcode: {apply: {add: "zipcode"}, equal: 1002}}) { length } }'
    )
    assert data['filter']['length'] == 1
    data = client.execute(
        '{ filter(query: {zipcode: {apply: {subtract: "zipcode"}, equal: 0}}) { length } }'
    )
    assert data['filter']['length'] == 41700
    data = client.execute(
        '{ filter(query: {latitude: {apply: {multiply: "longitude"}, greater: 0}}) { length } }'
    )
    assert data['filter']['length'] == 0
    data = client.execute('{ filter(query: {longitude: {absolute: {less: 66}}}) { length } }')
    assert data['filter']['length'] == 30


def test_sort(client):
    with pytest.raises(ValueError, match="is required"):
        client.execute('{ sort { columns { state { values } } } }')
    data = client.execute('{ sort(by: ["state"]) { columns { state { values } } } }')
    assert data['sort']['columns']['state']['values'][0] == 'AK'
    data = client.execute('{ sort(by: ["state"], reverse: true) { columns { state { values } } } }')
    assert data['sort']['columns']['state']['values'][0] == 'WY'
    data = client.execute('{ sort(by: ["state"], length: 1) { columns { state { values } } } }')
    assert data['sort']['columns']['state']['values'] == ['AK']
    data = client.execute(
        '{ sort(by: ["state"], reverse: true, length: 1) { columns { state { values } } } }'
    )
    assert data['sort']['columns']['state']['values'] == ['WY']
    data = client.execute('{ sort(by: ["state", "county"]) { columns { county { values } } } }')
    assert data['sort']['columns']['county']['values'][0] == 'Aleutians East'
    data = client.execute(
        '''{ sort(by: ["state", "county"], reverse: true, length: 1)
        { columns { county { values } } } }'''
    )
    assert data['sort']['columns']['county']['values'] == ['Weston']
    data = client.execute('{ sort(by: ["state"], length: 2) { columns { state { values } } } }')
    assert data['sort']['columns']['state']['values'] == ['AK', 'AK']


def test_group(client):
    with pytest.raises(ValueError, match="is required"):
        client.execute('{ group { length } }')
    data = client.execute(
        '''{ group(by: ["state"]) { length tables { length columns { state { min max } } }
        aggregate(count: "c") { column(name: "c") { ... on IntColumn { values } } } } }'''
    )
    assert len(data['group']['tables']) == data['group']['length'] == 52
    table = data['group']['tables'][0]
    assert table['length'] == data['group']['aggregate']['column']['values'][0] == 2205
    states = table['columns']['state']
    assert states['min'] == states['max'] == 'NY'
    data = client.execute(
        '''{ group(by: ["state", "county"], reverse: true, length: 3)
        { tables { length row { state county } }
        aggregate(first: [{name: "city", alias: "f"}], last: [{name: "city", alias: "l"}]) {
        f: column(name: "f") { ... on StringColumn { values } }
        l: column(name: "l") { ... on StringColumn { values } } } } }'''
    )
    tables = data['group']['tables']
    assert [group['length'] for group in tables] == [9, 3, 1]
    rows = [group['row'] for group in tables]
    assert [row['state'] for row in rows] == ['AK'] * 3
    counties = [row['county'] for row in rows]
    assert counties == ['Prince Wales Ketchikan', 'Ketchikan Gateway', 'Sitka']
    assert data['group']['aggregate']['f']['values'] == ['Meyers Chuck', 'Ketchikan', 'Sitka']
    assert data['group']['aggregate']['l']['values'] == ['Point Baker', 'Ketchikan', 'Sitka']
    data = client.execute(
        '''{ group(by: ["state", "county"], reverse: true) { filter(greaterEqual: 200) {
        tables { length }
        aggregate(min: [{name: "city", alias: "min"}], max: [{name: "city", alias: "max"}]) {
        min: column(name: "min") { ... on StringColumn { values } }
        max: column(name: "max") { ... on StringColumn { values } } } } } }'''
    )
    assert [row['length'] for row in data['group']['filter']['tables']] == [525, 242, 219, 284]
    agg = data['group']['filter']['aggregate']
    assert agg['min']['values'] == ['Acton', 'Alief', 'Alsip', 'Naval Anacost Annex']
    assert agg['max']['values'] == ['Woodland Hills', 'Webster', 'Worth', 'Washington Navy Yard']
    data = client.execute(
        '''{ group(by: ["state", "county"]) { sort(reverse: true, length: 4) {
        tables { length }
        aggregate(sum: [{name: "latitude"}], mean: [{name: "longitude"}]) {
        columns { latitude { values } longitude { values } }
        column(name: "zipcode") { ... on ListColumn { count { values } } } } } } }'''
    )
    counts = [row['length'] for row in data['group']['sort']['tables']]
    agg = data['group']['sort']['aggregate']
    assert counts == agg['column']['count']['values'] == [525, 284, 242, 219]
    assert all(latitude > 1000 for latitude in agg['columns']['latitude']['values'])
    assert all(77 > longitude > -119 for longitude in agg['columns']['longitude']['values'])
    data = client.execute(
        '''{ sc: group(by: ["state", "county"]) { length }
        cs: group(by: ["county", "state"]) { length } }'''
    )
    assert data['sc']['length'] == data['cs']['length'] == 3216
    data = client.execute(
        '''{ sc: group(by: ["state", "county"]) { length }
        cs: group(by: ["county", "state"]) { length } }'''
    )
    assert data['sc']['length'] == data['cs']['length'] == 3216


def test_unique(client):
    with pytest.raises(ValueError, match="is required"):
        client.execute('{ unique { length } }')
    with pytest.raises(ValueError, match="out of range"):
        client.execute('{ unique(by: []) { length } }')
    assert client.execute('{ unique(by: ["state"]) { length } }') == {'unique': {'length': 52}}
    data = client.execute('{ unique(by: ["state"]) { length columns { zipcode { min max } } } }')
    assert data == {'unique': {'length': 52, 'columns': {'zipcode': {'min': 501, 'max': 99501}}}}
    data = client.execute(
        '{ unique(by: ["state"], reverse: true) { length columns { zipcode { min max } } } }'
    )
    assert data == {'unique': {'length': 52, 'columns': {'zipcode': {'min': 988, 'max': 99950}}}}
    data = client.execute(
        '{ unique(by: ["state", "county"]) { length columns { zipcode { min max } } } }'
    )
    assert data == {'unique': {'length': 3216, 'columns': {'zipcode': {'min': 501, 'max': 99903}}}}
    data = client.execute(
        '''{ unique(by: ["state"], count: "counts")
        { column(name: "counts") { ... on LongColumn { mean } } } }'''
    )
    assert data['unique']['column']['mean'] == pytest.approx(801.923076)
    data = client.execute(
        '''{ group(by: ["state"]) { aggregate(unique: {name: "city"}) {
        column(name: "city") { ... on ListColumn { count { values } } } } } }'''
    )
    counts = data['group']['aggregate']['column']['count']['values']
    assert sum(counts) == 29734
    data = client.execute(
        '''{ group(by: ["state"]) { aggregate(unique: {name: "city", count: true}) {
        column(name: "city") { ... on IntColumn { values } } } } }'''
    )
    assert data['group']['aggregate']['column']['values'] == counts


def test_rows(client):
    with pytest.raises(ValueError, match="out of bounds"):
        client.execute('{ row(index: 100000) { zipcode } }')
    data = client.execute('{ row { state } }')
    assert data == {'row': {'state': 'NY'}}
    data = client.execute('{ row(index: -1) { state } }')
    assert data == {'row': {'state': 'AK'}}
    data = client.execute('{ min(by: ["latitude"]) { row { state } } }')
    assert data == {'min': {'row': {'state': 'PR'}}}
    data = client.execute('{ max(by: ["latitude", "longitude"]) { row { state } } }')
    assert data == {'max': {'row': {'state': 'AK'}}}
