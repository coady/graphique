import pytest


def test_slice(client):
    data = client.execute('{ count slice(limit: 3) { columns { zipcode { values } } } }')
    assert data == {'count': 41700, 'slice': {'columns': {'zipcode': {'values': [501, 544, 601]}}}}
    data = client.execute('{ slice(offset: 1) { columns { zipcode { values } } } }')
    zipcodes = data['slice']['columns']['zipcode']['values']
    assert zipcodes[0] == 544
    assert len(zipcodes) == 41699
    data = client.execute('{ columns { zipcode { count } } }')
    assert data['columns']['zipcode']['count'] == 41700
    data = client.execute('{ columns { zipcode { count(mode: "only_null") } } }')
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
    data = client.execute('{ columns { latitude { quantile(q: [0.5]) } } }')
    (quantile,) = data['columns']['latitude']['quantile']
    assert quantile == pytest.approx(39.12054)
    data = client.execute(
        """{ scan(columns: {alias: "latitude", elementWise: {min: [{name: "latitude"}, {name: "longitude"}]}}) {
        columns { latitude { min } } } }"""
    )
    assert data == {'scan': {'columns': {'latitude': {'min': pytest.approx(-174.213333)}}}}
    data = client.execute(
        """{project(columns: {alias: "l", numeric: {bucket: {name: "latitude"}, buckets: [40, 50]}}) {
        column(name: "l") { ... on IntColumn { unique { values } } } } }"""
    )
    assert data == {"project": {"column": {"unique": {"values": [0, None]}}}}
    data = client.execute(
        '{ scan(columns: {alias: "latitude", log: {logb: [{name: "latitude"}, {value: 3}]}}) { row { latitude } } }'
    )
    assert data == {'scan': {'row': {'latitude': pytest.approx(3.376188)}}}
    data = client.execute(
        '{ scan(columns: {alias: "latitude", rounding: {round: {name: "latitude"}}}) {row { latitude } } }'
    )
    assert data == {'scan': {'row': {'latitude': 41.0}}}
    data = client.execute(
        '{ scan(columns: {alias: "latitude", rounding: {round: {name: "latitude"}, multiple: 2.0}}) {row { latitude } } }'
    )
    assert data == {'scan': {'row': {'latitude': 40.0}}}
    data = client.execute(
        '{ scan(columns: {alias: "latitude", trig: {sin: {name: "latitude"}}}) {row { latitude } } }'
    )
    assert data == {'scan': {'row': {'latitude': pytest.approx(0.02273553)}}}
    data = client.execute('{ scan(filter: {isFinite: {name: "longitude"}}) { count } }')
    assert data == {'scan': {'count': 41700}}
    data = client.execute('{ column(name: "latitude", cast: "int32", safe: false) { type } }')
    assert data == {'column': {'type': 'int32'}}


def test_strings(client):
    data = client.execute("""{ columns {
        state { values unique { values counts } countDistinct }
        county { unique { values } }
        city { min max }
    } }""")
    states = data['columns']['state']
    assert len(states['values']) == 41700
    assert len(states['unique']['values']) == states['countDistinct'] == 52
    assert sum(states['unique']['counts']) == 41700
    assert data['columns']['city'] == {'min': 'Aaronsburg', 'max': 'Zwolle'}
    data = client.execute("""{ filter(state: {eq: "CA"}) {
        scan(filter: {gt: [{utf8: {length: {name: "city"}}}, {value: 23}]}) { count } } }""")
    assert data == {'filter': {'scan': {'count': 1}}}
    data = client.execute(
        '{ scan(columns: {utf8: {swapcase: {name: "city"}}, alias: "city"}) { row { city } } }'
    )
    assert data == {'scan': {'row': {'city': 'hOLTSVILLE'}}}
    data = client.execute(
        '{ scan(columns: {utf8: {capitalize: {name: "state"}}, alias: "state"}) { row { state } } }'
    )
    assert data == {'scan': {'row': {'state': 'Ny'}}}
    data = client.execute('{ scan(filter: {utf8: {isLower: {name: "city"}}}) { count } }')
    assert data == {'scan': {'count': 0}}
    data = client.execute('{ scan(filter: {utf8: {isTitle: {name: "city"}}}) { count } }')
    assert data == {'scan': {'count': 41700}}
    data = client.execute(
        """{ scan(columns: {alias: "city", substring: {match: {name: "city"}, pattern: "Mountain"}})
        { scan(filter: {name: "city"}) { count } } }"""
    )
    assert data == {'scan': {'scan': {'count': 88}}}
    data = client.execute(
        """{ scan(filter: {substring: {match: {name: "city"}, pattern: "mountain", ignoreCase: true}})
        { count } }"""
    )
    assert data == {'scan': {'count': 88}}
    data = client.execute(
        """{ scan(filter: {substring: {match: {name: "city"}, pattern: "^Mountain", regex: true}})
        { count } }"""
    )
    assert data == {'scan': {'count': 42}}
    data = client.execute(
        """{ project(columns: {alias: "has", isin: [{name: "state"}, {value: ["CA", "OR"]}]})
        { column(name: "has") { ... on BooleanColumn { unique { values } } } } }"""
    )
    assert data == {'project': {'column': {'unique': {'values': [False, True]}}}}


def test_string_methods(client):
    data = client.execute(
        """{ scan(columns: {alias: "split", substring: {split: {name: "city"}, pattern: "-", maxSplits: 1}}) {
        column(name: "split") { type } } }"""
    )
    assert data == {'scan': {'column': {'type': 'list<item: string>'}}}
    data = client.execute("""{ scan(columns: {alias: "split", substring: {split: {name: "city"}}}) {
        column(name: "split") { type } } }""")
    assert data == {'scan': {'column': {'type': 'list<item: string>'}}}
    data = client.execute(
        """{ scan(columns: {alias: "state", utf8: {trim: {name: "state"}, characters: "C"}}) {
        columns { state { values } } } }"""
    )
    states = data['scan']['columns']['state']['values']
    assert 'CA' not in states and 'A' in states
    data = client.execute(
        '{ scan(columns: {alias: "state", utf8: {ltrim: {name: "state"}}}) { count } }'
    )
    assert data == {'scan': {'count': 41700}}
    data = client.execute(
        """{ scan(columns: {alias: "state", utf8: {center: {name: "state"}, width: 4, padding: "_"}})
        { row { state } } }"""
    )
    assert data == {'scan': {'row': {'state': '_NY_'}}}
    data = client.execute(
        """{ scan(columns: {alias: "state", utf8: {replaceSlice: {name: "state"}, start: 0, stop: 2, replacement: ""}})
        { columns { state { unique { values } } } } }"""
    )
    assert data == {'scan': {'columns': {'state': {'unique': {'values': ['']}}}}}
    data = client.execute(
        '{ scan(columns: {alias: "state", utf8: {sliceCodeunits: {name: "state"}, start: 0, stop: 1}}) { row { state } } }'
    )
    assert data == {'scan': {'row': {'state': 'N'}}}
    data = client.execute(
        """{ scan(columns: {alias: "state", substring: {replace: {name: "state"}, pattern: "C", replacement: "A"}})
        { columns { state { values } } } }"""
    )
    assert 'AA' in data['scan']['columns']['state']['values']


def test_search(client):
    data = client.execute('{ filter { count } }')
    assert data == {'filter': {'count': 41700}}
    data = client.execute('{ filter(zipcode: {eq: 501}) { columns { zipcode { values } } } }')
    assert data == {'filter': {'columns': {'zipcode': {'values': [501]}}}}
    data = client.execute('{ filter(zipcode: {ne: 501}) { count } }')
    assert data['filter']['count'] == 41699

    data = client.execute('{ filter(zipcode: {ge: 99929}) { columns { zipcode { values } } } }')
    assert data == {'filter': {'columns': {'zipcode': {'values': [99929, 99950]}}}}
    data = client.execute('{ filter(zipcode: {lt: 601}) { columns { zipcode { values } } } }')
    assert data == {'filter': {'columns': {'zipcode': {'values': [501, 544]}}}}
    data = client.execute(
        '{ filter(zipcode: {gt: 501, le: 601}) { columns { zipcode { values } } } }'
    )
    assert data == {'filter': {'columns': {'zipcode': {'values': [544, 601]}}}}

    data = client.execute('{ filter(zipcode: {eq: []}) { count } }')
    assert data == {'filter': {'count': 0}}
    data = client.execute('{ filter(zipcode: {eq: [0]}) { count } }')
    assert data == {'filter': {'count': 0}}
    data = client.execute(
        '{ filter(zipcode: {eq: [501, 601]}) { columns { zipcode { values } } } }'
    )
    assert data == {'filter': {'columns': {'zipcode': {'values': [501, 601]}}}}


def test_filter(client):
    data = client.execute('{ filter { count } }')
    assert data['filter']['count'] == 41700
    data = client.execute('{ filter(city: {eq: "Mountain View"}) { count } }')
    assert data['filter']['count'] == 11
    data = client.execute('{ filter(state: {ne: "CA"}) { count } }')
    assert data['filter']['count'] == 39053
    data = client.execute('{ filter(city: {eq: "Mountain View"}, state: {le: "CA"}) { count } }')
    assert data['filter']['count'] == 7
    data = client.execute('{ filter(state: {eq: null}) { columns { state { values } } } }')
    assert data['filter']['columns']['state']['values'] == []
    data = client.execute(
        '{ filter(where: {le: [{numeric: {abs: {name: "longitude"}}}, {value: 66}]}) { count } }'
    )
    assert data['filter']['count'] == 30
    with pytest.raises(ValueError, match="optional, not nullable"):
        client.execute('{ filter(city: {le: null}) { count } }')


def test_scan(client):
    data = client.execute('{ scan(filter: {eq: [{name: "county"}, {name: "city"}]}) { count } }')
    assert data['scan']['count'] == 2805
    data = client.execute("""{ scan(filter: {or: [{eq: [{name: "state"}, {name: "county"}]},
        {eq: [{name: "county"}, {name: "city"}]}]}) { count } }""")
    assert data['scan']['count'] == 2805
    data = client.execute(
        """{ scan(columns: {alias: "zipcode", add: [{name: "zipcode"}, {name: "zipcode"}]})
        { columns { zipcode { min } } } }"""
    )
    assert data['scan']['columns']['zipcode']['min'] == 1002
    data = client.execute(
        """{ scan(columns: {alias: "zipcode", subtract: [{name: "zipcode"}, {name: "zipcode"}]})
        { columns { zipcode { unique { values } } } } }"""
    )
    assert data['scan']['columns']['zipcode']['unique']['values'] == [0]
    data = client.execute(
        """{ scan(columns: {alias: "product", multiply: [{name: "latitude"}, {name: "longitude"}]})
        { filter(where: {gt: [{name: "product"}, {value: 0}]}) { count } } }"""
    )
    assert data['scan']['filter']['count'] == 0
    data = client.execute(
        '{ scan(columns: {name: "zipcode", cast: "float"}) { column(name: "zipcode") { type } } }'
    )
    assert data['scan']['column']['type'] == 'float'
    data = client.execute(
        '{ scan(filter: {inv: {eq: [{name: "state"}, {value: "CA"}]}}) { count } }'
    )
    assert data == {'scan': {'count': 39053}}
    data = client.execute(
        '{ scan(columns: {name: "latitude", cast: "int32", safe: false}) { column(name: "latitude") { type } } }'
    )
    assert data == {'scan': {'column': {'type': 'int32'}}}
    data = client.execute(
        """{ scan(columns: {alias: "longitude", elementWise: {max: [{name: "longitude"}, {name: "latitude"}]}})
        { columns { longitude { min } } } }"""
    )
    assert data['scan']['columns']['longitude']['min'] == pytest.approx(17.963333)
    data = client.execute(
        """{ scan(columns: {alias: "latitude", elementWise: {min: [{name: "longitude"}, {name: "latitude"}]}})
        { columns { latitude { max } } } }"""
    )
    assert data['scan']['columns']['latitude']['max'] == pytest.approx(-65.301389)
    data = client.execute(
        """{ scan(columns: {alias: "state", elementWise: {min: [{name: "state"}, {name: "county"}], skipNulls: false}})
        { columns { state { values } } } }"""
    )
    assert data['scan']['columns']['state']['values'][0] == 'NY'


def test_project(client):
    assert client.execute('{ project(columns: []) { type } }')
    assert client.execute('{ project(columns: [{}]) { optional { type } } }')
    with pytest.raises(ValueError, match="conflict"):
        client.execute('{ project(columns: [{name: "state", value: ""}]) { type } }')
    data = client.execute("""{ project(columns: {alias: "zipcode", numeric: {cumsum: {name: "zipcode"}}}) {
        columns { zipcode { value(index: -1) } } } }""")
    assert data == {'project': {'columns': {'zipcode': {'value': 2066562337}}}}
    data = client.execute("""{ project(columns: {alias: "state", cummin: {name: "state"}}) {
        columns { state { value(index: -1) } } } }""")
    assert data == {'project': {'columns': {'state': {'value': "AK"}}}}
    data = client.execute("""{ project(columns: {alias: "idx", denseRank: {name: "state"}}) {
        column(name: "idx") { ... on LongColumn { min max } } } }""")
    assert data == {'project': {'column': {'min': 0, 'max': 51}}}


def test_order(client):
    with pytest.raises(ValueError, match="is required"):
        client.execute('{ order { columns { state { values } } } }')
    data = client.execute('{ order(by: ["state"]) { columns { state { values } } } }')
    assert data['order']['columns']['state']['values'][0] == 'AK'
    data = client.execute('{ order(by: "-state") { columns { state { values } } } }')
    assert data['order']['columns']['state']['values'][0] == 'WY'
    data = client.execute('{ order(by: ["state"], limit: 1) { columns { state { values } } } }')
    assert data['order']['columns']['state']['values'] == ['AK']
    data = client.execute('{ order(by: "-state", limit: 1) { columns { state { values } } } }')
    assert data['order']['columns']['state']['values'] == ['WY']
    data = client.execute('{ order(by: ["state", "county"]) { columns { county { values } } } }')
    assert data['order']['columns']['county']['values'][0] == 'Aleutians East'
    data = client.execute(
        """{ order(by: ["-state", "-county"], limit: 1) { columns { county { values } } } }"""
    )
    assert data['order']['columns']['county']['values'] == ['Weston']
    data = client.execute('{ order(by: ["state"], limit: 2) { columns { state { values } } } }')
    assert data['order']['columns']['state']['values'] == ['AK', 'AK']
    data = client.execute("""{ group(by: ["state"], aggregate: {collect: {name: "county"}}) {
        order(by: "state") {
        project(columns: {array: {value: {array: {sort: {name: "county"}}}, offset: 0}, alias: "county"}) {
        row { state county } } } } }""")
    assert data['group']['order']['project']['row'] == {'state': 'AK', 'county': 'Aleutians East'}
    data = client.execute("""{ project(columns: {alias: "index", rowNumber: null}) {
        group(by: "state", aggregate: {first: {name: "index"}, collect: {name: "city"}}) {
        order(by: "index") {
        project(columns: {alias: "city", array: {slice: {name: "city"}, limit: 1}}) {
        row { state } } } } } }""")
    assert data['project']['group']['order']['project']['row'] == {'state': 'NY'}


def test_group(client):
    with pytest.raises(ValueError, match="cannot represent"):
        client.execute("""{ group(by: "state", aggregate: {collect: {name: "city"}}) {
            row { city } } }""")
    data = client.execute("""{ group(by: [], counts: "c", aggregate: {max: {alias: "z", name: "zipcode"}}) {
        c: column(name: "c") { ... on LongColumn { values } }
        z: column(name: "z") { ... on IntColumn { values } } } }""")
    assert data == {'group': {'c': {'values': [41700]}, 'z': {'values': [99950]}}}
    data = client.execute("""{ group(by: "state", aggregate: {collect: {name: "county", distinct: true}}) {
        columns { state { values } }
        c: column(name: "county") { ... on ListColumn { values { count } } } } }""")
    index = data['group']['columns']['state']['values'].index('NY')
    assert data['group']['c']['values'][index] == {'count': 62}


def test_unnest(client):
    data = client.execute("""{ group(by: "state", aggregate: {collect: {name: "city"}}) {
        unnest(name: "city") { columns { city { type } } } } }""")
    assert data == {'group': {'unnest': {'columns': {'city': {'type': 'string'}}}}}
    data = client.execute("""{ group(by: "state", aggregate: {collect: {name: "city"}}) {
        unnest(name: "city", offset: "idx") { column(name: "idx") { type } } } }""")
    assert data == {'group': {'unnest': {'column': {'type': 'int64'}}}}
    data = client.execute("""{ group(by: "state", aggregate: {collect: {name: "city"}}) {
         unnest(name: "city", rowNumber: "idx") { column(name: "idx") { 
        ... on LongColumn { values } } } } }""")
    assert set(data['group']['unnest']['column']['values']) == set(range(52))


def test_rows(client):
    with pytest.raises(ValueError, match="out of bounds"):
        client.execute('{ row(index: 100000) { zipcode } }')
    data = client.execute('{ row { state } }')
    assert data == {'row': {'state': 'NY'}}
    data = client.execute('{ row(index: -1) { state } }')
    assert data == {'row': {'state': 'AK'}}
