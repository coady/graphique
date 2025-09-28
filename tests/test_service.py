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
    data = client.execute('{ take(indices: [0]) { row { zipcode } } }')
    assert data == {'take': {'row': {'zipcode': 501}}}
    data = client.execute('{ any many: any(limit: 50000)}')
    assert data == {'any': True, 'many': False}
    data = client.execute('{ slice { count c: count } }')
    assert data == {'slice': {'count': 41700, 'c': 41700}}


def test_ints(client):
    data = client.execute('{ columns { zipcode { values sum mean } } }')
    zipcodes = data['columns']['zipcode']
    assert len(zipcodes['values']) == 41700
    assert zipcodes['sum'] == 2066562337
    assert zipcodes['mean'] == pytest.approx(49557.849808)
    data = client.execute('{ columns { zipcode { values min max distinct { values counts } } } }')
    zipcodes = data['columns']['zipcode']
    assert len(zipcodes['values']) == 41700
    assert zipcodes['min'] == 501
    assert zipcodes['max'] == 99950
    assert len(zipcodes['distinct']['values']) == 41700
    assert set(zipcodes['distinct']['counts']) == {1}


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
    data = client.execute('{ columns { latitude { quantile(q: 0.5) } } }')
    assert data == {'columns': {'latitude': {'quantile': [pytest.approx(39.12054)]}}}
    data = client.execute("""{project(columns: {alias: "l", numeric: {bucket: {name: "latitude"}, buckets: [40, 50]}})
        { column(name: "l") { ... on IntColumn { distinct { values } } } } }""")
    assert set(data['project']['column']['distinct']['values']) == {0, None}
    data = client.execute("""{ project(columns: {alias: "latitude", numeric: {log: [{name: "latitude"}, {value: 3}]}}) {
        row { latitude } } }""")
    assert data == {'project': {'row': {'latitude': pytest.approx(3.376188)}}}
    data = client.execute("""{ project(columns: {alias: "latitude", numeric: {round: {name: "latitude"}}}) {
        row { latitude } } }""")
    assert data == {'project': {'row': {'latitude': 41.0}}}
    data = client.execute("""{ project(columns: {alias: "latitude", numeric: {round: [{name: "latitude"}, {value: 1}]}}) {
        row { latitude } } }""")
    assert data == {'project': {'row': {'latitude': 40.8}}}
    data = client.execute("""{ project(columns: {alias: "latitude", numeric: {sin: {name: "latitude"}}}) {
        row { latitude } } }""")
    assert data == {'project': {'row': {'latitude': pytest.approx(0.02273553)}}}
    data = client.execute('{ filter(where: {numeric: {isinf: {name: "longitude"}}}) { count } }')
    assert data == {'filter': {'count': 0}}
    data = client.execute('{ column(name: "latitude", cast: "int32", try: true) { type } }')
    assert data == {'column': {'type': 'int32'}}


def test_strings(client):
    data = client.execute("""{ columns {
        state { values nunique distinct { values counts } }
        county { distinct { values } }
        city { min max }
    } }""")
    states = data['columns']['state']
    assert len(states['values']) == 41700
    assert len(states['distinct']['values']) == states['nunique'] == 52
    assert sum(states['distinct']['counts']) == 41700
    assert data['columns']['city'] == {'min': 'Aaronsburg', 'max': 'Zwolle'}
    data = client.execute("""{ filter(state: {eq: "CA"}, where: {gt: [{string: {length: {name: "city"}}}, {value: 23}]})
        { count } }""")
    assert data == {'filter': {'count': 1}}
    data = client.execute("""{ project(columns: {string: {capitalize: {name: "state"}}, alias: "state"})
        { row { state } } }""")
    assert data == {'project': {'row': {'state': 'Ny'}}}
    data = client.execute("""{ project(columns: {alias: "city", string: {contains: [{name: "city"}, {value: "Mountain"}]}})
        { filter(where: {name: "city"}) { count } } }""")
    assert data == {'project': {'filter': {'count': 88}}}
    data = client.execute("""{ filter(where: {string: {reSearch: [{name: "city"}, {value: "^Mountain"}]}})
        { count } }""")
    assert data == {'filter': {'count': 42}}
    data = client.execute(
        """{ project(columns: {alias: "has", isin: [{name: "state"}, {value: ["CA", "OR"]}]})
        { column(name: "has") { ... on BooleanColumn { distinct { values } } } } }"""
    )
    assert set(data['project']['column']['distinct']['values']) == {False, True}
    data = client.execute('{ columns { state { quantile } } }')
    assert data == {'columns': {'state': {'quantile': ['MS']}}}


def test_string_methods(client):
    data = client.execute("""{ project(columns: {alias: "split", string: {reSplit: [{name: "city"}, {value: "-"}]}})
        { column(name: "split") { type } } }""")
    assert data == {'project': {'column': {'type': 'array<string>'}}}
    data = client.execute("""{ project(columns: {alias: "split", string: {split: [{name: "city"}, {value: " "}]}})
        { column(name: "split") { type } } }""")
    assert data == {'project': {'column': {'type': 'array<string>'}}}
    data = client.execute("""{ project(columns: {alias: "state", string: {lstrip: {name: "state"}}})
        { count } }""")
    assert data == {'project': {'count': 41700}}
    data = client.execute("""{ project(columns: {alias: "state", string: {replace: [{name: "state"}, {value: "C"}, {value: "A"}]}})
        { columns { state { values } } } }""")
    assert 'AA' in data['project']['columns']['state']['values']


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
    data = client.execute('{ filter(zipcode: {ne: [0, 1]}) { count } }')
    assert data == {'filter': {'count': 41700}}
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
    data = client.execute('{ filter(state: {ne: null}) { count } }')
    assert data == {'filter': {'count': 41700}}
    data = client.execute(
        '{ filter(where: {le: [{numeric: {abs: {name: "longitude"}}}, {value: 66}]}) { count } }'
    )
    assert data['filter']['count'] == 30
    data = client.execute('{ dropNull { count } }')
    assert data == {'dropNull': {'count': 41700}}


def test_where(client):
    data = client.execute('{ filter(where: {eq: [{name: "county"}, {name: "city"}]}) { count } }')
    assert data['filter']['count'] == 2805
    data = client.execute("""{ filter(where: {or: [{eq: [{name: "state"}, {name: "county"}]},
        {eq: [{name: "county"}, {name: "city"}]}]}) { count } }""")
    assert data['filter']['count'] == 2805
    data = client.execute("""{ project(columns: {alias: "zipcode", add: [{name: "zipcode"}, {name: "zipcode"}]})
        { columns { zipcode { min } } } }""")
    assert data['project']['columns']['zipcode']['min'] == 1002
    data = client.execute("""{ project(columns: {alias: "zipcode", sub: [{name: "zipcode"}, {name: "zipcode"}]})
        { columns { zipcode { distinct { values } } } } }""")
    assert data['project']['columns']['zipcode']['distinct']['values'] == [0]
    data = client.execute("""{ project(columns: {alias: "product", mul: [{name: "latitude"}, {name: "longitude"}]})
        { filter(where: {gt: [{name: "product"}, {value: 0}]}) { count } } }""")
    assert data['project']['filter']['count'] == 0
    data = client.execute("""{ filter(where: {inv: {eq: [{name: "state"}, {value: "CA"}]}})
        { count } }""")
    assert data == {'filter': {'count': 39053}}
    with pytest.raises(ValueError):
        client.execute('{ filter(where: {name: "state", value: "CA"}) { count } }')
    with pytest.raises(ValueError, match="name or alias"):
        client.execute('{ project(columns: {}) { count } }')
    data = client.execute("""{ filter(where: {eq: [{name: "state"}, {value: "CA"}]})
        { filter(where: {eq: [{name: "county"}, {value: "Santa Clara"}]})
        { count row { county } } } }""")
    assert data == {'filter': {'filter': {'count': 108, 'row': {'county': 'Santa Clara'}}}}


def test_cast(client):
    data = client.execute("""{ cast(schema: {name: "zipcode", type: "float"})
        { column(name: "zipcode") { type } } }""")
    assert data['cast']['column']['type'] == 'float64'
    data = client.execute("""{ cast(schema: {name: "latitude",, type: "int32"}, try: true)
        { column(name: "latitude") { type } } }""")
    assert data == {'cast': {'column': {'type': 'int32'}}}


def test_project(client):
    assert client.execute('{ project(columns: []) { optional { type } } }')
    with pytest.raises(ValueError, match="conflict"):
        client.execute('{ project(columns: [{name: "state", value: ""}]) { type } }')
    with pytest.raises(ValueError, match="alias"):
        client.execute('{ project(columns: {inv: {name: "state"}}) { type } }')
    data = client.execute("""{ project(columns: {alias: "zipcode", numeric: {cumsum: {name: "zipcode"}}}) {
        columns { zipcode { first last } } } }""")
    assert data == {'project': {'columns': {'zipcode': {'first': 501, 'last': 2066562337}}}}
    data = client.execute("""{ project(columns: {alias: "state", cummin: {name: "state"}}) {
        columns { state { last } } } }""")
    assert data == {'project': {'columns': {'state': {'last': "AK"}}}}
    data = client.execute("""{ project(columns: {alias: "idx", denseRank: {name: "state"}}) {
        column(name: "idx") { ... on BigIntColumn { min max } } } }""")
    assert data == {'project': {'column': {'min': 0, 'max': 51}}}


def test_window(client):
    data = client.execute("""{ project(columns: {alias: "state", window: {lag: {name: "state"}, offset: 2}})
        { columns { state { values } } } }""")
    assert data['project']['columns']['state']['values'][:3] == [None, None, 'NY']
    data = client.execute("""{ project(columns: {alias: "state", window: {lead: {name: "state"}, default: ""}})
        { columns { state { values } } } }""")
    assert data['project']['columns']['state']['values'][-2:] == ['AK', '']
    data = client.execute("""{ project(columns: {alias: "runs", window: {ne: {name: "state"}, default: false}})
        { column(name: "runs") { ... on BooleanColumn { values } } } }""")
    assert data['project']['column']['values'][:4] == [False, False, True, False]


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


def test_distinct(client):
    data = client.execute('{ distinct { count } }')
    assert data == {'distinct': {'count': 41700}}
    data = client.execute("""{ distinct(on: "state", order: "idx") 
        { count column(name: "idx") { type } } }""")
    assert data == {'distinct': {'count': 52, 'column': {'type': 'int64'}}}
    data = client.execute('{ distinct(on: ["state", "county"], keep: null) { count } }')
    assert data == {'distinct': {'count': 132}}
    data = client.execute("""{ distinct(counts: "c") { column(name: "c")
        { ... on BigIntColumn { distinct { values counts } } } } }""")
    assert data == {'distinct': {'column': {'distinct': {'values': [1], 'counts': [41700]}}}}
    data = client.execute('{ distinct(on: "state", counts: "c") { schema { names types } } }')
    assert data['distinct']['schema'] == {
        'names': ['state', 'latitude', 'longitude', 'city', 'county', 'zipcode', 'c'],
        'types': ['string', 'float64', 'float64', 'string', 'string', 'int32', 'int64'],
    }


def test_group(client):
    with pytest.raises(ValueError, match="cannot represent"):
        client.execute("""{ group(by: "state", aggregate: {collect: {name: "city"}}) {
            row { city } } }""")
    data = client.execute("""{ group(by: [], counts: "c", aggregate: {max: {alias: "z", name: "zipcode"}}) {
        c: column(name: "c") { ... on BigIntColumn { values } }
        z: column(name: "z") { ... on IntColumn { values } } } }""")
    assert data == {'group': {'c': {'values': [41700]}, 'z': {'values': [99950]}}}
    data = client.execute("""{ group(by: "state", aggregate: {collect: {name: "county", distinct: true}}) {
        columns { state { values } }
        project(columns: {array: {length: {name: "county"}}, alias: "c"}) { column(name: "c") { ... on BigIntColumn { min } } } } }""")
    assert len(data['group']['columns']['state']['values']) == 52
    assert data['group']['project'] == {'column': {'min': 1}}
    data = client.execute("""{ group(by: ["state"], order: "idx", aggregate: {min: {name: "county"}})
        { row { state county } } }""")
    assert data == {'group': {'row': {'state': 'NY', 'county': 'Albany'}}}
    data = client.execute("""{ group(by: "state", aggregate: {collect: {name: "city", orderBy: "longitude", where: {eq: [{name: "county"}, {name: "city"}]}}})
        { project(columns: {alias: "num", array: {length: {name: "city"}}})
        { filter(where: {eq: [{name: "num"}, {value: null}]}) { count } } } }""")
    assert data == {'group': {'project': {'filter': {'count': 2}}}}
    data = client.execute("""{ group(aggregate: {nunique: {name: "state", alias: "num"}, quantile: {name: "state"}})
        { row { state } column(name: "num") { ... on BigIntColumn { first } } } }""")
    assert data == {'group': {'row': {'state': 'MS'}, 'column': {'first': 52}}}
    data = client.execute("""{ group(aggregate: {std: {name: "latitude", how: "pop"}})
        { row { latitude } } }""")
    assert data == {'group': {'row': {'latitude': pytest.approx(5.378499)}}}
    assert client.execute("""{ group(by: "state", counts: "c", order: "_") { count } }""")


def test_unnest(client):
    data = client.execute("""{ group(by: "state", aggregate: {collect: {name: "city"}}) {
        unnest(name: "city") { columns { city { type } } } } }""")
    assert data == {'group': {'unnest': {'columns': {'city': {'type': 'string'}}}}}
    data = client.execute("""{ group(by: "state", aggregate: {collect: {name: "city"}}) {
        unnest(name: "city", offset: "idx") { column(name: "idx") { type } } } }""")
    assert data == {'group': {'unnest': {'column': {'type': 'int64'}}}}
    data = client.execute("""{ group(by: "state", aggregate: {collect: {name: "city"}}) {
         unnest(name: "city", rowNumber: "idx") { column(name: "idx") { 
        ... on BigIntColumn { values } } } } }""")
    assert set(data['group']['unnest']['column']['values']) == set(range(52))


def test_rows(client):
    with pytest.raises(ValueError, match="not enough values"):
        client.execute('{ row(index: 100000) { zipcode } }')
    data = client.execute('{ row { state } }')
    assert data == {'row': {'state': 'NY'}}
    data = client.execute('{ row(index: -1) { state } }')
    assert data == {'row': {'state': 'AK'}}


def test_runs(client):
    data = client.execute("""{ runs(by: "state", aggregate: {collect: {name: "county"}})
        { count columns { state { values } } column(name: "county") { type } } }""")
    assert data['runs'].pop('columns')['state']['values'][:3] == ['NY', 'PR', 'MA']
    assert data == {'runs': {'count': 66, 'column': {'type': 'array<string>'}}}
    data = client.execute("""{ runs(split: {alias: "lat", window: {gt: {name: "latitude"}}}, counts: "c")
        { count schema { names types } } }""")
    assert data == {
        "runs": {"count": 20888, "schema": {"names": ["lat", "c"], "types": ["int64", "int64"]}}
    }
