import pytest


def test_slice(client):
    data = client.execute('{ length slice(length: 3) { columns { zipcode { values } } } }')
    assert data == {'length': 41700, 'slice': {'columns': {'zipcode': {'values': [501, 544, 601]}}}}
    data = client.execute('{ slice(offset: 1) { columns { zipcode { values } } } }')
    zipcodes = data['slice']['columns']['zipcode']['values']
    assert zipcodes[0] == 544
    assert len(zipcodes) == 41699
    data = client.execute('{ slice(offset: -1, reverse: true) { columns { zipcode { values } } } }')
    assert data['slice']['columns']['zipcode']['values'] == [99950]
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
        '''{ column(name: "latitude", apply: {minElementWise: "longitude"})
        { ... on FloatColumn { min } } }'''
    )
    assert data['column']['min'] == pytest.approx(-174.213333)
    data = client.execute(
        f'''{{ apply(int: {{digitize: {{name: "latitude", bins: {list(range(90))} }}}})
        {{ columns {{ latitude {{ min max unique {{ length }} }} }} }} }}'''
    )
    latitudes = data['apply']['columns']['latitude']
    assert latitudes == {'min': 18.0, 'max': 72.0, 'unique': {'length': 52}}
    data = client.execute(
        '''{ slice(length: 1) { columns { latitude { logb(base: 3) { values } } } } }'''
    )
    assert data['slice']['columns']['latitude']['logb']['values'] == [pytest.approx(3.376188)]
    data = client.execute(
        '''{ apply(float: {round: {name: "latitude"}, isFinite: {name: "longitude"}}) {
        row { latitude longitude } } }'''
    )
    assert data == {'apply': {'row': {'latitude': 41.0, 'longitude': 1}}}
    data = client.execute(
        '''{ slice(length: 1) { columns { latitude { round(ndigits: 1) { values } } } } }'''
    )
    assert data == {'slice': {'columns': {'latitude': {'round': {'values': [40.8]}}}}}
    data = client.execute(
        '''{ slice(length: 1) { columns { latitude { round(multiple: 10) { values } } } } }'''
    )
    assert data == {'slice': {'columns': {'latitude': {'round': {'values': [40.0]}}}}}
    with pytest.raises(ValueError, match="only one"):
        client.execute('{ columns { latitude { round(ndigits: 1, multiple: 10) { length } } } }')


def test_strings(client):
    data = client.execute(
        '''{ columns {
        state { values unique { values counts } }
        county { unique { length values } }
        city { min max }
    } }'''
    )
    states = data['columns']['state']
    assert len(states['values']) == 41700
    assert len(states['unique']['values']) == 52
    assert sum(states['unique']['counts']) == 41700
    counties = data['columns']['county']
    assert len(counties['unique']['values']) == counties['unique']['length'] == 1920
    assert data['columns']['city'] == {'min': 'Aaronsburg', 'max': 'Zwolle'}
    data = client.execute(
        '''{ filter(state: {eq: "CA"}) {
        apply(string: {utf8Length: {name: "city", alias: "size"}}) {
        scan(filter: {gt: [{name: "size"}, {int: 23}]}) { length }
        column(name: "size") { ... on IntColumn { max } } } } }'''
    )
    assert data == {'filter': {'apply': {'scan': {'length': 1}, 'column': {'max': 24}}}}
    data = client.execute('{ apply(string: {utf8Swapcase: {name: "city"}}) { row { city } } }')
    assert data == {'apply': {'row': {'city': 'hOLTSVILLE'}}}
    data = client.execute('{ apply(string: {utf8Capitalize: {name: "state"}}) { row { state } } }')
    assert data == {'apply': {'row': {'state': 'Ny'}}}
    data = client.execute(
        '''{ apply(string: {utf8IsLower: {name: "city"}})
        { scan(filter: {name: "city"}) { length } } }'''
    )
    assert data == {'apply': {'scan': {'length': 0}}}
    data = client.execute(
        '''{ apply(string: {utf8IsTitle: {name: "city"}})
        { scan(filter: {name: "city"}) { length } } }'''
    )
    assert data == {'apply': {'scan': {'length': 41700}}}
    data = client.execute(
        '''{ apply(string: {matchSubstring: {name: "city", value: "Mountain"}})
        { scan(filter: {name: "city"}) { length } } }'''
    )
    assert data == {'apply': {'scan': {'length': 88}}}
    data = client.execute(
        '''{ apply(string: {matchSubstring: {name: "city", value: "mountain", ignoreCase: true}})
        { scan(filter: {name: "city"}) { length } } }'''
    )
    assert data == {'apply': {'scan': {'length': 88}}}
    data = client.execute(
        '''{ apply(string: {matchSubstringRegex: {name: "city", value: "^Mountain"}})
        { scan(filter: {name: "city"}) { length } } }'''
    )
    assert data == {'apply': {'scan': {'length': 42}}}


def test_string_methods(client):
    data = client.execute(
        '''{ apply(string: {splitPattern: {name: "city", value: "-", maxSplits: 1}}) {
        columns { city { type } } } }'''
    )
    assert data == {'apply': {'columns': {'city': {'type': 'list<item: string>'}}}}
    data = client.execute(
        '''{ apply(string: {utf8Trim: {name: "state", value: "C"}}) {
        columns { state { values } } } }'''
    )
    states = data['apply']['columns']['state']['values']
    assert 'CA' not in states and 'A' in states
    data = client.execute(
        '{ apply(string: {utf8Center: {name: "state", width: 4, padding: "_"}}) { row { state } } }'
    )
    assert data == {'apply': {'row': {'state': '_NY_'}}}
    data = client.execute(
        '''{ apply(string: {utf8ReplaceSlice: {name: "state", start: 0, stop: 2, replacement: ""}})
        { columns { state { unique { values } } } } }'''
    )
    assert data == {'apply': {'columns': {'state': {'unique': {'values': ['']}}}}}
    data = client.execute(
        '{ apply(string: {utf8SliceCodeunits: {name: "state", start: 0, stop: 1}}) { row { state } } }'
    )
    assert data == {'apply': {'row': {'state': 'N'}}}
    data = client.execute(
        '''{ apply(string: {replaceSubstring: {name: "state", pattern: "C", replacement: "A"}})
        { columns { state { values } } } }'''
    )
    assert 'AA' in data['apply']['columns']['state']['values']


def test_search(client):
    data = client.execute('{ index filter { length } }')
    assert data == {'index': ['zipcode'], 'filter': {'length': 41700}}
    data = client.execute('{ filter(zipcode: {eq: 501}) { columns { zipcode { values } } } }')
    assert data == {'filter': {'columns': {'zipcode': {'values': [501]}}}}
    data = client.execute('{ filter(zipcode: {ne: 501}) { length } }')
    assert data['filter']['length'] == 41699

    data = client.execute('{ filter(zipcode: {ge: 99929}) { columns { zipcode { values } } } }')
    assert data == {'filter': {'columns': {'zipcode': {'values': [99929, 99950]}}}}
    data = client.execute('{ filter(zipcode: {lt: 601}) { columns { zipcode { values } } } }')
    assert data == {'filter': {'columns': {'zipcode': {'values': [501, 544]}}}}
    data = client.execute(
        '{ filter(zipcode: {gt: 501, le: 601}) { columns { zipcode { values } } } }'
    )
    assert data == {'filter': {'columns': {'zipcode': {'values': [544, 601]}}}}

    data = client.execute('{ filter(zipcode: {eq: []}) { length } }')
    assert data == {'filter': {'length': 0}}
    data = client.execute('{ filter(zipcode: {eq: [0]}) { length } }')
    assert data == {'filter': {'length': 0}}
    data = client.execute(
        '{ filter(zipcode: {eq: [501, 601]}) { columns { zipcode { values } } } }'
    )
    assert data == {'filter': {'columns': {'zipcode': {'values': [501, 601]}}}}


def test_filter(client):
    data = client.execute('{ filter { length } }')
    assert data['filter']['length'] == 41700
    data = client.execute('{ filter(city: {eq: "Mountain View"}) { length } }')
    assert data['filter']['length'] == 11
    data = client.execute('{ filter(state: {ne: "CA"}) { length } }')
    assert data['filter']['length'] == 39053
    data = client.execute('{ filter(city: {eq: "Mountain View"}, state: {le: "CA"}) { length } }')
    assert data['filter']['length'] == 7
    data = client.execute('{ filter(state: {eq: null}) { columns { state { values } } } }')
    assert data['filter']['columns']['state']['values'] == []
    data = client.execute(
        '''{ apply(float: {abs: {name: "longitude"}})
        { filter(longitude: {le: 66}) { length } } }'''
    )
    assert data['apply']['filter']['length'] == 30
    with pytest.raises(ValueError, match="optional, not nullable"):
        client.execute('{ filter(city: {le: null}) { length } }')


def test_scan(client):
    data = client.execute('{ scan(filter: {eq: [{name: "county"}, {name: "city"}]}) { length } }')
    assert data['scan']['length'] == 2805
    data = client.execute(
        '''{ scan(filter: {or: [{eq: [{name: "state"}, {name: "county"}]},
        {eq: [{name: "county"}, {name: "city"}]}]}) { length } }'''
    )
    assert data['scan']['length'] == 2805
    data = client.execute(
        '''{ scan(columns: {alias: "zipcode", add: [{name: "zipcode"}, {name: "zipcode"}]})
        { columns { zipcode { min } } } }'''
    )
    assert data['scan']['columns']['zipcode']['min'] == 1002
    data = client.execute(
        '''{ scan(columns: {alias: "zipcode", sub: [{name: "zipcode"}, {name: "zipcode"}]})
        { columns { zipcode { unique { values } } } } }'''
    )
    assert data['scan']['columns']['zipcode']['unique']['values'] == [0]
    data = client.execute(
        '''{ scan(columns: {alias: "product", mul: [{name: "latitude"}, {name: "longitude"}]})
        { scan(filter: {gt: [{name: "product"}, {float: 0}]}) { length } } }'''
    )
    assert data['scan']['scan']['length'] == 0
    data = client.execute(
        '{ scan(columns: {name: "zipcode", cast: "float"}) { column(name: "zipcode") { type } } }'
    )
    assert data['scan']['column']['type'] == 'float'
    data = client.execute(
        '{ scan(filter: {inv: {eq: [{name: "state"}, {string: "CA"}]}}) { length } }'
    )
    assert data == {'scan': {'length': 39053}}


def test_apply(client):
    data = client.execute(
        '''{ apply(float: {maxElementWise: {name: ["longitude", "latitude"]}})
        { columns { longitude { min } } } }'''
    )
    assert data['apply']['columns']['longitude']['min'] == pytest.approx(17.963333)
    data = client.execute(
        '''{ apply(float: {minElementWise: {name: ["latitude", "longitude"]}})
        { columns { latitude { max } } } }'''
    )
    assert data['apply']['columns']['latitude']['max'] == pytest.approx(-65.301389)
    data = client.execute(
        '''{ apply(string: {findSubstring: {name: "city", value: "mountain"}})
        { column(name: "city") { ... on IntColumn { unique { values } } } } }'''
    )
    assert data['apply']['column']['unique']['values'] == [-1]
    data = client.execute(
        '''{ apply(string: {countSubstring: {name: "city", value: "mountain", ignoreCase: true}})
        { column(name: "city") { ... on IntColumn { unique { values } } } } }'''
    )
    assert data['apply']['column']['unique']['values'] == [0, 1]
    data = client.execute(
        '''{ apply(string: {binaryJoinElementWise: {name: ["state", "county"], value: " "}})
        { columns { state { values } } } }'''
    )
    assert data['apply']['columns']['state']['values'][0] == 'NY Suffolk'
    data = client.execute(
        '''{ apply(string: {minElementWise: {name: ["state", "county"], skipNulls: false}})
        { columns { state { values } } } }'''
    )
    assert data['apply']['columns']['state']['values'][0] == 'NY'


def test_sort(client):
    with pytest.raises(ValueError, match="is required"):
        client.execute('{ sort { columns { state { values } } } }')
    data = client.execute('{ sort(by: ["state"]) { columns { state { values } } } }')
    assert data['sort']['columns']['state']['values'][0] == 'AK'
    data = client.execute('{ sort(by: "-state") { columns { state { values } } } }')
    assert data['sort']['columns']['state']['values'][0] == 'WY'
    data = client.execute('{ sort(by: ["state"], length: 1) { columns { state { values } } } }')
    assert data['sort']['columns']['state']['values'] == ['AK']
    data = client.execute('{ sort(by: "-state", length: 1) { columns { state { values } } } }')
    assert data['sort']['columns']['state']['values'] == ['WY']
    data = client.execute('{ sort(by: ["state", "county"]) { columns { county { values } } } }')
    assert data['sort']['columns']['county']['values'][0] == 'Aleutians East'
    data = client.execute(
        '''{ sort(by: ["-state", "-county"], length: 1) { columns { county { values } } } }'''
    )
    assert data['sort']['columns']['county']['values'] == ['Weston']
    data = client.execute('{ sort(by: ["state"], length: 2) { columns { state { values } } } }')
    assert data['sort']['columns']['state']['values'] == ['AK', 'AK']
    data = client.execute(
        '''{ group(by: ["state"]) { sort(by: ["county"])
        { aggregate(first: [{name: "county"}]) { row { state county } } } } }'''
    )
    assert data['group']['sort']['aggregate']['row'] == {'state': 'NY', 'county': 'Albany'}
    data = client.execute(
        '''{ group(by: ["state"]) { sort(by: ["-county"], length: 1)
        { aggregate(first: [{name: "county"}]) { row { state county } } } } }'''
    )
    assert data['group']['sort']['aggregate']['row'] == {'state': 'NY', 'county': 'Yates'}
    data = client.execute('{ group(by: ["state"]) { sort(by: ["state", "county"]) { length } } }')
    assert data['group']['sort']['length']


def test_group(client):
    with pytest.raises(ValueError, match="is required"):
        client.execute('{ group { length } }')
    with pytest.raises(ValueError, match="list"):
        client.execute('{ group(by: ["state"]) { tables { length } } }')
    data = client.execute(
        '''{ group(by: ["state"]) { length tables { length
        columns { state { values } county { min max } } }
        apply(list: {valueLength: {name: "county", alias: "c"}}) {
        column(name: "c") { ... on IntColumn { values } } } } }'''
    )
    assert len(data['group']['tables']) == data['group']['length'] == 52
    table = data['group']['tables'][0]
    assert table['length'] == data['group']['apply']['column']['values'][0] == 2205
    assert set(table['columns']['state']['values']) == {'NY'}
    assert table['columns']['county'] == {'min': 'Albany', 'max': 'Yates'}
    data = client.execute(
        '''{ group(by: ["state", "county"], counts: "counts") {
        scan(filter: {gt: [{name: "counts"}, {int: 200}]}) {
        aggregate(min: [{name: "city", alias: "min"}], max: [{name: "city", alias: "max"}]) {
        min: column(name: "min") { ... on StringColumn { values } }
        max: column(name: "max") { ... on StringColumn { values } } } } } }'''
    )
    agg = data['group']['scan']['aggregate']
    assert agg['min']['values'] == ['Naval Anacost Annex', 'Alsip', 'Alief', 'Acton']
    assert agg['max']['values'] == ['Washington Navy Yard', 'Worth', 'Webster', 'Woodland Hills']
    data = client.execute(
        '''{ group(by: ["state", "county"], counts: "c") { sort(by: ["-c"], length: 4) {
        aggregate(sum: [{name: "latitude"}], mean: [{name: "longitude"}]) {
        columns { latitude { values } longitude { values } }
        column(name: "zipcode") { type } } } } }'''
    )
    agg = data['group']['sort']['aggregate']
    assert agg['column']['type'] == 'list<item: int32>'
    assert all(latitude > 1000 for latitude in agg['columns']['latitude']['values'])
    assert all(77 > longitude > -119 for longitude in agg['columns']['longitude']['values'])
    data = client.execute(
        '''{ scan(columns: {name: "zipcode", cast: "bool"})
        { group(by: ["state"]) { slice(length: 3) {
        aggregate(any: [{name: "zipcode", alias: "a"}], all: [{name: "zipcode", alias: "b"}]) {
        a: column(name: "a") { ... on BooleanColumn { values } }
        b: column(name: "b") { ... on BooleanColumn { values } }
        column(name: "zipcode") { type } } } } } }'''
    )
    assert data['scan']['group']['slice']['aggregate'] == {
        'a': {'values': [True, True, True]},
        'b': {'values': [True, True, True]},
        'column': {'type': 'list<item: bool>'},
    }
    data = client.execute(
        '''{ sc: group(by: ["state", "county"]) { length }
        cs: group(by: ["county", "state"]) { length } }'''
    )
    assert data['sc']['length'] == data['cs']['length'] == 3216


def test_aggregate(client):
    data = client.execute(
        '''{ group(by: ["state"] counts: "c", aggregate: {first: [{name: "county"}]
        countDistinct: [{name: "city", alias: "cd"}]}) { slice(length: 3) {
        c: column(name: "c") { ... on LongColumn { values } }
        cd: column(name: "cd") { ... on LongColumn { values } }
        columns { state { values } county { values } } } } }'''
    )
    assert data['group']['slice'] == {
        'c': {'values': [2205, 176, 703]},
        'cd': {'values': [1612, 99, 511]},
        'columns': {
            'state': {'values': ['NY', 'PR', 'MA']},
            'county': {'values': ['Suffolk', 'Adjuntas', 'Hampden']},
        },
    }
    data = client.execute(
        '''{ group(by: ["state", "county"], aggregate: {min: {name: "city", alias: "first"}}) {
        aggregate(max: {name: "city", alias: "last"}) { slice(length: 3) {
        first: column(name: "first") { ... on StringColumn { values } }
        last: column(name: "last") { ... on StringColumn { values } } } } } }'''
    )
    assert data['group']['aggregate']['slice'] == {
        'first': {'values': ['Amagansett', 'Adjuntas', 'Aguada']},
        'last': {'values': ['Yaphank', 'Adjuntas', 'Aguada']},
    }


def test_partition(client):
    with pytest.raises(ValueError, match="is required"):
        client.execute('{ group { length } }')
    data = client.execute(
        '''{ partition(by: ["state"]) { aggregate { length columns { state { values } }
        column(name: "county") { type } } } }'''
    )
    agg = data['partition']['aggregate']
    assert agg['length'] == 66
    assert agg['columns']['state']['values'][:3] == ['NY', 'PR', 'MA']
    assert agg['column']['type'] == 'list<item: string>'
    data = client.execute(
        '''{ sort(by: ["state", "longitude"]) {
        partition(by: ["state", "longitude"], diffs: [{name: "longitude", gt: 1.0}]) {
        length columns { state { values } }
        column(name: "longitude") { type } } } }'''
    )
    groups = data['sort']['partition']
    assert groups['length'] == 62
    assert groups['columns']['state']['values'][:7] == ['AK'] * 7
    assert groups['column']['type'] == 'list<item: double>'
    data = client.execute(
        '''{ partition(by: ["state"], diffs: [{name: "state", lt: null}]) {
        length column(name: "state") {
        ... on ListColumn {values { ... on StringColumn { values } } } } } }'''
    )
    assert data['partition']['length'] == 34
    assert data['partition']['column']['values'][0]['values'] == (['NY'] * 2) + (['PR'] * 176)
    data = client.execute(
        '''{ partition(by: ["state"]) { sort(by: ["state"]) {
        columns { state { values } } } } }'''
    )
    assert data['partition']['sort']['columns']['state']['values'][-2:] == ['WY', 'WY']
    data = client.execute(
        '''{ partition(by: ["state"]) { slice(offset: 2, length: 2) {
        aggregate(count: {name: "zipcode", alias: "c"}) {
        column(name: "c") { ... on LongColumn { values } } columns { state { values } } } } } }'''
    )
    agg = data['partition']['slice']['aggregate']
    assert agg['column']['values'] == [701, 91]
    assert agg['columns']['state']['values'] == ['MA', 'RI']
    data = client.execute(
        '''{ partition(by: ["state"]) {
        apply(list: {filter: {gt: [{name: "zipcode"}, {int: 90000}]}}) {
        column(name: "zipcode") { type } } } }'''
    )
    assert data['partition']['apply']['column']['type'] == 'list<item: int32>'
    data = client.execute(
        '''{ partition(by: ["state"], counts: "c") { filter(state: {eq: "NY"}) {
        column(name: "c") { ... on LongColumn { values } } columns { state { values } } } } }'''
    )
    agg = data['partition']['filter']
    assert agg['column']['values'] == [2, 1, 2202]
    assert agg['columns']['state']['values'] == ['NY'] * 3


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
    data = client.execute(
        '''{ group(by: ["state"]) { min(by: ["state", "longitude"]) {
        aggregate(first: [{name: "city"}]) { row { state } columns { city { values } } } } } }'''
    )
    agg = data['group']['min']['aggregate']
    assert agg == {'row': {'state': 'AK'}, 'columns': {'city': {'values': ['Atka']}}}
