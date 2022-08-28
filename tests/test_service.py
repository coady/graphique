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
        f'''{{ apply(int: {{name: "latitude", digitize: {list(range(90))}}})
        {{ columns {{ latitude {{ min max unique {{ length }} }} }} }} }}'''
    )
    latitudes = data['apply']['columns']['latitude']
    assert latitudes == {'min': 18.0, 'max': 72.0, 'unique': {'length': 52}}
    data = client.execute(
        '''{ slice(length: 1) { columns { latitude { logb(base: 3) { values } } } } }'''
    )
    assert data['slice']['columns']['latitude']['logb']['values'] == [pytest.approx(3.376188)]
    data = client.execute(
        '''{ apply(float: [{name: "latitude", round: true},
        {name: "longitude", isFinite: true}]) { row { latitude longitude } } }'''
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
        '''{ filter(query: {state: {equal: "CA"}}) {
        apply(string: {name: "city", utf8Length: true, alias: "size"}) {
        scan(filter: {gt: [{name: "size"}, {int: 23}]}) { length }
        column(name: "size") { ... on IntColumn { max } } } } }'''
    )
    assert data == {'filter': {'apply': {'scan': {'length': 1}, 'column': {'max': 24}}}}
    data = client.execute('{ apply(string: {name: "city", utf8Swapcase: true}) { row { city } } }')
    assert data == {'apply': {'row': {'city': 'hOLTSVILLE'}}}
    data = client.execute(
        '{ apply(string: {name: "state", utf8Capitalize: true}) { row { state } } }'
    )
    assert data == {'apply': {'row': {'state': 'Ny'}}}
    data = client.execute(
        '''{ apply(string: {name: "city", utf8IsLower: true})
        { scan(filter: {name: "city"}) { length } } }'''
    )
    assert data == {'apply': {'scan': {'length': 0}}}
    data = client.execute(
        '''{ apply(string: {name: "city", utf8IsTitle: true})
        { scan(filter: {name: "city"}) { length } } }'''
    )
    assert data == {'apply': {'scan': {'length': 41700}}}
    data = client.execute(
        '''{ apply(string: {name: "city", matchSubstring: "Mountain"})
        { scan(filter: {name: "city"}) { length } } }'''
    )
    assert data == {'apply': {'scan': {'length': 88}}}
    data = client.execute(
        '''{ apply(string: {name: "city", matchSubstring: "mountain", ignoreCase: true})
        { scan(filter: {name: "city"}) { length } } }'''
    )
    assert data == {'apply': {'scan': {'length': 88}}}
    data = client.execute(
        '''{ apply(string: {name: "city", matchSubstring: "^Mountain", regex: true})
        { scan(filter: {name: "city"}) { length } } }'''
    )
    assert data == {'apply': {'scan': {'length': 42}}}


def test_string_methods(client):
    data = client.execute(
        '''{ group(by: ["city"]) { columns { city { split {
        element { ... on StringColumn { values } }
        count { ... on LongColumn { max } } } } } } }'''
    )
    cities = data['group']['columns']['city']['split']
    assert cities['element']['values'].count('New') == 177
    assert cities['count'] == {'max': 6}
    data = client.execute(
        '''{ group(by: ["city"]) { columns { city {
        split(pattern: "-", maxSplits: 1, reverse: true) {
        count { unique { values counts } } } } } } }'''
    )
    cities = data['group']['columns']['city']
    cities['split']['count']['unique'] == {'values': [1, 2], 'counts': [18718, 1]}
    data = client.execute(
        '''{ columns { city { split(pattern: "-", maxSplits: 1, regex: true)
        { count { values } } } } }'''
    )
    assert all(count > 0 for count in data['columns']['city']['split']['count']['values'])
    data = client.execute(
        '{ columns { state { utf8Trim { values } utf8Ltrim { values } utf8Rtrim { values } } } }'
    )
    states = data['columns']['state']
    assert 'CA' in states['utf8Trim']['values']
    assert 'CA' in states['utf8Ltrim']['values']
    assert 'CA' in states['utf8Rtrim']['values']
    data = client.execute(
        '''{ columns { state { utf8Trim(characters: "C") { values }
        utf8Ltrim(characters: "C") { values } utf8Rtrim(characters: "A") { values } } } }'''
    )
    states = data['columns']['state']
    assert 'A' in states['utf8Trim']['values']
    assert 'A' in states['utf8Ltrim']['values']
    assert 'C' in states['utf8Rtrim']['values']
    data = client.execute(
        '''{ columns { state { utf8Center(width: 4, padding: "_") { values }
        utf8Lpad(width: 3) { values } utf8Rpad(width: 3) { values } } } }'''
    )
    states = data['columns']['state']
    assert all(len(state.split('_')) == 3 for state in states['utf8Center']['values'])
    assert all(state.startswith(' ') for state in states['utf8Lpad']['values'])
    assert all(state.endswith(' ') for state in states['utf8Rpad']['values'])
    data = client.execute(
        '''{ columns { state { utf8ReplaceSlice(start: 0, stop: 2, replacement: "")
        { unique { values } } } } }'''
    )
    assert data['columns']['state']['utf8ReplaceSlice']['unique']['values'] == ['']
    data = client.execute('{ columns { state { utf8SliceCodeunits(stop: 1) { values } } } }')
    assert data['columns']['state']['utf8SliceCodeunits']['values'][0] == 'N'
    data = client.execute(
        '''{ columns { state { replaceSubstring(pattern: "C", replacement: "A") { values } } } }'''
    )
    assert 'AA' in data['columns']['state']['replaceSubstring']['values']
    data = client.execute(
        '''{ group(by: "state") { column(name: "city") { ... on ListColumn
        { stringJoin(separator: ",") { ... on StringColumn { values } } } } } }'''
    )
    assert ','.join(['New York'] * 3) in data['group']['column']['stringJoin']['values'][0]


def test_search(client):
    data = client.execute('{ index search { length } }')
    assert data['index'] == ['zipcode']
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
    with pytest.raises(ValueError, match="optional, not nullable"):
        client.execute('{ search(zipcode: null) { length } }')


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
    data = client.execute(
        '''{ apply(float: {name: "longitude", abs: true})
        { filter(query: {longitude: {less: 66}}) { length } } }'''
    )
    assert data['apply']['filter']['length'] == 30
    with pytest.raises(ValueError, match="optional, not nullable"):
        client.execute('{ filter(query: {city: {less: null}}) { length } }')


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
        '''{ apply(float: {name: "longitude", maxElementWise: "latitude"})
        { columns { longitude { min } } } }'''
    )
    assert data['apply']['columns']['longitude']['min'] == pytest.approx(17.963333)
    data = client.execute(
        '''{ apply(float: {name: "latitude", minElementWise: "longitude"})
        { columns { latitude { max } } } }'''
    )
    assert data['apply']['columns']['latitude']['max'] == pytest.approx(-65.301389)
    data = client.execute(
        '''{ apply(string: {name: "city", findSubstring: "mountain"})
        { column(name: "city") { ... on IntColumn { unique { values } } } } }'''
    )
    assert data['apply']['column']['unique']['values'] == [-1]
    data = client.execute(
        '''{ apply(string: {name: "city", countSubstring: "mountain", ignoreCase: true})
        { column(name: "city") { ... on IntColumn { unique { values } } } } }'''
    )
    assert data['apply']['column']['unique']['values'] == [0, 1]
    data = client.execute(
        '''{ apply(string: {name: "state", binaryJoinElementWise: ["county", "city"]})
        { columns { state { values } } } }'''
    )
    assert data['apply']['columns']['state']['values'][0] == 'NYHoltsvilleSuffolk'


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
        apply(list: {name: "county", alias: "c", valueLength: true}) {
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
        column(name: "zipcode") { ... on ListColumn { count { values } } } } } } }'''
    )
    agg = data['group']['sort']['aggregate']
    assert agg['column']['count']['values'] == [525, 284, 242, 219]
    assert all(latitude > 1000 for latitude in agg['columns']['latitude']['values'])
    assert all(77 > longitude > -119 for longitude in agg['columns']['longitude']['values'])
    data = client.execute(
        '''{ scan(columns: {name: "zipcode", cast: "bool"})
        { group(by: ["state"]) { slice(length: 3) {
        aggregate(any: [{name: "zipcode", alias: "a"}], all: [{name: "zipcode", alias: "b"}]) {
        a: column(name: "a") { ... on BooleanColumn { values } }
        b: column(name: "b") { ... on BooleanColumn { values } }
        column(name: "zipcode") { ... on ListColumn { any { type } all { type } } } } } } } }'''
    )
    assert data['scan']['group']['slice']['aggregate'] == {
        'a': {'values': [True, True, True]},
        'b': {'values': [True, True, True]},
        'column': {'any': {'type': 'bool'}, 'all': {'type': 'bool'}},
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
        column(name: "county") { ... on ListColumn { count { values } } } } } }'''
    )
    agg = data['partition']['aggregate']
    assert agg['length'] == 66
    assert agg['columns']['state']['values'][:3] == ['NY', 'PR', 'MA']
    assert agg['column']['count']['values'][:3] == [2, 176, 701]
    data = client.execute(
        '''{ sort(by: ["state", "longitude"]) {
        partition(by: ["state", "longitude"], diffs: [{name: "longitude", greater: 1.0}]) {
        length columns { state { values } }
        column(name: "longitude") { ... on ListColumn { count { values } } } } } }'''
    )
    groups = data['sort']['partition']
    assert groups['length'] == 62
    assert groups['columns']['state']['values'][:7] == ['AK'] * 7
    assert groups['column']['count']['values'][:7] == [1, 1, 5, 232, 1, 32, 1]
    data = client.execute(
        '''{ partition(by: ["state"], diffs: [{name: "state", less: null}]) {
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
        '''{ partition(by: ["state"]) { filter(on: {int: {name: "zipcode", gt: 90000}}) {
        column(name: "zipcode") { ... on ListColumn { count { values } } } } } }'''
    )
    counts = data['partition']['filter']['column']['count']['values']
    assert len(counts) == 66
    assert counts.count(0) == 61
    data = client.execute(
        '''{ partition(by: ["state"], counts: "c") { filter(query: {state: {equal: "NY"}}) {
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
