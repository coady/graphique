import sys
import pytest


def test_camel(aliasclient):
    data = aliasclient.execute('{ index schema { names } }')
    assert data == {'index': ['snakeId', 'camelId'], 'schema': {'names': ['snakeId', 'camelId']}}
    data = aliasclient.execute('{ row { snakeId } columns { snakeId { type } } }')
    assert data == {'row': {'snakeId': 1}, 'columns': {'snakeId': {'type': 'int64'}}}
    data = aliasclient.execute('{ filter(snakeId: {eq: 1}) { index length } }')
    assert data == {'filter': {'index': [], 'length': 1}}
    data = aliasclient.execute('{ filter(camelId: {eq: 1}) { index length } }')
    assert data == {'filter': {'index': [], 'length': 1}}


def test_snake(executor):
    data = executor('{ index schema { names } }')
    assert data['index'] == ['snake_id', 'camelId']
    assert 'snake_id' in data['schema']['names']
    data = executor('{ row { snake_id } columns { snake_id { type } } }')
    assert data == {'row': {'snake_id': 1}, 'columns': {'snake_id': {'type': 'int64'}}}
    data = executor('{ filter(snake_id: {eq: 1}) { index length } }')
    assert data == {'filter': {'index': ['camelId'], 'length': 1}}
    data = executor('{ filter(camelId: {eq: 1}) { index length } }')
    assert data == {'filter': {'index': [], 'length': 1}}


def test_columns(executor):
    def execute(query):
        return executor(f'{{ columns {query} }}')['columns']

    for name in ('uint8', 'int8', 'uint16', 'int16', 'int32'):
        assert execute(f'{{ {name} {{ values }} }}') == {name: {'values': [0, None]}}
        assert execute(f'{{ {name} {{ index(value: 0) }} }}') == {name: {'index': 0}}
        data = execute(f'{{ {name} {{ dropNull {{ length }} }} }}')
        assert data == {name: {'dropNull': {'length': 1}}}
        assert execute(f'{{ {name} {{ type }} }}') == {name: {'type': name}}
        assert execute(f'{{ {name} {{ min max }} }}')
    for name in ('uint32', 'uint64', 'int64'):
        assert execute(f'{{ {name} {{ values }} }}') == {name: {'values': [0, None]}}
        assert execute(f'{{ {name} {{ index(value: 0) }} }}') == {name: {'index': 0}}
        data = execute(f'{{ {name} {{ dropNull {{ length }} }} }}')
        assert data == {name: {'dropNull': {'length': 1}}}
        assert execute(f'{{ {name} {{ min max }} }}')

    for name in ('float', 'double'):
        assert execute(f'{{ {name} {{ values }} }}') == {name: {'values': [0.0, None]}}
        assert execute(f'{{ {name} {{ index(value: 0.0) }} }}') == {name: {'index': 0}}
        data = execute(f'{{ {name} {{ dropNull {{ length }} }} }}')
        assert data == {name: {'dropNull': {'length': 1}}}
        assert execute(f'{{ {name} {{ min max }} }}')
    assert execute('{ decimal { values } }') == {'decimal': {'values': ['0', None]}}
    assert execute('{ decimal { min max } }')

    for name in ('date32', 'date64'):
        assert execute(f'{{ {name} {{ values }} }}') == {name: {'values': ['1970-01-01', None]}}
        assert execute(f'{{ {name} {{ index(value: "1970-01-01") }} }}') == {name: {'index': 0}}
        data = execute(f'{{ {name} {{ dropNull {{ length }} }} }}')
        assert data == {name: {'dropNull': {'length': 1}}}
        assert execute(f'{{ {name} {{ min max }} }}')

    data = execute('{ timestamp { values } }')
    assert data == {'timestamp': {'values': ['1970-01-01T00:00:00', None]}}
    data = execute(f'{{ {name} {{ dropNull {{ length }} }} }}')
    assert data == {name: {'dropNull': {'length': 1}}}
    assert execute('{ timestamp { index(value: "1970-01-01") } }') == {'timestamp': {'index': 0}}
    assert execute('{ timestamp { min max } }')

    for name in ('time32', 'time64'):
        assert execute(f'{{ {name} {{ values }} }}') == {name: {'values': ['00:00:00', None]}}
        assert execute(f'{{ {name} {{ index(value: "00:00:00") }} }}') == {name: {'index': 0}}
        data = execute(f'{{ {name} {{ dropNull {{ length }} }} }}')
        assert data == {name: {'dropNull': {'length': 1}}}
        assert execute(f'{{ {name} {{ min max }} }}')

    for name in ('binary', 'string'):
        assert execute(f'{{ {name} {{ values }} }}') == {name: {'values': ['', None]}}
        assert execute(f'{{ {name} {{ index(value: "") }} }}') == {name: {'index': 0}}
        data = execute(f'{{ {name} {{ dropNull {{ length }} }} }}')
        assert data == {name: {'dropNull': {'length': 1}}}

    assert execute('{ string { type } }') == {
        'string': {'type': 'dictionary<values=string, indices=int32, ordered=0>'}
    }
    assert execute('{ string { min max } }')


def test_boolean(executor):
    def execute(query):
        return executor(f'{{ columns {query} }}')['columns']

    assert execute('{ bool { values } }') == {'bool': {'values': [False, None]}}
    assert execute('{ bool { index(value: false) } }') == {'bool': {'index': 0}}
    assert execute('{ bool { index(value: false, start: 1, end: 2) } }') == {'bool': {'index': -1}}
    assert execute('{ bool { type } }') == {'bool': {'type': 'bool'}}
    assert execute('{ bool { unique { length } } }') == {'bool': {'unique': {'length': 2}}}
    assert execute('{ bool { any all } }') == {'bool': {'any': False, 'all': False}}

    data = executor(
        '{ apply(boolean: {xor: {name: ["bool", "bool"]}}) { columns { bool { values } } } }'
    )
    assert data == {'apply': {'columns': {'bool': {'values': [False, None]}}}}
    data = executor(
        '''{ apply(boolean: {andNotKleene: {name: ["bool", "bool"]}})
        { columns { bool { values } } } }'''
    )
    assert data == {'apply': {'columns': {'bool': {'values': [False, None]}}}}


def test_numeric(executor):
    for name in ('int32', 'int64', 'float'):
        data = executor(f'{{ columns {{ {name} {{ mean stddev variance }} }} }}')
        assert data == {'columns': {name: {'mean': 0.0, 'stddev': 0.0, 'variance': 0.0}}}
        data = executor(f'{{ columns {{ {name} {{ mode {{ values }} }} }} }}')
        assert data == {'columns': {name: {'mode': {'values': [0]}}}}
        data = executor(f'{{ columns {{ {name} {{ mode(length: 2) {{ counts }} }} }} }}')
        assert data == {'columns': {name: {'mode': {'counts': [1]}}}}
        data = executor(f'{{ columns {{ {name} {{ quantile }} }} }}')
        assert data == {'columns': {name: {'quantile': [0.0]}}}
        data = executor(f'{{ columns {{ {name} {{ tdigest }} }} }}')
        assert data == {'columns': {name: {'tdigest': [0.0]}}}
        data = executor(f'{{ columns {{ {name} {{ product }} }} }}')
        assert data == {'columns': {name: {'product': 0.0}}}

    data = executor(
        '''{ column(name: "float", apply: {minElementWise: "int32", maxElementWise: "int32"}) {
        ... on FloatColumn { values } } }'''
    )
    assert data == {'column': {'values': [0.0, None]}}
    data = executor('{ column(name: "float", cast: "int32") { type } }')
    assert data == {'column': {'type': 'int32'}}
    data = executor(
        '{ apply(int: {negateChecked: {name: "int32"}}) { columns { int32 { values } } } }'
    )
    assert data == {'apply': {'columns': {'int32': {'values': [0, None]}}}}
    data = executor(
        '{ apply(float: {coalesce: {name: ["float", "int32"]}}) { columns { float { values } } } }'
    )
    assert data == {'apply': {'columns': {'float': {'values': [0.0, None]}}}}
    data = executor(
        '{ apply(int: {bitWiseNot: {name: "int32"}}) { columns { int32 { values } } } }'
    )
    assert data == {'apply': {'columns': {'int32': {'values': [-1, None]}}}}
    data = executor(
        '{ apply(int: {bitWiseOr: {name: ["int32", "int64"]}}) { columns { int32 { values } } } }'
    )
    assert data == {'apply': {'columns': {'int32': {'values': [0, None]}}}}


@pytest.mark.skipif(sys.version_info < (3, 9), reason="requires python3.9 or pytz")
def test_datetime(executor):
    for name in ('timestamp', 'date32'):
        data = executor(
            f'''{{ apply(datetime: {{year: {{name: "{name}", alias: "year"}}}})
            {{ column(name: "year") {{ ... on LongColumn {{ values }} }} }} }}'''
        )
        assert data == {'apply': {'column': {'values': [1970, None]}}}
        data = executor(
            f'''{{ apply(datetime: {{quarter: {{name: "{name}", alias: "quarter"}}}})
            {{ column(name: "quarter") {{ ... on LongColumn {{ values }} }} }} }}'''
        )
        assert data == {'apply': {'column': {'values': [1, None]}}}
        data = executor(
            f'''{{ column(name: "{name}", apply: {{yearsBetween: "{name}"}})
            {{ ... on LongColumn {{ values }} }} }}'''
        )
        assert data == {'column': {'values': [0, None]}}
    data = executor(
        '''{ apply(datetime: {strftime: {name: "timestamp"}}) {
        column(name: "timestamp") { type } } }'''
    )
    assert data == {'apply': {'column': {'type': 'string'}}}
    for name in ('timestamp', 'time32'):
        data = executor(
            f'''{{ apply(datetime: {{hour: {{name: "{name}", alias: "hour"}}}})
            {{ column(name: "hour") {{ ... on LongColumn {{ values }} }} }} }}'''
        )
        assert data == {'apply': {'column': {'values': [0, None]}}}
        data = executor(
            f'''{{ apply(datetime: {{subsecond: {{name: "{name}", alias: "subsecond"}}}})
            {{ column(name: "subsecond") {{ ... on FloatColumn {{ values }} }} }} }}'''
        )
        assert data == {'apply': {'column': {'values': [0.0, None]}}}
        data = executor(
            f'''{{ apply(datetime: {{hoursBetween: {{name: ["{name}", "{name}"]}}}})
            {{ column(name: "{name}") {{ ... on LongColumn {{ values }} }} }} }}'''
        )
        assert data == {'apply': {'column': {'values': [0, None]}}}
    with pytest.raises(ValueError):
        executor('{ columns { time64 { between(unit: "hours") { values } } } }')
    data = executor(
        '''{ apply(datetime: {assumeTimezone: {name: "timestamp", timezone: "UTC"}}) {
        columns { timestamp { values } } } }'''
    )
    dates = data['apply']['columns']['timestamp']['values']
    assert dates == ['1970-01-01T00:00:00+00:00', None]
    data = executor(
        '''{ apply(time: {roundTemporal: {name: "time32", unit: "hour"}}) {
        columns { time32 { values } } } }'''
    )
    assert data == {'apply': {'columns': {'time32': {'values': ['00:00:00', None]}}}}


def test_duration(executor):
    data = executor(
        '''{ scan(columns: {alias: "diff", sub: [{name: "timestamp"}, {name: "timestamp"}]})
        { column(name: "diff") { ... on DurationColumn { values } } } }'''
    )
    assert data == {'scan': {'column': {'values': [0.0, None]}}}
    data = executor(
        '''{ partition(by: ["timestamp"] diffs: [{name: "timestamp", gt: 0.0}]) { length } }'''
    )
    assert data == {'partition': {'length': 1}}


def test_list(executor):
    data = executor('{ columns { list { length type } } }')
    assert data == {'columns': {'list': {'length': 2, 'type': 'list<item: int32>'}}}
    data = executor('{ columns { list { values { length } } } }')
    assert data == {'columns': {'list': {'values': [{'length': 3}, None]}}}
    data = executor('{ row { list { ... on IntColumn { values } } } }')
    assert data == {'row': {'list': {'values': [0, 1, 2]}}}
    data = executor('{ row(index: -1) { list { ... on IntColumn { values } } } }')
    assert data == {'row': {'list': None}}
    data = executor(
        '''{ aggregate(approximateMedian: {name: "list"}) {
        column(name: "list") { ... on FloatColumn { values } } } }'''
    )
    assert data == {'aggregate': {'column': {'values': [1.0, None]}}}
    data = executor(
        '''{ aggregate(tdigest: {name: "list", q: [0.25, 0.75]}) {
        columns { list { flatten { ... on FloatColumn { values } } } } } }'''
    )
    column = data['aggregate']['columns']['list']
    assert column == {'flatten': {'values': [0.0, 2.0]}}

    data = executor(
        '''{ apply(list: {element: {name: "list", index: 1}}) {
        column(name: "list") { ... on IntColumn { values } } } }'''
    )
    assert data == {'apply': {'column': {'values': [1, None]}}}
    data = executor(
        '''{ aggregate(distinct: {name: "list", mode: "only_null"})
        { columns { list { flatten { length } } } } }'''
    )
    assert data['aggregate']['columns']['list'] == {'flatten': {'length': 0}}
    data = executor(
        '''{ apply(list: {filter: {ne: [{name: "list"}, {int: 1}]}}) {
        columns { list { values { ... on IntColumn { values } } } } } }'''
    )
    column = data['apply']['columns']['list']
    assert column == {'values': [{'values': [0, 2]}, {'values': []}]}
    data = executor('{ apply(list: {mode: {name: "list"}}) { column(name: "list") { type } } }')
    assert data['apply']['column']['type'] == 'large_list<item: struct<mode: int32, count: int64>>'
    data = executor(
        '''{ aggregate(stddev: {name: "list"}, variance: {name: "list", alias: "var", ddof: 1}) {
        column(name: "list") { ... on FloatColumn { values } }
        var: column(name: "var") { ... on FloatColumn { values } } } }'''
    )
    assert data['aggregate']['column']['values'] == [pytest.approx((2 / 3) ** 0.5), None]
    assert data['aggregate']['var']['values'] == [1, None]
    data = executor(
        '''{ partition(by: "int32") { apply(base64: {binaryJoin: {name: "binary", value: ""}}) {
        column(name: "binary") { ... on Base64Column { values } } } } }'''
    )
    assert data == {'partition': {'apply': {'column': {'values': [None]}}}}


def test_struct(executor):
    data = executor('{ columns { struct { names column(name: "x") { length } } } }')
    assert data == {'columns': {'struct': {'names': ['x', 'y'], 'column': {'length': 2}}}}
    data = executor(
        '''{apply(struct: {structField: {name: "struct", indices: 0}}) {
        column(name: "struct") { ... on IntColumn { values } } } }'''
    )
    assert data == {'apply': {'column': {'values': [0, None]}}}
    with pytest.raises(ValueError, match="must be BOOL"):
        executor('{ apply(struct: {caseWhen: {name: ["struct", "int32", "float"]}}) { type } }')


def test_dictionary(executor):
    data = executor('{ column(name: "string") { length } }')
    assert data == {'column': {'length': 2}}
    data = executor(
        '''{ group(by: ["string"]) { tables {
        columns { string { values } } column(name: "camelId") { length } } } }'''
    )
    assert data['group']['tables'] == [
        {'columns': {'string': {'values': ['']}}, 'column': {'length': 1}},
        {'columns': {'string': {'values': [None]}}, 'column': {'length': 1}},
    ]
    data = executor(
        '''{ group(by: ["camelId"]) { aggregate(countDistinct: {name: "string"}) {
        column(name: "string") { ... on LongColumn { values } } } } }'''
    )
    assert data == {'group': {'aggregate': {'column': {'values': [1, 0]}}}}
    data = executor(
        '''{ apply(string: {coalesce: {name: "string", value: ""}}) {
        columns { string { values } } } }'''
    )
    assert data == {'apply': {'columns': {'string': {'values': ['', '']}}}}


def test_selections(executor):
    data = executor('{ slice { length } slice { sort(by: "snake_id") { length } } }')
    assert data == {'slice': {'length': 2, 'sort': {'length': 2}}}


def test_conditions(executor):
    data = executor(
        '''{ apply(boolean: {ifElse: {name: ["bool", "int32", "float"]}}) {
        column(name: "bool") { type } } }'''
    )
    assert data == {'apply': {'column': {'type': 'float'}}}
    with pytest.raises(ValueError, match="no kernel"):
        executor('{ apply(boolean: {ifElse: {name: ["struct", "int32", "float"]}}) { type } }')


def test_long(executor):
    with pytest.raises(ValueError, match="Long cannot represent value"):
        executor('{ filter(int64: {eq: 0.0}) { length } }')


def test_base64(executor):
    data = executor(
        '''{ apply(base64: {binaryLength: {name: "binary"}}) {
        column(name: "binary") { ...on IntColumn { values } } } }'''
    )
    assert data == {'apply': {'column': {'values': [0, None]}}}
    data = executor(
        '{ apply(base64: {fillNullForward: {name: "binary"}}) { columns { binary { values } } } }'
    )
    assert data == {'apply': {'columns': {'binary': {'values': ['', '']}}}}
    data = executor(
        '''{ apply(base64: {coalesce: {name: "binary", value: "Xw=="}}) {
        columns { binary { values } } } }'''
    )
    assert data == {'apply': {'columns': {'binary': {'values': ['', 'Xw==']}}}}
    data = executor(
        '''{ apply(base64: {coalesce: {name: [null, "binary"], value: "Xw=="}}) {
        columns { binary { values } } } }'''
    )
    assert data == {'apply': {'columns': {'binary': {'values': ['Xw==', 'Xw==']}}}}
    data = executor(
        '''{ apply(base64: {binaryJoinElementWise: {
        name: ["binary", "binary"], value: "Xw==", nullHandling: "replace"}}) {
        columns { binary { values } } } }'''
    )
    assert data == {'apply': {'columns': {'binary': {'values': ['Xw==', 'Xw==']}}}}
    data = executor(
        '''{ apply(base64: {binaryReplaceSlice:
        {name: "binary", start: 0, stop: 1, replacement: "Xw=="}}) {
        columns { binary { values } } } }'''
    )
    assert data == {'apply': {'columns': {'binary': {'values': ['Xw==', None]}}}}
