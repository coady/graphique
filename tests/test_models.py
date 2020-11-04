import pytest
from strawberry.printer import print_schema
from .conftest import fixtures


def test_schema(schema):
    with open(fixtures / 'schema.graphql', 'w') as file:
        file.write(print_schema(schema))


def test_case(executor):
    data = executor('{ columns { snakeId { values } camelId { values } } }')
    assert data == {'columns': {'snakeId': {'values': [1, 2]}, 'camelId': {'values': [1, 2]}}}
    data = executor('{ columns { camelId(minimum: "snakeId") { values } } }')
    assert data == {'columns': {'camelId': {'values': [1, 2]}}}
    data = executor('{ column(name: "snakeId") { length } }')
    assert data == {'column': {'length': 2}}
    data = executor('{ row { snakeId camelId } }')
    assert data == {'row': {'snakeId': 1, 'camelId': 1}}
    data = executor('{ filter(query: {snakeId: {equal: 1}, camelId: {equal: 1}}) { length } }')
    assert data == {'filter': {'length': 1}}
    data = executor(
        '{ filter(query: {snakeId: {equal: 1}, camelId: {equal: 1}}, invert: true) { length } }'
    )
    assert data == {'filter': {'length': 1}}
    data = executor('{ filter(query: {camelId: {apply: {equal: "snakeId"}}}) { length } }')
    assert data == {'filter': {'length': 2}}
    data = executor('{ apply(camelId: {add: "snakeId"}) { columns { camelId { values } } } }')
    assert data == {'apply': {'columns': {'camelId': {'values': [2, 4]}}}}
    data = executor('{ index search(snakeId: {equal: 1}) { length } }')
    assert data == {'index': ['snakeId', 'camelId'], 'search': {'length': 1}}
    data = executor('{ min(by: ["snakeId", "camelId"]) { row { snakeId camelId } } }')
    assert data == {'min': {'row': {'snakeId': 1, 'camelId': 1}}}
    data = executor('{ max(by: ["snakeId", "camelId"]) { row { snakeId camelId } } }')
    assert data == {'max': {'row': {'snakeId': 2, 'camelId': 2}}}
    with pytest.raises(ValueError, match="non-equal query for"):
        executor('{ index search(snakeId: {less: 1}, camelId: {equal: 1}) { length } }')
    with pytest.raises(ValueError, match="expected query for"):
        executor('{ index search(camelId: {equal: 1}) { length } }')


def test_columns(executor):
    def execute(query):
        return executor(f'{{ columns {query} }}')['columns']

    assert execute('{ bool { values } }') == {'bool': {'values': [False, None]}}
    assert execute('{ bool { count(equal: false) } }') == {'bool': {'count': 1}}
    assert execute('{ bool { type } }') == {'bool': {'type': 'bool'}}
    assert execute('{ bool { unique { length } } }') == {'bool': {'unique': {'length': 2}}}

    for name in ('uint8', 'int8', 'uint16', 'int16', 'int32'):
        assert execute(f'{{ {name} {{ values }} }}') == {name: {'values': [0, None]}}
        assert execute(f'{{ {name} {{ count(equal: 0) }} }}') == {name: {'count': 1}}
        data = execute(f'{{ {name} {{ fillNull(value: 1) {{ values }} }} }}')
        assert data == {name: {'fillNull': {'values': [0, 1]}}}
        assert execute(f'{{ {name} {{ type }} }}') == {name: {'type': name}}
        assert execute(f'{{ {name} {{ min max sort first: sort(length: 1) }} }}')
    for name in ('uint32', 'uint64', 'int64'):
        assert execute(f'{{ {name} {{ values }} }}') == {name: {'values': [0, None]}}
        assert execute(f'{{ {name} {{ count(equal: 0) }} }}') == {name: {'count': 1}}
        data = execute(f'{{ {name} {{ fillNull(value: 1) {{ values }} }} }}')
        assert data == {name: {'fillNull': {'values': [0, 1]}}}
        assert execute(f'{{ {name} {{ min max sort first: sort(length: 1) }} }}')

    for name in ('float', 'double'):
        assert execute(f'{{ {name} {{ values }} }}') == {name: {'values': [0.0, None]}}
        assert execute(f'{{ {name} {{ count(equal: 0.0) }} }}') == {name: {'count': 1}}
        data = execute(f'{{ {name} {{ fillNull(value: 1.0) {{ values }} }} }}')
        assert data == {name: {'fillNull': {'values': [0.0, 1.0]}}}
        assert execute(f'{{ {name} {{ min max sort first: sort(length: 1) }} }}')
    assert execute('{ decimal { values } }') == {'decimal': {'values': ['0', None]}}

    for name in ('date32', 'date64'):
        assert execute(f'{{ {name} {{ values }} }}') == {name: {'values': ['1970-01-01', None]}}
        assert execute(f'{{ {name} {{ count(equal: "1970-01-01") }} }}') == {name: {'count': 1}}
        data = execute(f'{{ {name} {{ fillNull(value: "1970-01-02") {{ values }} }} }}')
        assert data == {name: {'fillNull': {'values': ['1970-01-01', '1970-01-02']}}}
        assert execute(f'{{ {name} {{ min max }} }}')

    data = execute('{ timestamp { values } }')
    assert data == {'timestamp': {'values': ['1970-01-01T00:00:00', None]}}
    data = execute('{ timestamp { fillNull(value: "1970-01-02T00:00:00") { values } } }')
    assert data == {
        'timestamp': {'fillNull': {'values': ['1970-01-01T00:00:00', '1970-01-02T00:00:00']}}
    }
    assert execute('{ timestamp { count(equal: "1970-01-01") } }') == {'timestamp': {'count': 1}}
    assert execute('{ timestamp { min max } }')

    for name in ('time32', 'time64'):
        assert execute(f'{{ {name} {{ values }} }}') == {name: {'values': ['00:00:00', None]}}
        assert execute(f'{{ {name} {{ count(equal: "00:00:00") }} }}') == {name: {'count': 1}}
        data = execute(f'{{ {name} {{ fillNull(value: "00:00:01") {{ values }} }} }}')
        assert data == {name: {'fillNull': {'values': ['00:00:00', '00:00:01']}}}
        assert execute(f'{{ {name} {{ min max }} }}')

    assert execute('{ binary { values } }') == {'binary': {'values': ['', None]}}
    assert execute('{ binary { count(equal: "") } }') == {'binary': {'count': 1}}
    assert execute('{ string { values } }') == {'string': {'values': ['', None]}}
    assert execute('{ string { count(equal: "") } }') == {'string': {'count': 1}}
    assert execute('{ string { count(stringIsAscii: true) } }') == {'string': {'count': 1}}
    assert execute('{ string { count(utf8IsAlnum: false) } }') == {'string': {'count': 0}}
    assert execute('{ string { count(utf8IsAlpha: true) } }') == {'string': {'count': 0}}
    assert execute('{ string { count(utf8IsDigit: true) } }') == {'string': {'count': 0}}
    assert execute('{ string { type } }') == {
        'string': {'type': 'dictionary<values=string, indices=int32, ordered=0>'}
    }
    assert execute('{ string { min max sort first: sort(length: 1) } }')


def test_numeric(executor):
    for name in ('int32', 'int64', 'float'):
        data = executor(f'{{ columns {{ {name} {{ add(value: 1) {{ sum }} }} }} }}')
        assert data == {'columns': {name: {'add': {'sum': 1}}}}
        data = executor(f'{{ columns {{ {name} {{ subtract(value: 1) {{ sum }} }} }} }}')
        assert data == {'columns': {name: {'subtract': {'sum': 1}}}}
        data = executor(f'{{ columns {{ {name} {{ multiply(value: 1) {{ sum }} }} }} }}')
        assert data == {'columns': {name: {'multiply': {'sum': 0}}}}
        with pytest.raises(ValueError):
            executor(f'{{ columns {{ {name} {{ divide(value: 1) {{ sum }} }} }} }}')

        data = executor(f'{{ columns {{ {name} {{ minimum(value: -1) {{ sum }} }} }} }}')
        assert data == {'columns': {name: {'minimum': {'sum': -1}}}}
        data = executor(f'{{ columns {{ {name} {{ maximum(value: 1) {{ sum }} }} }} }}')
        assert data == {'columns': {name: {'maximum': {'sum': 1}}}}
        data = executor(f'{{ columns {{ {name} {{ absolute {{ sum }} }} }} }}')
        assert data == {'columns': {name: {'absolute': {'sum': 0}}}}
        data = executor(f'{{ columns {{ {name} {{ mean mode stddev variance }} }} }}')
        assert data == {'columns': {name: {'mean': 0.0, 'mode': 0, 'stddev': 0.0, 'variance': 0.0}}}

    data = executor(
        '''{ apply(int32: {fillNull: -1, alias: "i"})
        { column(name: "i") { type ... on IntColumn { values } } } }'''
    )
    assert data == {'apply': {'column': {'type': 'int32', 'values': [0, -1]}}}
    data = executor('{ columns { float { add(value: 2.0) { divide(value: 1.0) { sum } } } } }')
    assert data == {'columns': {'float': {'add': {'divide': {'sum': 0.5}}}}}


def test_duration(executor):
    data = executor(
        '''{ apply(timestamp: {fillNull: "0001-01-01"})
        { columns { timestamp { values subtract(value: "0001-01-01")
        { values quantile(q: [0.5]) } } } } }'''
    )
    column = data['apply']['columns']['timestamp']
    assert column['values'] == ['1970-01-01T00:00:00', '0001-01-01T00:00:00']
    assert column['subtract'] == {'values': [-62135596800.0, 0.0], 'quantile': [-31067798400.0]}
    data = executor(
        '''{ apply(timestamp: {alias: "diff", subtract: "timestamp"}) { column(name: "diff")
        { ... on DurationColumn { values count(equal: 0.0) } } } }'''
    )
    column = data['apply']['column']
    assert column['values'] == [0.0, None]
    assert column['count'] == 1
    data = executor(
        '''{ filter(query: {timestamp: {duration: {equal: 0.0}, apply: {subtract: "timestamp"}}})
        { length } }'''
    )
    assert data == {'filter': {'length': 1}}


def test_list(executor):
    data = executor('{ columns { list { length type } } }')
    assert data == {'columns': {'list': {'length': 2, 'type': 'list<item: int32>'}}}
    data = executor('{ columns { list { values { length } } } }')
    assert data == {'columns': {'list': {'values': [{'length': 3}, None]}}}
    data = executor('{ row { list { ... on IntColumn { values } } } }')
    assert data == {'row': {'list': {'values': [0, 1, 2]}}}
    data = executor('{ row(index: -1) { list { ... on IntColumn { values } } } }')
    assert data == {'row': {'list': None}}
    data = executor('{ columns { list { unique { flatten { ... on IntColumn { values } } } } } }')
    assert data == {'columns': {'list': {'unique': {'flatten': {'values': [0, 1, 2]}}}}}

    data = executor(
        '''{ columns { list { count { values }
        first { ... on IntColumn { values } } last { ... on IntColumn { values } }
        min { ... on IntColumn { values } } max { ... on IntColumn { values } }
        sum { ... on IntColumn { values } } mean { values }
        mode { ... on IntColumn { values } } stddev { values } variance { values } } } }'''
    )
    assert data['columns']['list'] == {
        'count': {'values': [3, None]},
        'first': {'values': [0, None]},
        'last': {'values': [2, None]},
        'min': {'values': [0, None]},
        'max': {'values': [2, None]},
        'sum': {'values': [3, None]},
        'mean': {'values': [1.0, None]},
        'mode': {'values': [0, None]},
        'stddev': {'values': [pytest.approx((2 / 3) ** 0.5), None]},
        'variance': {'values': [pytest.approx(2 / 3), None]},
    }
    data = executor(
        '''{ apply(list: {count: true}) {
        column(name: "list") { ... on IntColumn { values } } } }'''
    )
    assert data == {'apply': {'column': {'values': [3, None]}}}
    data = executor(
        '''{ apply(list: {first: true, alias: "first"}) {
        column(name: "first") { ... on IntColumn { values } } } }'''
    )
    assert data == {'apply': {'column': {'values': [0, None]}}}
    data = executor('{ apply(list: {unique: true}) { columns { list { count { values } } } } }')
    assert data == {'apply': {'columns': {'list': {'count': {'values': [3, 0]}}}}}


def test_struct(executor):
    data = executor('{ columns { struct { names column(name: "x") { length } } } }')
    assert data == {'columns': {'struct': {'names': ['x', 'y'], 'column': {'length': 2}}}}


def test_dictionary(executor):
    data = executor(
        '''{ group(by: ["camelId"]) { aggregate { column(name: "string") {
        ... on ListColumn { unique { count { values } } } } } } }'''
    )
    assert data == {'group': {'aggregate': {'column': {'unique': {'count': {'values': [1, 1]}}}}}}
    data = executor(
        '''{ group(by: ["camelId"]) { aggregate(unique: {name: "string"}) { column(name: "string") {
        ... on ListColumn { count { values } } } } } }'''
    )
    assert data == {'group': {'aggregate': {'column': {'count': {'values': [1, 1]}}}}}
    data = executor(
        '''{ group(by: ["camelId"]) { aggregate(unique: {name: "string", count: true})
        { column(name: "string") { ... on IntColumn { values } } } } }'''
    )
    assert data == {'group': {'aggregate': {'column': {'values': [1, 1]}}}}
