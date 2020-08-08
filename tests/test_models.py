import pytest
from strawberry.printer import print_schema
from .conftest import fixtures


def test_schema(schema):
    with open(fixtures / 'schema.graphql', 'w') as file:
        file.write(print_schema(schema))


def test_case(executor):
    data = executor('{ columns { snakeId { values } camelId { values } } }')
    assert data == {'columns': {'snakeId': {'values': [1, 2]}, 'camelId': {'values': [1, 2]}}}
    data = executor('{ row { snakeId camelId } }')
    assert data == {'row': {'snakeId': 1, 'camelId': 1}}
    data = executor('{ filter(snakeId: {equal: 1}, camelId: {equal: 1}) { length } }')
    assert data == {'filter': {'length': 1}}
    data = executor('{ exclude(snakeId: {equal: 1}, camelId: {equal: 1}) { length } }')
    assert data == {'exclude': {'length': 1}}
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

    for name in ('uint8', 'int8', 'uint16', 'int16', 'int32'):
        assert execute(f'{{ {name} {{ values }} }}') == {name: {'values': [0, None]}}
        assert execute(f'{{ {name} {{ count(equal: 0) }} }}') == {name: {'count': 1}}
    for name in ('uint32', 'uint64', 'int64'):
        assert execute(f'{{ {name} {{ values }} }}') == {name: {'values': [0, None]}}
        assert execute(f'{{ {name} {{ count(equal: 0) }} }}') == {name: {'count': 1}}

    for name in ('float', 'double'):
        assert execute(f'{{ {name} {{ values }} }}') == {name: {'values': [0.0, None]}}
        assert execute(f'{{ {name} {{ count(equal: 0.0) }} }}') == {name: {'count': 1}}
    assert execute('{ decimal { values } }') == {'decimal': {'values': ['0', None]}}

    for name in ('date32', 'date64'):
        assert execute(f'{{ {name} {{ values }} }}') == {name: {'values': ['1970-01-01', None]}}
        assert execute(f'{{ {name} {{ count(equal: "1970-01-01") }} }}') == {name: {'count': 1}}
    assert execute('{ timestamp { values } }') == {
        'timestamp': {'values': ['1970-01-01T00:00:00', None]}
    }
    assert execute('{ timestamp { count(equal: "1970-01-01") } }') == {'timestamp': {'count': 1}}
    for name in ('time32', 'time64'):
        assert execute(f'{{ {name} {{ values }} }}') == {name: {'values': ['00:00:00', None]}}
        assert execute(f'{{ {name} {{ count(equal: "00:00:00") }} }}') == {name: {'count': 1}}

    assert execute('{ binary { values } }') == {'binary': {'values': ['', None]}}
    assert execute('{ binary { count(equal: "") } }') == {'binary': {'count': 1}}
    assert execute('{ string { values } }') == {'string': {'values': ['', None]}}
    assert execute('{ string { count(equal: "") } }') == {'string': {'count': 1}}
    assert execute('{ string { count(stringIsAscii: true) } }') == {'string': {'count': 1}}
    assert execute('{ string { count(utf8IsAlnum: false) } }') == {'string': {'count': 0}}
    assert execute('{ string { count(utf8IsAlpha: true) } }') == {'string': {'count': 0}}
    assert execute('{ string { count(utf8IsDigit: true) } }') == {'string': {'count': 0}}


def test_numeric(executor):
    for name in ('int32', 'int64', 'float'):
        data = executor(f'{{ columns {{ {name} {{ add(value: 1) {{ sum }} }} }} }}')
        assert data == {'columns': {name: {'add': {'sum': 1}}}}

        data = executor(f'{{ columns {{ {name} {{ subtract(value: 1) {{ sum }} }} }} }}')
        assert data == {'columns': {name: {'subtract': {'sum': 1}}}}

        data = executor(f'{{ columns {{ {name} {{ multiply(value: 1) {{ sum }} }} }} }}')
        assert data == {'columns': {name: {'multiply': {'sum': 0}}}}
