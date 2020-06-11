import graphql
from strawberry.printer import print_type
from graphique import models


def test_schema(schema):
    assert graphql.parse(schema)


def test_columns(execute):
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


def test_boolean(schema):
    assert print_type(models.BooleanQuery) in schema
    assert print_type(models.BooleanSet) in schema
    assert print_type(models.BooleanColumn) in schema


def test_int(schema):
    assert print_type(models.IntQuery) in schema
    assert print_type(models.IntSet) in schema
    assert print_type(models.IntColumn) in schema


def test_long(schema):
    assert print_type(models.LongQuery) in schema
    assert print_type(models.LongSet) in schema
    assert print_type(models.LongColumn) in schema


def test_float(schema):
    assert print_type(models.FloatQuery) in schema
    assert print_type(models.FloatColumn) in schema


def test_decimal(schema):
    assert print_type(models.DecimalQuery) in schema
    assert print_type(models.DecimalColumn) in schema


def test_date(schema):
    assert print_type(models.DateQuery) in schema
    assert print_type(models.DateSet) in schema
    assert print_type(models.DateColumn) in schema


def test_timestamp(schema):
    assert print_type(models.DateTimeQuery) in schema
    assert print_type(models.DateTimeColumn) in schema


def test_time(schema):
    assert print_type(models.TimeQuery) in schema
    assert print_type(models.TimeColumn) in schema


def test_binary(schema):
    assert print_type(models.BinaryQuery) in schema
    assert print_type(models.BinaryColumn) in schema


def test_string(schema):
    assert print_type(models.StringQuery) in schema
    assert print_type(models.StringSet) in schema
    assert print_type(models.StringColumn) in schema
