import functools
from typing import List, Optional
import pyarrow as pa
import pyarrow.parquet as pq
import strawberry.asgi
from starlette.applications import Starlette
from strawberry.types.type_resolver import resolve_type
from strawberry.types.types import ArgumentDefinition, FieldDefinition
from strawberry.utils.str_converters import to_camel_case
from .core import Column as C, Table as T
from .models import Long, column_map, query_map, selections, type_map
from .settings import COLUMNS, DEBUG, DICTIONARIES, INDEX, MMAP, PARQUET_PATH

table = pq.read_table(
    PARQUET_PATH, COLUMNS, memory_map=MMAP, use_legacy_dataset=True, read_dictionary=DICTIONARIES
)
indexed = T.index(table) if INDEX is None else list(INDEX)
types = {name: type_map[tp.id] for name, tp in T.types(table).items()}
to_snake_case = {to_camel_case(name): name for name in types}.__getitem__


def resolver(name):
    column = column_map[types[name]]

    def method(self) -> column:
        return column(self.table[name])

    method.__name__ = name
    return strawberry.field(method)


@strawberry.type(description="fields for each column")
class Columns:
    locals().update({name: resolver(name) for name in types})

    def __init__(self, table):
        self.table = table


@strawberry.type(description="scalar fields")
class Row:
    __annotations__ = {name: Optional[types[name]] for name in types}
    locals().update(dict.fromkeys(types))


def query_field(func, names=types):
    arguments = [
        ArgumentDefinition(
            name=to_camel_case(name),
            origin_name=name,
            type=Optional[query_map[types[name]]],
            origin=func,
        )
        for name in names
    ]
    for argument in arguments:
        resolve_type(argument)
    func._field_definition = FieldDefinition(
        name=to_camel_case(func.__name__),
        origin_name=func.__name__,
        type=func.__annotations__['return'],
        origin=func,
        arguments=arguments,
        description=func.__doc__,
        base_resolver=func,
    )
    return func


def doc_field(func):
    return strawberry.field(func, description=func.__doc__)


@strawberry.type(description="a column-oriented table")
class Table:
    def __init__(self, table):
        self.table = table

    @doc_field
    def length(self) -> Long:
        """number of rows"""
        return len(self.table)  # type: ignore

    @doc_field
    def columns(self) -> Columns:
        """fields for each column"""
        return Columns(self.table)

    @doc_field
    def row(self, info, index: Long = 0) -> Row:  # type: ignore
        """Return scalar values at index."""
        names = map(to_snake_case, selections(*info.field_nodes))
        return Row(**{name: self.table[name][index].as_py() for name in names})  # type: ignore

    @doc_field
    def slice(self, offset: Long = 0, length: Optional[Long] = None) -> 'Table':  # type: ignore
        """Return table slice."""
        return Table(self.table.slice(offset, length))

    @doc_field
    def group(
        self, by: List[str], reverse: bool = False, length: Optional[Long] = None
    ) -> List['Table']:
        """Return tables grouped by columns, with stable ordering."""
        tables = T.grouped(self.table, *map(to_snake_case, by), reverse=reverse, length=length)
        return list(map(Table, tables))

    @doc_field
    def unique(self, by: List[str], reverse: bool = False) -> 'Table':
        """Return table of first or last occurrences grouped by columns, with stable ordering."""
        by = list(map(to_snake_case, by))
        tables = [T.unique(table, by[-1], reverse) for table in T.grouped(self.table, *by[:-1])]
        return Table(pa.concat_tables(tables[::-1] if reverse else tables))

    @doc_field
    def sort(self, by: List[str], reverse: bool = False, length: Optional[Long] = None) -> 'Table':
        """Return table slice sorted by specified columns.
        Optimized for a single column with fixed length.
        """
        return Table(T.sort(self.table, *map(to_snake_case, by), reverse=reverse, length=length))

    @doc_field
    def min(self, by: List[str]) -> 'Table':
        """Return table with minimum values per column."""
        return Table(T.matched(self.table, C.min, *map(to_snake_case, by)))

    @doc_field
    def max(self, by: List[str]) -> 'Table':
        """Return table with maximum values per column."""
        return Table(T.matched(self.table, C.max, *map(to_snake_case, by)))

    @query_field
    def filter(self, **queries) -> 'Table':
        """Return table with rows which match all queries."""
        queries = {name: queries[name].asdict() for name in queries}
        return Table(T.filtered(self.table, queries, invert=False))

    @query_field
    def exclude(self, **queries) -> 'Table':
        """Return table with rows which don't match all queries; inverse of filter."""
        queries = {name: queries[name].asdict() for name in queries}
        return Table(T.filtered(self.table, queries, invert=True))


@strawberry.type(description="a table sorted by a composite index")
class IndexedTable(Table):
    def __init__(self, table):
        self.table = table

    @doc_field
    def index(self) -> List[str]:
        """indexed columns"""
        return list(map(to_camel_case, indexed))

    @functools.partial(query_field, names=indexed)
    def search(self, **queries) -> Table:
        """Return table with matching values for compound `index`.
        Queries must be a prefix of the `index`.
        Only one non-equal query is allowed, and applied last.
        """
        table = self.table
        for name in indexed:
            query = queries.pop(name, None)
            if query is None:
                break
            query = query.asdict()
            if 'equal' in query:
                table = T.is_in(table, name, query.pop('equal'))
            if query and queries:
                raise ValueError(f"non-equal query for {name} not last; have {queries} remaining")
            if 'not_equal' in query:
                table = T.not_equal(table, name, query['not_equal'])
            if 'is_in' in query:
                table = T.is_in(table, name, *query['is_in'])
            lower, upper = query.get('greater'), query.get('less')
            includes = {'include_lower': False, 'include_upper': False}
            if 'greater_equal' in query and (lower is None or query['greater_equal'] > lower):
                lower, includes['include_lower'] = query['greater_equal'], True
            if 'less_equal' in query and (upper is None or query['less_equal'] > upper):
                upper, includes['include_upper'] = query['less_equal'], True
            if {lower, upper} != {None}:
                table = T.range(table, name, lower, upper, **includes)
        if queries:
            raise ValueError(f"expected query for {name}; have {queries} remaining")
        return Table(table)


Query = IndexedTable if indexed else Table
schema = strawberry.Schema(query=Query)
app = Starlette(debug=DEBUG)
app.add_route('/graphql', strawberry.asgi.GraphQL(schema, root_value=Query(table), debug=DEBUG))
