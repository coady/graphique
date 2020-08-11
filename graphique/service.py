import enum
import functools
import itertools
from datetime import datetime
from typing import Callable, List, Optional
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import strawberry.asgi
from starlette.applications import Starlette
from starlette.middleware import Middleware, base
from strawberry.types.types import ArgumentDefinition, undefined
from strawberry.utils.str_converters import to_camel_case
from .core import Column as C, Table as T
from .models import (
    Long,
    column_map,
    doc_field,
    filter_map,
    query_map,
    resolve_arguments,
    selections,
    type_map,
)
from .settings import COLUMNS, DEBUG, DICTIONARIES, INDEX, MMAP, PARQUET_PATH

table = pq.read_table(
    PARQUET_PATH, COLUMNS, memory_map=MMAP, use_legacy_dataset=True, read_dictionary=DICTIONARIES
)
indexed = T.index(table) if INDEX is None else list(INDEX)
types = {name: type_map[tp.id] for name, tp in T.types(table).items()}
case_map = {to_camel_case(name): name for name in types}
to_snake_case = case_map.__getitem__


def numeric_field(func):
    arguments = [
        ArgumentDefinition(origin_name=name, type=Optional[str])
        for name in ('add', 'subtract', 'multiply')
    ]
    return resolve_arguments(func, arguments)


def resolver(name):
    cls = column_map[types[name]]
    if types[name] not in (int, float, Long):

        def method(self) -> cls:
            return cls(self.table[name])

        method.__name__ = name
        return strawberry.field(method)

    def method(self, **fields) -> cls:
        """Return column with optional projection."""
        column = self.table[name]
        for func in fields:
            column = getattr(pc, func)(column, self.table[fields[func]])
        return cls(column)

    method.__name__ = name
    return numeric_field(method)


@strawberry.type(description="fields for each column")
class Columns:
    locals().update({name: resolver(name) for name in types})

    def __init__(self, table):
        self.table = table


@strawberry.type(description="scalar fields")
class Row:
    __annotations__ = {name: Optional[types[name]] for name in types}
    locals().update(dict.fromkeys(types))


def query_field(func: Callable) -> Callable:
    arguments = [
        ArgumentDefinition(origin_name=name, type=Optional[query_map[types[name]]])
        for name in indexed
    ]
    return resolve_arguments(func, arguments)


@strawberry.input(description="predicates for each column")
class Filters:
    __annotations__ = {name: Optional[filter_map[types[name]]] for name in types}
    locals().update(dict.fromkeys(types, undefined))
    asdict = next(iter(query_map.values())).asdict


@strawberry.enum
class Operator(enum.Enum):
    AND = 'and'
    OR = 'or'
    XOR = 'xor'


def references(node):
    """Generate every possible column reference."""
    yield node.name.value
    value = getattr(node, 'value', None)
    yield getattr(value, 'value', None)
    for val in getattr(value, 'values', []):
        yield val.value
    nodes = itertools.chain(
        getattr(node, 'arguments', []),
        getattr(value, 'fields', []),
        getattr(getattr(node, 'selection_set', None), 'selections', []),
    )
    for node in nodes:
        yield from references(node)


@strawberry.type(description="a column-oriented table")
class Table:
    def __init__(self, table):
        self.table = table

    def select(self, info) -> pa.Table:
        """Return table with only the columns necessary to proceed."""
        names = map(case_map.get, references(*info.field_nodes))
        return self.table.select(set(names) - {None})

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
    def slice(
        self, info, offset: Long = 0, length: Optional[Long] = None  # type: ignore
    ) -> 'Table':
        """Return table slice."""
        table = self.select(info)
        return Table(table.slice(offset, length))

    @doc_field
    def group(
        self, info, by: List[str], reverse: bool = False, length: Optional[Long] = None
    ) -> List['Table']:
        """Return tables grouped by columns, with stable ordering."""
        tables = [self.select(info)]
        for name in map(to_snake_case, by):
            groups = (T.group(table, name, reverse) for table in tables)
            tables = list(itertools.islice(itertools.chain.from_iterable(groups), length))
        return list(map(Table, tables))

    @doc_field
    def unique(self, info, by: List[str], reverse: bool = False) -> 'Table':
        """Return table of first or last occurrences grouped by columns, with stable ordering."""
        name = to_snake_case(by[-1])
        groups = self.group(info, by[:-1], reverse=reverse)
        tables = [T.unique(group.table, name, reverse) for group in groups]
        return Table(pa.concat_tables(tables[::-1] if reverse else tables))

    @doc_field
    def sort(
        self, info, by: List[str], reverse: bool = False, length: Optional[Long] = None
    ) -> 'Table':
        """Return table slice sorted by specified columns.
        Optimized for a single column with fixed length.
        """
        table = self.select(info)
        return Table(T.sort(table, *map(to_snake_case, by), reverse=reverse, length=length))

    @doc_field
    def min(self, info, by: List[str]) -> 'Table':
        """Return table with minimum values per column."""
        table = self.select(info)
        return Table(T.matched(table, C.min, *map(to_snake_case, by)))

    @doc_field
    def max(self, info, by: List[str]) -> 'Table':
        """Return table with maximum values per column."""
        table = self.select(info)
        return Table(T.matched(table, C.max, *map(to_snake_case, by)))

    @doc_field
    def filter(
        self, info, query: Filters, invert: bool = False, reduce: Operator = 'and'  # type: ignore
    ) -> 'Table':
        """Return table with rows which match all (by default) queries."""
        table = self.select(info)
        masks = list(T.masks(table, **query.asdict()))  # type: ignore
        if not masks:
            return self
        mask = functools.reduce(lambda *args: pc.call_function(reduce.value, args), masks)
        return Table(table.filter(pc.call_function('invert', [mask]) if invert else mask))


@strawberry.type(description="a table sorted by a composite index")
class IndexedTable(Table):
    def __init__(self, table):
        self.table = table

    @doc_field
    def index(self) -> List[str]:
        """indexed columns"""
        return list(map(to_camel_case, indexed))

    @query_field
    def search(self, info, **queries) -> Table:
        """Return table with matching values for compound `index`.
        Queries must be a prefix of the `index`.
        Only one non-equal query is allowed, and applied last.
        """
        table = self.select(info)
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


class TimingMiddleware(base.BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):  # pragma: no cover
        start = datetime.now()
        try:
            return await call_next(request)
        finally:
            end = datetime.now()
            print(f"[{end.replace(microsecond=0)}]: {end - start}")


Query = IndexedTable if indexed else Table
schema = strawberry.Schema(query=Query)
app = Starlette(debug=DEBUG, middleware=[Middleware(TimingMiddleware)] * DEBUG)
app.add_route('/graphql', strawberry.asgi.GraphQL(schema, root_value=Query(table), debug=DEBUG))
