"""
GraphQL service and top-level resolvers.
"""
import functools
import itertools
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, get_type_hints
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import strawberry.asgi
from starlette.applications import Starlette
from starlette.middleware import Middleware, base
from strawberry.types.types import ArgumentDefinition, undefined
from strawberry.utils.str_converters import to_camel_case
from .core import Column as C, Table as T
from .inputs import LongReduce, filter_map, query_map, function_map
from .models import Column, column_map, doc_field, resolve_arguments, selections
from .scalars import Long, Operator, type_map
from .settings import COLUMNS, DEBUG, DICTIONARIES, INDEX, MMAP, PARQUET_PATH

path = Path(PARQUET_PATH).resolve()
table = pq.ParquetDataset(path, memory_map=MMAP, read_dictionary=DICTIONARIES).read(COLUMNS)
indexed = T.index(table) if INDEX is None else list(INDEX)
types = {name: type_map[tp.id] for name, tp in T.types(table).items()}
case_map = {to_camel_case(name): name for name in types}
for name in indexed:
    assert not table[name].null_count  # binary search requires non-null columns


def to_snake_case(name):
    return case_map.get(name, name)


def resolver(name):
    cls = column_map[types[name]]
    arguments = [
        ArgumentDefinition(origin_name=field, type=Optional[str])
        for field in T.projected
        if hasattr(cls, field) and get_type_hints(getattr(cls, field))['return'] is cls
    ]

    def method(self, **fields) -> cls:
        column = self.table[name]
        for func in fields:
            column = T.projected[func](column, self.table[to_snake_case(fields[func])])
        return cls(column)

    method.__name__ = name
    if arguments:
        method.__doc__ = "Return column with optional projection."
    return resolve_arguments(method, arguments)


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


def function_field(func: Callable) -> Callable:
    arguments = [
        ArgumentDefinition(origin_name=name, type=Optional[function_map[types[name]]])
        for name in types
        if types[name] in function_map
    ]
    return resolve_arguments(func, arguments)


@strawberry.input(description="predicates for each column")
class Filters:
    __annotations__ = {name: Optional[filter_map[types[name]]] for name in types}
    locals().update(dict.fromkeys(types, undefined))
    asdict = next(iter(query_map.values())).asdict


def references(node):
    """Generate every possible column reference."""
    if hasattr(node, 'name'):
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
        names = set(map(to_snake_case, references(*info.field_nodes)))
        return self.table.select(names & set(self.table.column_names))

    @doc_field
    def length(self) -> Long:
        """number of rows"""
        return len(self.table)  # type: ignore

    @doc_field
    def columns(self) -> Columns:
        """fields for each column"""
        return Columns(self.table)

    @doc_field
    def column(self, alias: str) -> Column:
        """Return column by alias.
        This is only needed for aliased columns added by `apply`; otherwise see `columns`."""
        column = self.table[alias]
        return column_map[type_map[column.type.id]](column)

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
        self,
        info,
        by: List[str],
        reverse: bool = False,
        length: Optional[Long] = None,
        count: Optional[LongReduce] = None,
    ) -> List['Table']:
        """Return tables grouped by columns, with stable ordering.
        `length` is the maximum number of tables to return.
        `count` filters and sorts tables based on the number of rows within each table."""
        chain = itertools.chain.from_iterable
        tables = [self.select(info)]
        names = list(map(to_snake_case, by))
        if count is None:
            for name in names:
                tables = chain(T.group(table, name, reverse) for table in tables)  # type: ignore
        else:
            predicate = count.predicate(lower=True)
            for name in names[:-1]:
                tables = chain(T.group(table, name, predicate=predicate) for table in tables)  # type: ignore
            kwargs = {'predicate': count.predicate(), 'sort': count.sort}
            groups = [
                itertools.islice(T.group(table, names[-1], reverse, **kwargs), length)
                for table in tables
            ]
            if count.sort:
                tables = sorted(chain(groups), key=len, reverse=reverse)
            else:
                tables = chain(reversed(groups) if reverse else groups)  # type: ignore
        return list(map(Table, itertools.islice(tables, length)))

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
        """Return table slice sorted by specified columns."""
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
        """Return table with rows which match all (by default) queries.
        `invert` optionally excludes matching rows.
        `reduce` is the binary operator to combine filters; within a column all predicates must match."""
        table = self.select(info)
        masks = []
        for name, value in query.asdict().items():  # type: ignore
            apply = value.get('apply', {})
            apply.update({key: to_snake_case(apply[key]) for key in apply})
            masks.append(T.mask(table, name, **value))
        if not masks:
            return self
        mask = functools.reduce(lambda *args: pc.call_function(reduce.value, args), masks)
        if selections(*info.field_nodes) == {'length'}:  # optimized for count
            return Table(range(C.count(mask, not invert)))  # type: ignore
        return Table(table.filter(pc.call_function('invert', [mask]) if invert else mask))

    @function_field
    def apply(self, **functions) -> 'Table':
        """Return view of table with functions applied across columns.
        If no alias is provided, the column is replaced and must be of the same type.
        If an alias is provided, a column is added and may be referenced in the `column` interface,
        and in the `by` arguments of grouping and sorting."""
        table = self.table
        for name in functions:
            value = functions[name].asdict()
            value.update({key: to_snake_case(value[key]) for key in value if key in T.projected})
            table = T.apply(table, name, **value)
        return Table(table)


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
        Only one non-equal query is allowed, and applied last."""
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


class GraphQL(strawberry.asgi.GraphQL):
    def __init__(self, root_value, **kwargs):
        super().__init__(strawberry.Schema(type(root_value)), **kwargs)
        self.root_value = root_value

    async def get_root_value(self, request):
        return self.root_value


Query = IndexedTable if indexed else Table
app = Starlette(debug=DEBUG, middleware=[Middleware(TimingMiddleware)] * DEBUG)
app.add_route('/graphql', GraphQL(Query(table), debug=DEBUG))
