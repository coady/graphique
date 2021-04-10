"""
GraphQL service and top-level resolvers.
"""
import functools
import itertools
from pathlib import Path
from typing import Callable, List, Optional
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import strawberry
from starlette.applications import Starlette
from starlette.middleware import Middleware
from strawberry.types.types import undefined
from strawberry.utils.str_converters import to_camel_case
from .core import Column as C, ListChunk, Table as T, rpartial
from .inputs import Field, Filter, asdict, diff_map, filter_map, function_map, query_map
from .middleware import AbstractTable, GraphQL, TimingMiddleware, references
from .models import Column, ListColumn
from .models import annotate, column_map, doc_field, resolve_annotations, selections
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

    def method(self, **fields) -> cls:
        column = self.table[name]
        for func in fields:
            column = T.projected[func](column, self.table[to_snake_case(fields[func])])
        return cls(column)

    method.__name__ = name
    annotations = [key for key in T.projected if cls.__annotations__.get(key) == cls.__name__]
    if annotations:
        method.__doc__ = "Return column with optional projection."
    return resolve_annotations(method, dict.fromkeys(annotations, Optional[str]))


@strawberry.type(description="fields for each column")
class Columns:
    locals().update({name: resolver(name) for name in types})

    def __init__(self, table):
        self.table = table


@strawberry.type(description="scalar fields")
class Row:
    __annotations__ = {
        name: Optional[Column if types[name] is list else types[name]]
        for name in types
        if types[name] is not dict
    }
    locals().update(dict.fromkeys(types))


def query_field(func: Callable) -> Callable:
    annotations = {name: Optional[query_map[types[name]]] for name in indexed}
    return resolve_annotations(func, annotations)


def function_field(func: Callable) -> Callable:
    annotations = {
        name: Optional[function_map[types[name]]] for name in types if types[name] in function_map
    }
    return resolve_annotations(func, annotations)


@strawberry.input(description="predicates for each column")
class Filters:
    __annotations__ = {
        name: Optional[filter_map[types[name]]] for name in types if types[name] in filter_map
    }
    locals().update(dict.fromkeys(types, undefined))
    asdict = asdict


@strawberry.input(description="discrete difference predicates for each column")
class Diffs:
    __annotations__ = {
        name: Optional[diff_map[types[name]]] for name in types if types[name] in diff_map
    }
    locals().update(dict.fromkeys(types, undefined))
    asdict = asdict


@strawberry.type(description="a column-oriented table")
class Table(AbstractTable):
    __init__ = AbstractTable.__init__

    def select(self, info) -> pa.Table:
        """Return table with only the columns necessary to proceed."""
        names = set(map(to_snake_case, references(*info.field_nodes)))
        return self.table.select(names & set(self.table.column_names))

    @doc_field
    def columns(self) -> Columns:
        """fields for each column"""
        return Columns(self.table)

    @doc_field
    def column(self, name: str) -> Column:
        """Return column of any type by name.
        This is typically only needed for aliased columns added by `apply` or `aggregate`.
        If the column is in the schema, `columns` can be used instead."""
        column = self.table[to_snake_case(name)]
        return column_map[type_map[column.type.id]](column)

    @doc_field
    def row(self, info, index: Long = 0) -> Row:  # type: ignore
        """Return scalar values at index."""
        row = {}
        for name in map(to_snake_case, selections(*info.field_nodes)):
            scalar = self.table[name][index]
            row[name] = (
                Column.fromlist(scalar) if isinstance(scalar, pa.ListScalar) else scalar.as_py()
            )
        return Row(**row)  # type: ignore

    @doc_field
    def slice(self, info, offset: Long = 0, length: Optional[Long] = None) -> 'Table':  # type: ignore
        """Return table slice."""
        table = self.select(info)
        return type(self)(table.slice(offset, length))

    @doc_field(
        reverse="return groups in reversed stable order",
        length="maximum number of groups to return",
        count="optionally include counts in an aliased column",
    )
    def group(
        self,
        info,
        by: List[str],
        reverse: bool = False,
        length: Optional[Long] = None,
        count: str = '',
    ) -> 'Table':
        """Return table grouped by columns, with stable ordering.
        Other columns can be accessed by the `column` field as a `ListColumn`.
        Typically used in conjunction with `aggregate` or `tables`."""
        table = self.select(info)
        table, counts = T.group(table, *map(to_snake_case, by), reverse=reverse, length=length)
        return Table(table.add_column(len(table.columns), count, counts) if count else table)

    @doc_field(
        diffs="predicates defaulting to `not_equal`; scalars are compared to the adjacent difference",
        count="optionally include counts in an aliased column",
    )
    def partition(
        self, info, by: List[str], diffs: Optional[Diffs] = None, count: str = ''
    ) -> 'Table':
        """Return table partitioned by discrete differences of the values.
        Differs from `group` by relying on adjacency, and is typically faster.
        Other columns can be accessed by the `column` field as a `ListColumn`.
        Typically used in conjunction with `aggregate` or `tables`."""
        table = self.select(info)
        funcs = diffs.asdict() if diffs else {}
        names = list(map(to_snake_case, itertools.takewhile(lambda name: name not in funcs, by)))
        predicates = {}
        for name in by[len(names) :]:  # noqa: E203
            ((func, value),) = funcs.pop(name, {'not_equal': None}).items()
            func = getattr(pc, func)
            predicates[to_snake_case(name)] = func if value is None else rpartial(func, value)
        table, counts = T.partition(table, *names, **predicates)
        return Table(table.add_column(len(table.columns), count, counts) if count else table)

    @doc_field(
        reverse="return last occurrences in reversed order",
        length="maximum number of rows to return",
        count="optionally include counts in an aliased column",
    )
    def unique(
        self,
        info,
        by: List[str],
        reverse: bool = False,
        length: Optional[Long] = None,
        count: str = '',
    ) -> 'Table':
        """Return table of first or last occurrences grouped by columns, with stable ordering.
        Faster than `group` when only scalars are needed."""
        table = self.select(info)
        names = map(to_snake_case, by)
        if selections(*info.field_nodes) == {'length'}:  # optimized for count
            return Table(T.unique_indices(table, *names)[0][:length])
        table, counts = T.unique(table, *names, reverse=reverse, length=length, count=bool(count))
        return Table(table.add_column(len(table.columns), count, counts) if count else table)

    @doc_field(
        reverse="descending stable order",
        length="maximum number of rows to return; may be significantly faster on a single column",
    )
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

    @doc_field(
        query="filters organized by column",
        invert="optionally exclude matching rows",
        reduce="binary operator to combine filters; within a column all predicates must match",
        predicates="additional filters for columns of unknown types, as the result of `apply`",
    )
    def filter(
        self,
        info,
        query: Optional[Filters] = None,
        invert: bool = False,
        reduce: Operator = 'and',  # type: ignore
        predicates: List[Filter] = [],
    ) -> 'Table':
        """Return table with rows which match all (by default) queries.
        List columns apply their respective filters to their own scalar values."""
        table = self.select(info)
        filters = query.asdict() if query else {}
        for predicate in predicates:
            filters.update(predicate.asdict())
        masks = []
        for name, value in filters.items():
            apply = value.get('apply', {})
            apply.update({key: to_snake_case(apply[key]) for key in apply})
            mask = T.mask(table, name, **value)
            if isinstance(table[name].type, pa.ListType):
                column = pa.chunked_array(C.map(ListChunk.filter_list, table[name], mask))
                table = table.set_column(table.column_names.index(name), name, column)
            else:
                masks.append(mask)
        if not masks:
            return Table(table)
        mask = functools.reduce(getattr(pc, reduce.value), masks)
        if selections(*info.field_nodes) == {'length'}:  # optimized for count
            return Table(range(C.count(mask, not invert)))  # type: ignore
        return Table(table.filter(pc.invert(mask) if invert else mask))

    @function_field
    def apply(self, **functions) -> 'Table':
        """Return view of table with functions applied across columns.
        If no alias is provided, the column is replaced and should be of the same type.
        If an alias is provided, a column is added and may be referenced in the `column` field,
        in filter `predicates`, and in the `by` arguments of grouping and sorting."""
        table = self.table
        for name in functions:
            value = functions[name].asdict()
            value.update({key: to_snake_case(value[key]) for key in value if key in T.projected})
            table = T.apply(table, name, **value)
        return Table(table)

    @doc_field
    def tables(self, info) -> List['Table']:  # type: ignore
        """Return a list of tables by splitting list columns, typically used after grouping.
        At least one list column must be referenced, and all list columns must have the same lengths."""
        table = self.select(info)
        lists = {name for name in table.column_names if isinstance(table[name].type, pa.ListType)}
        columns = {name: table[name] for name in lists}
        # use simplest list column to determine the lengths
        shape = C.combine_chunks(min(columns.values(), key=lambda col: col.type.value_type.id))
        counts = shape.value_lengths().to_pylist()
        indices = pa.concat_arrays(pa.array(np.repeat(*pair)) for pair in enumerate(counts))
        for name in set(table.column_names) - lists:
            column = pa.DictionaryArray.from_arrays(indices, C.combine_chunks(table[name]))
            columns[name] = pa.ListArray.from_arrays(shape.offsets, C.decode(column, check=True))
        for index in range(len(table)):
            row = {name: columns[name][index].values for name in columns}
            yield Table(pa.Table.from_pydict(row))

    @doc_field(
        count=ListColumn.count.__doc__,
        first=ListColumn.first.__doc__,
        last=ListColumn.last.__doc__,
        min=ListColumn.min.__doc__,
        max=ListColumn.max.__doc__,
        sum=ListColumn.sum.__doc__,
        mean=ListColumn.mean.__doc__,
        any=ListColumn.any.__doc__,
        all=ListColumn.all.__doc__,
        unique=ListColumn.unique.__doc__,
    )
    def aggregate(
        self,
        info,
        count: List[Field] = [],
        first: List[Field] = [],
        last: List[Field] = [],
        min: List[Field] = [],
        max: List[Field] = [],
        sum: List[Field] = [],
        mean: List[Field] = [],
        any: List[Field] = [],
        all: List[Field] = [],
        unique: List[Field] = [],
    ) -> 'Table':
        """Return table with aggregate functions applied to list columns, typically used after grouping.
        Columns which are aliased or change type can be accessed by the `column` field."""
        table = self.select(info)
        columns = {name: table[name] for name in table.column_names}
        for key in ('count', 'first', 'last', 'min', 'max', 'sum', 'mean', 'any', 'all', 'unique'):
            func = getattr(ListChunk, key)
            for field in locals()[key]:
                name = to_snake_case(field.name)
                columns[field.alias or name] = pa.chunked_array(C.map(func, table[name]))
        return Table(pa.Table.from_pydict(columns))


@strawberry.type(description="a table sorted by a composite index")
class IndexedTable(Table):
    __init__ = AbstractTable.__init__
    slice = annotate(Table.slice, 'IndexedTable')

    @doc_field
    def index(self) -> List[str]:
        """the composite index"""
        return list(map(to_camel_case, indexed))

    @query_field
    def search(self, info, **queries) -> Table:
        """Return table with matching values for composite `index`.
        Queries must be a prefix of the `index`.
        Only one inequality query is allowed, and must be last."""
        table = self.select(info)
        for name in indexed:
            if name not in queries:
                break
            query = queries.pop(name).asdict()
            if 'equal' in query:
                table = T.is_in(table, name, query.pop('equal'))
            if query and queries:
                raise ValueError(f"inequality query for {name} not last; have {queries} remaining")
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
app = Starlette(debug=DEBUG, middleware=[Middleware(TimingMiddleware)] * DEBUG)
app.add_route('/graphql', GraphQL(Query(table), debug=DEBUG))
