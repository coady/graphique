"""
GraphQL service and top-level resolvers.
"""
import functools
import itertools
from pathlib import Path
from typing import List, Optional, no_type_check
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import strawberry
from starlette.applications import Starlette
from starlette.middleware import Middleware
from strawberry.types.types import undefined
from strawberry.utils.str_converters import to_camel_case
from .core import Column as C, ListChunk, Table as T
from .inputs import Diff, Filters, Function, Input, Query as QueryInput
from .middleware import AbstractTable, GraphQL, TimingMiddleware
from .models import Column, ListColumn, annotate, doc_field, selections
from .scalars import Long, Operator, type_map
from .settings import COLUMNS, DEBUG, DICTIONARIES, INDEX, MMAP, PARQUET_PATH

path = Path(PARQUET_PATH).resolve()
table = pq.ParquetDataset(path, memory_map=MMAP, read_dictionary=DICTIONARIES).read(COLUMNS)
indexed = T.index(table) if INDEX is None else list(INDEX)
types = {name: type_map[tp.id] for name, tp in T.types(table).items()}
for name in indexed:
    assert not table[name].null_count, f"binary search requires non-null columns: {name}"


@strawberry.type(description="fields for each column")
class Columns:
    __annotations__ = {name: Column.type_map[types[name]] for name in types}  # type: ignore
    locals().update(dict.fromkeys(__annotations__))


@strawberry.type(description="scalar fields")
class Row:
    __annotations__ = {
        name: Optional[Column if types[name] is list else types[name]]
        for name in types
        if types[name] is not dict
    }
    locals().update(dict.fromkeys(__annotations__))


@strawberry.input(description="predicates for each column")
class Queries(Input):
    __annotations__ = QueryInput.annotations(types)
    locals().update(dict.fromkeys(__annotations__, undefined))


@strawberry.type(description="a column-oriented table")
class Table(AbstractTable):
    __init__ = AbstractTable.__init__

    @doc_field
    def columns(self, info) -> Columns:
        """fields for each column"""
        table = self.select(info)
        columns = {name: Columns.__annotations__[name](table[name]) for name in table.column_names}
        return Columns(**columns)  # type: ignore

    @doc_field
    def row(self, info, index: Long = 0) -> Row:  # type: ignore
        """Return scalar values at index."""
        row = {}
        for name in map(self.to_snake_case, selections(*info.field_nodes)):
            scalar = self.table[name][index]
            row[name] = (
                Column.fromscalar(scalar) if isinstance(scalar, pa.ListScalar) else scalar.as_py()
            )
        return Row(**row)  # type: ignore

    @doc_field
    def slice(self, info, offset: Long = 0, length: Optional[Long] = None) -> 'Table':  # type: ignore
        """Return table slice."""
        table = self.select(info)
        return type(self)(table.slice(offset, length))

    @doc_field(
        by="column names",
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
        table, counts = T.group(table, *map(self.to_snake_case, by), reverse=reverse, length=length)
        return Table(table.add_column(len(table.columns), count, counts) if count else table)

    @doc_field(
        by="column names",
        diffs="optional inequality predicates; scalars are compared to the adjacent difference",
        count="optionally include counts in an aliased column",
    )
    @no_type_check
    def partition(self, info, by: List[str], diffs: List[Diff] = [], count: str = '') -> 'Table':
        """Return table partitioned by discrete differences of the values.
        Differs from `group` by relying on adjacency, and is typically faster.
        Other columns can be accessed by the `column` field as a `ListColumn`.
        Typically used in conjunction with `aggregate` or `tables`."""
        table = self.select(info)
        funcs = {diff.pop('name'): diff for diff in map(dict, diffs)}
        names = itertools.takewhile(lambda name: name not in funcs, by)
        names = list(map(self.to_snake_case, names))
        predicates = {}
        for name in by[len(names) :]:  # noqa: E203
            ((func, value),) = funcs.pop(name, {'not_equal': None}).items()
            predicate = (getattr(pc, func),) + tuple((value or {}).values())
            predicates[self.to_snake_case(name)] = predicate
        table, counts = T.partition(table, *names, **predicates)
        return Table(table.add_column(len(table.columns), count, counts) if count else table)

    @doc_field(
        by="column names",
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
        names = map(self.to_snake_case, by)
        if selections(*info.field_nodes) == {'length'}:  # optimized for count
            return Table(T.unique_indices(table, *names)[0][:length])
        table, counts = T.unique(table, *names, reverse=reverse, length=length, count=bool(count))
        return Table(table.add_column(len(table.columns), count, counts) if count else table)

    @doc_field(
        by="column names",
        reverse="descending stable order",
        length="maximum number of rows to return; may be significantly faster on a single column",
    )
    def sort(
        self, info, by: List[str], reverse: bool = False, length: Optional[Long] = None
    ) -> 'Table':
        """Return table slice sorted by specified columns."""
        table = self.select(info)
        return Table(T.sort(table, *map(self.to_snake_case, by), reverse=reverse, length=length))

    @doc_field(by="column names")
    def min(self, info, by: List[str]) -> 'Table':
        """Return table with minimum values per column."""
        table = self.select(info)
        return Table(T.matched(table, C.min, *map(self.to_snake_case, by)))

    @doc_field(by="column names")
    def max(self, info, by: List[str]) -> 'Table':
        """Return table with maximum values per column."""
        table = self.select(info)
        return Table(T.matched(table, C.max, *map(self.to_snake_case, by)))

    @doc_field(
        query="simple queries by column",
        on="extended filters on columns organized by type",
        invert="optionally exclude matching rows",
        reduce="binary operator to combine filters; within a filter all predicates must match",
    )
    @no_type_check
    def filter(
        self,
        info,
        query: Optional[Queries] = (),
        on: Optional[Filters] = (),
        invert: bool = False,
        reduce: Operator = 'and',
    ) -> 'Table':
        """Return table with rows which match all (by default) queries.
        List columns apply their respective filters to their own scalar values."""
        table = self.select(info)
        filters = list(dict(query).items())
        for value in map(dict, itertools.chain(*dict(on).values())):
            filters.append((value.pop('name'), value))
        masks = []
        for name, value in filters:
            name = self.to_snake_case(name)
            apply = value.get('apply', {})
            apply.update({key: self.to_snake_case(apply[key]) for key in apply})
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
            return Table(range(C.count(mask, not invert)))
        return Table(table.filter(pc.invert(mask) if invert else mask))

    @Function.resolver
    @no_type_check
    def apply(self, info, **functions) -> 'Table':
        """Return view of table with functions applied across columns.
        If no alias is provided, the column is replaced and should be of the same type.
        If an alias is provided, a column is added and may be referenced in the `column` field,
        in filter `predicates`, and in the `by` arguments of grouping and sorting."""
        table = self.select(info)
        for value in map(dict, itertools.chain(*functions.values())):
            name = self.to_snake_case(value.pop('name'))
            for key in set(value) & set(T.projected):
                value[key] = self.to_snake_case(value[key])
            table = T.apply(table, name, **value)
        return Table(table)

    @doc_field
    def tables(self, info) -> List['Table']:  # type: ignore
        """Return a list of tables by splitting list columns, typically used after grouping.
        At least one list column must be referenced, and all list columns must have the same lengths."""
        table = self.select(info)
        lists = {name for name in table.column_names if isinstance(table[name].type, pa.ListType)}
        assert lists, f"no list columns referenced: {table.column_names}"
        columns = {name: table[name] for name in lists}
        # use simplest list column to determine the lengths
        shape = C.combine_chunks(min(columns.values(), key=lambda col: col.type.value_type.id))
        counts = shape.value_lengths().to_pylist()
        indices = pa.concat_arrays(pa.array(np.repeat(*pair)) for pair in enumerate(counts))
        for name in set(table.column_names) - lists:
            column = C.combine_chunks(table[name]).take(indices)
            columns[name] = pa.ListArray.from_arrays(shape.offsets, column)
        for index in range(len(table)):
            row = {name: columns[name][index].values for name in columns}
            yield Table(pa.Table.from_pydict(row))

    @ListColumn.resolver
    def aggregate(self, info, **fields) -> 'Table':
        """Return table with aggregate functions applied to list columns, typically used after grouping.
        Columns which are aliased or change type can be accessed by the `column` field."""
        table = self.select(info)
        columns = {name: table[name] for name in table.column_names}
        for key in fields:
            func = getattr(ListChunk, key)
            for field in fields[key]:
                name = self.to_snake_case(field.name)
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

    @QueryInput.resolve_types({name: types[name] for name in indexed})
    def search(self, info, **queries) -> Table:
        """Return table with matching values for composite `index`.
        Queries must be a prefix of the `index`.
        Only one inequality query is allowed, and must be last."""
        table = self.select(info)
        for name in indexed:
            if name not in queries:
                break
            query = dict(queries.pop(name))
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
