"""
GraphQL service and top-level resolvers.
"""
import functools
import itertools
from datetime import timedelta
from typing import List, Optional, no_type_check
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import strawberry
from strawberry.arguments import UNSET
from strawberry.utils.str_converters import to_camel_case
from .core import Column as C, ListChunk, Table as T
from .inputs import Diff, Filters, Function, Input, Query as QueryInput
from .middleware import AbstractTable, GraphQL, filter_expression
from .models import Column, ListColumn, doc_field, selections
from .scalars import Long, Operator, type_map
from .settings import COLUMNS, DEBUG, DICTIONARIES, FEDERATED, FILTERS, INDEX, PARQUET_PATH

format = ds.ParquetFileFormat(read_options={'dictionary_columns': DICTIONARIES})
table = dataset = ds.dataset(PARQUET_PATH, format=format, partitioning='hive')
indexed = list(map(to_camel_case, INDEX))
types = {to_camel_case(field.name): type_map[C.scalar_type(field).id] for field in dataset.schema}


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
    locals().update(dict.fromkeys(__annotations__, UNSET))


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
    def row(self, info, index: Long = 0) -> Row:
        """Return scalar values at index."""
        table = self.select(info)
        row = {}
        for name in table.column_names:
            scalar = table[name][index]
            row[name] = (
                Column.fromscalar(scalar) if isinstance(scalar, pa.ListScalar) else scalar.as_py()
            )
        return Row(**row)  # type: ignore

    @doc_field(
        offset="number of rows to skip; negative value skips from the end",
        length="maximum number of rows to return",
        reverse="reverse order after slicing; forces a copy",
    )
    def slice(
        self, info, offset: Long = 0, length: Optional[Long] = None, reverse: bool = False
    ) -> 'Table':
        """Return zero-copy slice of table."""
        table = self.select(info)
        table = table.slice(len(table) + offset if offset < 0 else offset, length)
        return Table(table[::-1] if reverse else table)

    @doc_field(
        by="column names",
        reverse="return groups in reversed stable order",
        length="maximum number of groups to return",
        counts="optionally include counts in an aliased column",
    )
    def group(
        self,
        info,
        by: List[str],
        reverse: bool = False,
        length: Optional[Long] = None,
        counts: str = '',
    ) -> 'Table':
        """Return table grouped by columns, with stable ordering.

        Other columns can be accessed by the `column` field as a `ListColumn`.
        Typically used in conjunction with `aggregate` or `tables`.
        """
        table = self.select(info)
        if selections(*info.selected_fields) == {'length'}:  # optimized for count
            return Table(T.encode(table, *by).unique()[:length])
        if set(table.column_names) <= set(by):
            table, counts_ = T.unique(table, *by, reverse=reverse, length=length, counts=counts)
        else:
            table, counts_ = T.group(table, *by, reverse=reverse, length=length)
        return Table(table.append_column(counts, counts_) if counts else table)

    @doc_field(
        by="column names",
        diffs="optional inequality predicates; scalars are compared to the adjacent difference",
        counts="optionally include counts in an aliased column",
    )
    @no_type_check
    def partition(self, info, by: List[str], diffs: List[Diff] = [], counts: str = '') -> 'Table':
        """Return table partitioned by discrete differences of the values.

        Differs from `group` by relying on adjacency, and is typically faster.
        Other columns can be accessed by the `column` field as a `ListColumn`.
        Typically used in conjunction with `aggregate` or `tables`.
        """
        table = self.select(info)
        funcs = {diff.pop('name'): diff for diff in map(dict, diffs)}
        names = list(itertools.takewhile(lambda name: name not in funcs, by))
        predicates = {}
        for name in by[len(names) :]:  # noqa: E203
            ((func, value),) = funcs.pop(name, {'not_equal': None}).items()
            predicates[name] = (getattr(pc, func),)
            if value is not None:
                if pa.types.is_timestamp(C.scalar_type(table[name])):
                    value = timedelta(seconds=value)
                predicates[name] += (value,)
        table, counts_ = T.partition(table, *names, **predicates)
        return Table(table.append_column(counts, counts_) if counts else table)

    @doc_field(
        by="column names",
        reverse="descending stable order",
        length="maximum number of rows to return; may be significantly faster but is unstable",
    )
    def sort(
        self, info, by: List[str], reverse: bool = False, length: Optional[Long] = None
    ) -> 'Table':
        """Return table slice sorted by specified columns.

        Sorting on list columns will sort within scalars, all of which must have the same lengths.
        """
        table = self.select(info)
        func = T.sort_list if all(C.is_list_type(table[name]) for name in by) else T.sort
        return Table(func(table, *by, reverse=reverse, length=length))

    @doc_field(by="column names")
    def min(self, info, by: List[str]) -> 'Table':
        """Return table with minimum values per column."""
        table = self.select(info)
        return Table(T.matched(table, C.min, *by))

    @doc_field(by="column names")
    def max(self, info, by: List[str]) -> 'Table':
        """Return table with maximum values per column."""
        table = self.select(info)
        return Table(T.matched(table, C.max, *by))

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
        query: Queries = {},
        on: Filters = {},
        invert: bool = False,
        reduce: Operator = Operator.AND,
    ) -> 'Table':
        """Return table with rows which match all (by default) queries.

        List columns apply their respective filters to the scalar values within lists.
        All referenced list columns must have the same lengths.
        """
        if not isinstance(self.table, pa.Table) and reduce.value in ('and', 'or'):
            query, table = {}, self.select(info, dict(query), invert=invert, reduce=reduce.value)
        else:
            table = self.select(info)
        filters = list(dict(query).items())
        for value in map(dict, itertools.chain(*dict(on).values())):
            filters.append((value.pop('name'), value))
        masks, list_masks = [], []
        lists = {name for name in table.column_names if C.is_list_type(table[name])}
        for name, value in filters:
            (list_masks if name in lists else masks).append(T.mask(table, name, **value))
        if list_masks:
            mask = functools.reduce(getattr(pc, reduce.value), list_masks)
            for name in lists:
                column = pa.chunked_array(C.map(ListChunk.filter_list, table[name], mask))
                table = table.set_column(table.column_names.index(name), name, column)
        if not masks:
            return Table(table)
        mask = functools.reduce(getattr(pc, reduce.value), masks)
        if selections(*info.selected_fields) == {'length'}:  # optimized for count
            return Table(range(C.count(mask, not invert)))
        return Table(table.filter(pc.invert(mask) if invert else mask))

    @Function.resolver
    @no_type_check
    def apply(self, info, **functions) -> 'Table':
        """Return view of table with functions applied across columns.

        If no alias is provided, the column is replaced and should be of the same type.
        If an alias is provided, a column is added and may be referenced in the `column` field,
        in filter `predicates`, and in the `by` arguments of grouping and sorting.
        """
        table = self.select(info)
        for value in map(dict, itertools.chain(*functions.values())):
            table = T.apply(table, value.pop('name'), **value)
        return Table(table)

    @doc_field
    def tables(self, info) -> List['Table']:  # type: ignore
        """Return a list of tables by splitting list columns, typically used after grouping.

        At least one list column must be referenced, and all list columns must have the same lengths.
        """
        table = self.select(info)
        lists = {name for name in table.column_names if C.is_list_type(table[name])}
        scalars = set(table.column_names) - lists
        for index, count in enumerate(T.list_value_length(table).to_pylist()):
            row = {name: pa.repeat(table[name][index], count) for name in scalars}
            row.update({name: table[name][index].values for name in lists})
            yield Table(pa.Table.from_pydict(row))

    @ListColumn.resolver
    def aggregate(self, info, **fields) -> 'Table':
        """Return table with aggregate functions applied to list columns, typically used after grouping.

        Columns which are aliased or change type can be accessed by the `column` field.
        """
        table = self.select(info)
        columns = {name: table[name] for name in table.column_names}
        for key in fields:
            func = getattr(ListChunk, key)
            for field in fields[key]:
                name = field.name
                columns[field.alias or name] = pa.chunked_array(C.map(func, table[name]))
        return Table(pa.Table.from_pydict(columns))


@strawberry.type(description="a table sorted by a composite index")
class IndexedTable(Table):
    index: List[str] = strawberry.field(default=tuple(indexed), description="the composite index")
    __init__ = AbstractTable.__init__

    @QueryInput.resolve_types({name: types[name] for name in indexed})
    def search(self, info, **queries) -> Table:
        """Return table with matching values for composite `index`.

        Queries must be a prefix of the `index`.
        Only one inequality query is allowed, and must be last.
        """
        for name in queries:
            if queries[name] is None:
                raise TypeError(f"`{name}` is optional, not nullable")
        queries = {name: dict(queries[name]) for name in queries}
        table = self.select(info, queries)
        if not isinstance(self.table, pa.Table):
            return Table(table)
        for name in self.index:
            if name not in queries:
                break
            query = queries.pop(name)
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


if COLUMNS or FILTERS:
    names = dataset.schema.names if ''.join(COLUMNS) in '*' else COLUMNS
    columns = {to_camel_case(name): ds.field(name) for name in names}
    table = dataset.to_table(columns=columns, filter=filter_expression(FILTERS))
    for name in indexed:
        assert not table[name].null_count, f"binary search requires non-null columns: {name}"
Query = IndexedTable if indexed else Table
app = GraphQL(Query(table), debug=DEBUG, federated=FEDERATED)
