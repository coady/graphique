"""
Service related utilities which don't require knowledge of the schema.
"""
import functools
import inspect
import itertools
import operator
import types
from datetime import datetime, timedelta
from typing import Iterable, Iterator, List, Mapping, Optional, Union, no_type_check
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import strawberry.asgi
from strawberry.utils.str_converters import to_camel_case
from .core import Column as C, ListChunk, Table as T
from .inputs import Diff, Function, Projections
from .models import Column, ListColumn, annotate, doc_field, selections
from .scalars import Long, scalar_map

comparisons = {
    'equal': operator.eq,
    'not_equal': operator.ne,
    'less': operator.lt,
    'less_equal': operator.le,
    'greater': operator.gt,
    'greater_equal': operator.ge,
    'is_in': ds.Expression.isin,
}
nulls = {
    'equal': ds.Expression.is_null,
    'not_equal': ds.Expression.is_valid,
}


def references(field) -> Iterator:
    """Generate every possible column reference from strawberry `SelectedField`."""
    if isinstance(field, str):
        yield field
    elif isinstance(field, Iterable):
        for value in field:
            yield from references(value)
        if isinstance(field, Mapping):
            for value in field.values():
                yield from references(value)
    else:
        for name in ('name', 'arguments', 'selections'):
            yield from references(getattr(field, name, []))


def filter_expression(queries: dict, invert=False, reduce: str = 'and') -> Optional[ds.Expression]:
    """Translate query format `field={predicate: value}` into dataset filter expression."""
    exprs: list = []
    for name, query in queries.items():
        field = ds.field(name)
        group = [
            nulls[predicate](field) if value is None else comparisons[predicate](field, value)
            for predicate, value in query.items()
        ]
        if group:
            exprs.append(functools.reduce(operator.and_, group))
    if not exprs:
        return None
    expr = functools.reduce(getattr(operator, f'{reduce}_'), exprs)
    return ~expr if invert else expr


class TimingExtension(strawberry.extensions.Extension):
    def on_request_start(self):
        self.start = datetime.now()

    def on_request_end(self):
        end = datetime.now()
        print(f"[{end.replace(microsecond=0)}]: {end - self.start}")


class GraphQL(strawberry.asgi.GraphQL):
    def __init__(self, root_value, debug=False, federated='', **kwargs):
        Schema = strawberry.Schema
        if federated:
            Schema = strawberry.federation.Schema
            Query = type('Query', (), {'__annotations__': {federated: type(root_value)}})
            root_value = strawberry.type(Query)(root_value)
        schema = Schema(
            type(root_value),
            types=Column.__subclasses__(),
            extensions=[TimingExtension] * bool(debug),
            scalar_overrides=scalar_map,
        )
        super().__init__(schema, debug=debug, **kwargs)
        self.root_value = root_value

    async def get_root_value(self, request):
        return self.root_value


@strawberry.interface(description="a schema-free table")
class AbstractTable:
    def __init__(self, table: Union[pa.Table, ds.dataset]):
        self.table = table

    def __init_subclass__(cls):
        """Downcast fields which return an `AbstractTable` to its implemented type."""
        cls.__init__ = AbstractTable.__init__
        resolvers = {'apply': Function.resolver, 'aggregate': ListColumn.resolver}
        for name, func in inspect.getmembers(cls, inspect.isfunction):
            if func.__annotations__.get('return') in ('AbstractTable', List['AbstractTable']):
                clone = types.FunctionType(func.__code__, func.__globals__)
                clone.__annotations__['return'] = cls
                if name in resolvers:
                    field = resolvers[name](clone)
                else:
                    field = annotate(func, List[cls] if name == 'tables' else cls)
                setattr(cls, name, field)

    def select(self, info, queries: dict = {}, invert=False, reduce: str = 'and') -> pa.Table:
        """Return table with only the rows and columns necessary to proceed."""
        case_map = {to_camel_case(name): name for name in self.table.schema.names}
        names = set(itertools.chain(*map(references, info.selected_fields))) & set(case_map)
        if isinstance(self.table, pa.Table):
            return self.table.select(names)
        columns = {name: ds.field(case_map[name]) for name in names}
        queries = {case_map[name]: queries[name] for name in queries}
        expr = filter_expression(queries, invert=invert, reduce=reduce)
        return self.table.to_table(columns=columns, filter=expr)

    @doc_field
    def length(self) -> Long:
        """number of rows"""
        if hasattr(self.table, '__len__'):
            return len(self.table)
        return len(self.table.to_table(columns=[]))

    @doc_field(
        cast="cast array to [arrow type](https://arrow.apache.org/docs/python/api/datatypes.html)",
        apply="projected functions",
    )
    def column(self, info, name: str, cast: str = '', apply: Projections = {}) -> Column:  # type: ignore
        """Return column of any type by name, with optional projection.

        This is typically only needed for aliased columns added by `apply` or `aggregate`.
        If the column is in the schema, `columns` can be used instead.
        """
        table = self.select(info)
        column = table[name]
        for func, name in dict(apply).items():
            others = (table[name] for name in (name if isinstance(name, list) else [name]))
            column = getattr(pc, func)(column, *others)
        return Column.cast(column.cast(cast) if cast else column)

    @doc_field(
        offset="number of rows to skip; negative value skips from the end",
        length="maximum number of rows to return",
        reverse="reverse order after slicing; forces a copy",
    )
    def slice(
        self, info, offset: Long = 0, length: Optional[Long] = None, reverse: bool = False
    ) -> 'AbstractTable':
        """Return zero-copy slice of table."""
        table = self.select(info)
        table = table.slice(len(table) + offset if offset < 0 else offset, length)
        return type(self)(table[::-1] if reverse else table)

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
    ) -> 'AbstractTable':
        """Return table grouped by columns, with stable ordering.

        Other columns can be accessed by the `column` field as a `ListColumn`.
        Typically used in conjunction with `aggregate` or `tables`.
        """
        table = self.select(info)
        if selections(*info.selected_fields) == {'length'}:  # optimized for count
            return type(self)(T.encode(table, *by).unique()[:length])
        if set(table.column_names) <= set(by):
            table, counts_ = T.unique(table, *by, reverse=reverse, length=length, counts=counts)
        else:
            table, counts_ = T.group(table, *by, reverse=reverse, length=length)
        return type(self)(table.append_column(counts, counts_) if counts else table)

    @doc_field(
        by="column names",
        diffs="optional inequality predicates; scalars are compared to the adjacent difference",
        counts="optionally include counts in an aliased column",
    )
    @no_type_check
    def partition(
        self, info, by: List[str], diffs: List[Diff] = [], counts: str = ''
    ) -> 'AbstractTable':
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
        return type(self)(table.append_column(counts, counts_) if counts else table)

    @doc_field(
        by="column names",
        reverse="descending stable order",
        length="maximum number of rows to return; may be significantly faster but is unstable",
    )
    def sort(
        self, info, by: List[str], reverse: bool = False, length: Optional[Long] = None
    ) -> 'AbstractTable':
        """Return table slice sorted by specified columns.

        Sorting on list columns will sort within scalars, all of which must have the same lengths.
        """
        table = self.select(info)
        lists = dict.fromkeys(name for name in by if C.is_list_type(table[name]))
        scalars = [name for name in by if name not in lists]
        if scalars:
            table = T.sort(table, *scalars, reverse=reverse, length=length)
        if lists:
            table = T.sort_list(table, *lists, reverse=reverse, length=length)
        return type(self)(table)

    @doc_field(by="column names")
    def min(self, info, by: List[str]) -> 'AbstractTable':
        """Return table with minimum values per column."""
        table = self.select(info)
        return type(self)(T.matched(table, C.min, *by))

    @doc_field(by="column names")
    def max(self, info, by: List[str]) -> 'AbstractTable':
        """Return table with maximum values per column."""
        table = self.select(info)
        return type(self)(T.matched(table, C.max, *by))

    @Function.resolver
    @no_type_check
    def apply(self, info, **functions) -> 'AbstractTable':
        """Return view of table with functions applied across columns.

        If no alias is provided, the column is replaced and should be of the same type.
        If an alias is provided, a column is added and may be referenced in the `column` field,
        in filter `predicates`, and in the `by` arguments of grouping and sorting.
        """
        table = self.select(info)
        for value in map(dict, itertools.chain(*functions.values())):
            table = T.apply(table, value.pop('name'), **value)
        return type(self)(table)

    @doc_field
    def tables(self, info) -> List['AbstractTable']:  # type: ignore
        """Return a list of tables by splitting list columns, typically used after grouping.

        At least one list column must be referenced, and all list columns must have the same lengths.
        """
        table = self.select(info)
        lists = {name for name in table.column_names if C.is_list_type(table[name])}
        scalars = set(table.column_names) - lists
        for index, count in enumerate(T.list_value_length(table).to_pylist()):
            row = {name: pa.repeat(table[name][index], count) for name in scalars}
            row.update({name: table[name][index].values for name in lists})
            yield type(self)(pa.Table.from_pydict(row))

    @ListColumn.resolver
    def aggregate(self, info, **fields) -> 'AbstractTable':
        """Return table with aggregate functions applied to list columns, typically used after grouping.

        Columns which are aliased or change type can be accessed by the `column` field.
        """
        table = self.select(info)
        columns = {name: table[name] for name in table.column_names}
        for key in fields:
            func = getattr(ListChunk, key)
            for field in fields[key]:
                columns[field.alias or field.name] = C.map(table[field.name], func)
        return type(self)(pa.Table.from_pydict(columns))
