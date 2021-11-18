"""
Service related utilities which don't require knowledge of the schema.
"""
import functools
import itertools
import operator
from datetime import datetime
from typing import Iterable, Iterator, Mapping, Optional, Union
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import strawberry.asgi
from strawberry.utils.str_converters import to_camel_case
from .inputs import Projections
from .models import Column, doc_field
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


@strawberry.interface
class AbstractTable:
    def __init__(self, table: Union[pa.Table, ds.dataset]):
        self.table = table

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
