"""
Service related utilities which don't require knowledge of the schema.
"""
import itertools
from datetime import datetime
from typing import Optional, no_type_check
import pyarrow as pa
import strawberry.asgi
from starlette.middleware import base
from strawberry.utils.str_converters import to_camel_case
from .core import Table as T
from .inputs import Projections
from .models import Column, doc_field
from .scalars import Long


def references(node):
    """Generate every possible column reference."""
    if hasattr(node, 'name'):
        yield node.name.value
    value = getattr(node, 'value', None)
    yield getattr(value, 'value', value)
    nodes = itertools.chain(
        getattr(node, 'arguments', []),
        getattr(node, 'fields', []),
        getattr(value, 'fields', []),
        getattr(value, 'values', []),
        getattr(getattr(node, 'selection_set', None), 'selections', []),
    )
    for node in nodes:
        yield from references(node)


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
        schema = strawberry.Schema(type(root_value), types=Column.__subclasses__())
        super().__init__(schema, **kwargs)
        self.root_value = root_value

    async def get_root_value(self, request):
        return self.root_value


@strawberry.interface
class AbstractTable:
    def __init__(self, table: pa.Table):
        self.table = table
        self.case_map = {to_camel_case(name): name for name in getattr(table, 'column_names', [])}

    def to_snake_case(self, name):
        return self.case_map.get(name, name)

    def select(self, info) -> pa.Table:
        """Return table with only the columns necessary to proceed."""
        names = set(map(self.to_snake_case, references(*info.field_nodes)))
        return self.table.select(names & set(self.table.column_names))

    @doc_field
    def length(self) -> Long:
        """number of rows"""
        return len(self.table)  # type: ignore

    @doc_field
    @no_type_check
    def column(self, name: str, apply: Optional[Projections] = ()) -> Column:
        """Return column of any type by name, with optional projection.
        This is typically only needed for aliased columns added by `apply` or `aggregate`.
        If the column is in the schema, `columns` can be used instead."""
        column = self.table[self.to_snake_case(name)]
        for func, name in dict(apply).items():
            column = T.projected[func](column, self.table[self.to_snake_case(name)])
        return Column.cast(column)
