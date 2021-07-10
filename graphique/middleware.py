"""
Service related utilities which don't require knowledge of the schema.
"""
import itertools
from datetime import datetime
import pyarrow as pa
import strawberry.asgi
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


class TimingExtension(strawberry.extensions.Extension):  # pragma: no cover
    def on_request_start(self):
        self.start = datetime.now()

    def on_request_end(self):
        end = datetime.now()
        print(f"[{end.replace(microsecond=0)}]: {end - self.start}")


class GraphQL(strawberry.asgi.GraphQL):
    def __init__(self, root_value, debug=False, **kwargs):
        options = dict(types=Column.__subclasses__(), extensions=[TimingExtension] * bool(debug))
        schema = strawberry.Schema(type(root_value), **options)
        super().__init__(schema, debug=debug, **kwargs)
        self.root_value = root_value

    async def get_root_value(self, request):
        return self.root_value


@strawberry.interface
class AbstractTable:
    def __init__(self, table: pa.Table):
        self.table = table

    def select(self, info) -> pa.Table:
        """Return table with only the columns necessary to proceed."""
        names = set(references(*info.field_nodes))
        return self.table.select(names & set(self.table.column_names))

    @doc_field
    def length(self) -> Long:
        """number of rows"""
        return len(self.table)

    @doc_field(
        cast="cast array to [arrow type](https://arrow.apache.org/docs/python/api/datatypes.html)",
        apply="projected functions",
    )
    def column(self, name: str, cast: str = '', apply: Projections = {}) -> Column:  # type: ignore
        """Return column of any type by name, with optional projection.
        This is typically only needed for aliased columns added by `apply` or `aggregate`.
        If the column is in the schema, `columns` can be used instead."""
        column = self.table[name]
        for func, name in dict(apply).items():
            column = T.projected[func](column, self.table[name])
        return Column.cast(column.cast(cast) if cast else column)
