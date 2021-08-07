"""
Service related utilities which don't require knowledge of the schema.
"""
import itertools
from datetime import datetime
from typing import Union
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import strawberry.asgi
from strawberry.utils.str_converters import to_camel_case
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
    def __init__(self, table: Union[pa.Table, pq.ParquetDataset]):
        self.table = table

    @property
    def case_map(self):
        return {to_camel_case(name): name for name in self.table.schema.names}

    def select(self, info) -> pa.Table:
        """Return table with only the columns necessary to proceed."""
        case_map = self.case_map
        names = set(itertools.chain(*map(references, info.field_nodes))) & set(case_map)
        if isinstance(self.table, pa.Table):
            return self.table.select(names)
        return self.table.read(list(map(case_map.get, names))).rename_columns(names)

    @doc_field
    def length(self) -> Long:
        """number of rows"""
        return len(self.table if hasattr(self.table, '__len__') else self.table.read([]))

    @doc_field(
        cast="cast array to [arrow type](https://arrow.apache.org/docs/python/api/datatypes.html)",
        apply="projected functions",
    )
    def column(self, info, name: str, cast: str = '', apply: Projections = {}) -> Column:  # type: ignore
        """Return column of any type by name, with optional projection.
        This is typically only needed for aliased columns added by `apply` or `aggregate`.
        If the column is in the schema, `columns` can be used instead."""
        table = self.select(info)
        column = table[name]
        for func, name in dict(apply).items():
            others = (table[name] for name in (name if isinstance(name, list) else [name]))
            column = getattr(pc, func)(column, *others)
        return Column.cast(column.cast(cast) if cast else column)
