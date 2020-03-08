from typing import List, Optional
import pyarrow as pa
import pyarrow.parquet as pq
import strawberry.asgi
from starlette.applications import Starlette
from .core import Table as T
from .settings import DEBUG, INDEX, MMAP, PARQUET_PATH

table = pq.read_table(PARQUET_PATH, memory_map=MMAP)
types = T.types(table)
index = list(INDEX) or T.index(table)


def selections(node):
    """Return tree of field name selections."""
    nodes = getattr(node.selection_set, 'selections', [])
    return {node.name.value: selections(node) for node in nodes}


def select(info, tbl=table) -> pa.Table:
    """Return table with selected columns."""
    return T.select(tbl, *selections(*info.field_nodes))


@strawberry.type
class Columns:
    __annotations__ = {
        name: List[Optional(cls) if table[name].null_count else cls]  # type: ignore
        for name, cls in types.items()
    }
    locals().update(dict.fromkeys(types, ()))


@strawberry.type
class Numbers:
    __annotations__ = {name: cls for name, cls in types.items() if cls in (int, float)}
    locals().update({name: types[name]() for name in __annotations__})


@strawberry.type
class Counts:
    __annotations__ = dict.fromkeys(types, int)
    locals().update(dict.fromkeys(types, 0))


def __init__(self, **kwargs):  # workaround for default inputs being overridden
    for name, value in kwargs.items():
        if value is not None:
            setattr(self, name, value)


@strawberry.type(is_input=True)
class Equals:
    __annotations__ = {name: Optional[types[name]] for name in index}
    locals().update(dict.fromkeys(index))
    __init__ = __init__


@strawberry.type(is_input=True)
class IsIn:
    __annotations__ = {name: Optional[List[types[name]]] for name in index}  # type: ignore
    locals().update(dict.fromkeys(index))
    __init__ = __init__


ranges = {}
namespace = {
    'lower': None,
    'upper': None,
    'include_lower': True,
    'include_upper': False,
    '__init__': __init__,
}
for cls in {types[name] for name in index}:
    name = cls.__name__.capitalize() + 'Range'
    namespace['__annotations__'] = {  # type: ignore
        'lower': Optional[cls],
        'upper': Optional[cls],
        'include_lower': Optional[bool],
        'include_upper': Optional[bool],
    }
    ranges[cls] = strawberry.type(type(name, (), namespace), is_input=True)


@strawberry.type(is_input=True)
class Range:
    __annotations__ = {name: Optional[ranges[types[name]]] for name in index}
    locals().update(dict.fromkeys(index))
    __init__ = __init__


@strawberry.type
class IntColumns:
    __annotations__ = {name: List[int] for name in types}  # type: ignore
    locals().update(dict.fromkeys(types, ()))


@strawberry.type
class Uniques:
    values: Columns
    counts: IntColumns


@strawberry.type
class Query:
    @strawberry.field
    def count(self, info) -> int:
        """total row count"""
        return len(table)

    @strawberry.field
    def null_count(self, info) -> Counts:
        """null value counts"""
        data = T.null_count(select(info))
        return Counts(**data)  # type: ignore

    @strawberry.field
    def slice(self, info, offset: int = 0, length: int = None) -> Columns:
        """Return table slice."""
        data = select(info).slice(offset, length)
        return Columns(**data.to_pydict())  # type: ignore

    @strawberry.field
    def unique(self, info) -> Uniques:
        """unique values and optionally counts"""
        names = selections(*info.field_nodes)
        counts = dict.fromkeys(names.get('counts', []))
        values = dict.fromkeys(set(names.get('values', [])) - set(counts))
        data = T.unique(T.select(table, *values))
        for name, value in data.items():
            values[name] = value.to_pylist()
        data = T.unique(T.select(table, *counts), counts=True)
        for name, (value, count) in data.items():
            values[name] = value.to_pylist()
            counts[name] = count.to_pylist()
        return Uniques(Columns(**values), IntColumns(**counts))  # type: ignore

    @strawberry.field
    def sum(self, info) -> Numbers:
        """sum of columns"""
        data = T.sum(select(info))
        return Numbers(**data)  # type: ignore

    @strawberry.field
    def search(
        self, info, equals: Equals = Equals(), isin: IsIn = IsIn(), range: Range = Range(),
    ) -> Columns:  # type: ignore
        f"""Return table with matching values for index: {index}.

        The values are matched in index order.
        Only one `range` or `isin` query is allowed, and applied last.
        """
        names = list(isin.__dict__) + list(range.__dict__)
        if len(names) > 1:
            raise ValueError(f"only one multi-valued selection allowed: {names}")
        names = list(equals.__dict__) + names
        if names != index[: len(names)]:
            raise ValueError(f"{names} is not a prefix of index: {index}")
        data = table
        for name in equals.__dict__:
            data = T.isin(data, name, getattr(equals, name))
        for name in isin.__dict__:
            data = T.isin(data, name, *getattr(isin, name))
        for name in range.__dict__:
            data = T.range(data, name, **getattr(range, name).__dict__)
        return Columns(**select(info, data).to_pydict())  # type: ignore


schema = strawberry.Schema(query=Query)
app = Starlette(debug=DEBUG)
app.add_route('/graphql', strawberry.asgi.GraphQL(schema, debug=DEBUG))
