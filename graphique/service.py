from typing import List, Optional, Union
import pyarrow as pa
import pyarrow.parquet as pq
import strawberry.asgi
from starlette.applications import Starlette
from .core import Table as T
from .settings import DEBUG, INDEX, MMAP, PARQUET_PATH

table = pq.read_table(PARQUET_PATH, memory_map=MMAP)
types = T.types(table)
index = list(INDEX) or T.index(table)
indexed = Optional[Union[tuple(types[name] for name in index)]]  # type: ignore


def select(info, tbl=table) -> pa.Table:
    """Return table with selected columns."""
    (node,) = info.field_nodes
    names = (selection.name.value for selection in node.selection_set.selections)
    return T.select(tbl, *names)


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


@strawberry.type(is_input=True)
class Range:
    lower: indexed = None  # type: ignore
    upper: indexed = None  # type: ignore
    include_lower: Optional[bool] = True
    include_upper: Optional[bool] = False

    def __init__(self, **kwargs):  # workaround for default inputs being overridden
        for name, value in kwargs.items():
            if value is not None:
                setattr(self, name, value)


@strawberry.type
class Query:
    @strawberry.field
    def count(self, info) -> int:
        """total row count"""
        return len(table)

    @strawberry.field
    def unique_count(self, info) -> Counts:
        """unique value count"""
        data = T.unique(select(info))
        return Counts(**{name: len(data[name]) for name in data})  # type: ignore

    @strawberry.field
    def null_count(self, info) -> Counts:
        """null value counts"""
        data = T.null_count(select(info))
        return Counts(**data)  # type: ignore

    @strawberry.field
    def slice(self, info, start: int = 0, stop: int = None) -> Columns:
        """Return table slice."""
        data = select(info)[start:stop]
        return Columns(**data.to_pydict())  # type: ignore

    @strawberry.field
    def unique(self, info) -> Columns:
        """unique values"""
        data = T.unique(select(info))
        return Columns(**{name: data[name].to_pylist() for name in data})  # type: ignore

    @strawberry.field
    def sum(self, info) -> Numbers:
        """sum of columns"""
        data = T.sum(select(info))
        return Numbers(**data)  # type: ignore

    @strawberry.field
    def search(self, info, ranges: List[Range] = []) -> Columns:
        f"""Return table within ranges for index: {index}.

        A multi-valued range can only appear last.
        """
        if len(ranges) > len(index):
            raise ValueError(f"too many ranges for index: {index}")
        data = table
        items = zip(index, ranges)
        for name, rng in items:
            data = T.search(data, name, **rng.__dict__)
            if None in (rng.lower, rng.upper) or rng.lower != rng.upper:  # type: ignore
                break
        for name, _ in items:
            raise ValueError(f"range for `{name}` appears after a multi-valued range")
        return Columns(**select(info, data).to_pydict())  # type: ignore


schema = strawberry.Schema(query=Query)
app = Starlette(debug=DEBUG)
app.add_route('/graphql', strawberry.asgi.GraphQL(schema, debug=DEBUG))
