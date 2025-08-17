"""
GraphQL output types and resolvers.
"""

from __future__ import annotations
import functools
import inspect
from collections.abc import Callable
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Annotated, Generic, TypeVar, TYPE_CHECKING, get_args
import ibis
import pyarrow as pa
import pyarrow.compute as pc
import strawberry
from strawberry import Info
from strawberry.types.field import StrawberryField
from .core import getitems
from .scalars import Long, py_type, scalar_map

if TYPE_CHECKING:  # pragma: no cover
    from .interface import Dataset
T = TypeVar('T')


class links:
    type = '[data type](https://ibis-project.org/reference/datatypes)'


def selections(*fields) -> set:
    """Return field name selections from strawberry `SelectedField`."""
    return {selection.name for field in fields for selection in field.selections}


def doc_field(func: Callable | None = None, **kwargs: str) -> StrawberryField:
    """Return strawberry field with argument and docstring descriptions."""
    if func is None:
        return functools.partial(doc_field, **kwargs)  # type: ignore
    for name in kwargs:
        alias = name.strip('_') if name.endswith('_') else None
        argument = strawberry.argument(name=alias, description=kwargs[name])
        func.__annotations__[name] = Annotated[func.__annotations__[name], argument]
    return strawberry.field(func, description=inspect.getdoc(func))


def compute_field(func: Callable):
    """Wrap compute function with its description."""
    doc = inspect.getdoc(getattr(pc, func.__name__))
    return strawberry.field(func, description=doc.splitlines()[0])  # type: ignore


def col_field(func: Callable):
    """Wrap `Column` method with its description."""
    doc = inspect.getdoc(getattr(ibis.Column, func.__name__))
    return strawberry.field(func, description=doc.splitlines()[0])  # type: ignore


@strawberry.interface(description="an arrow array")
class Column:
    registry = {}  # type: ignore

    def __init__(self, column: ibis.Column):
        self.column = column

    @property
    def array(self):
        return self.column.to_pyarrow()

    def __init_subclass__(cls):
        cls.__init__ = cls.__init__

    @classmethod
    def register(cls, *scalars):
        if cls is Column:
            return lambda cls: cls.register(*scalars) or cls
        # strawberry#1921: scalar python names are prepended to column name
        generic = issubclass(cls, Generic)
        for scalar in scalars:
            cls.registry[scalar] = cls[scalar_map.get(scalar, scalar)] if generic else cls

    @strawberry.field(description=links.type)
    def type(self) -> str:
        return str(self.column.type())

    @classmethod
    def cast(cls, column: ibis.Column) -> Column:
        """Return typed column based on array type."""
        return cls.registry[py_type(column.type().to_pyarrow())](column)

    @col_field
    def count(self) -> Long:
        return self.column.count().to_pyarrow().as_py()

    @classmethod
    def resolve_type(cls, obj, info, *_) -> str:
        config = Info(info, None).schema.config
        args = get_args(getattr(obj, '__orig_class__', None))
        return config.name_converter.from_generic(obj.__strawberry_definition__, args)


@strawberry.type(description="unique values and counts")
class Set(Generic[T]):
    counts: list[Long] = strawberry.field(description="list of counts")

    def __init__(self, array, counts=pa.array([])):
        self.array, self.counts = array, counts.to_pylist()

    @doc_field
    def values(self) -> list[T | None]:
        """list of values"""
        return self.array.to_pylist()


@Column.register(timedelta, pa.MonthDayNano)
@strawberry.type(name='Column', description="column of elapsed times")
class NominalColumn(Generic[T], Column):
    values = doc_field(Set.values)

    @compute_field
    def count_distinct(self, mode: str = 'only_valid') -> Long:
        return pc.count_distinct(self.array, mode=mode).as_py()

    @strawberry.field(description=Set.__strawberry_definition__.description)  # type: ignore
    def unique(self, info: Info) -> Set[T]:
        if 'counts' in selections(*info.selected_fields):
            return Set(*self.array.value_counts().flatten())
        return Set(self.array.unique())

    @doc_field
    def value(self, index: Long = 0) -> T | None:
        """scalar value at index"""
        return self.array[index].as_py()

    @col_field
    def fill_null(self, value: T) -> list[T]:
        return self.column.fill_null(value).to_list()


@Column.register(date, datetime, time, bytes)
@strawberry.type(name='Column', description="column of ordinal values")
class OrdinalColumn(NominalColumn[T]):
    @compute_field
    def first(self, skip_nulls: bool = True, min_count: int = 0) -> T | None:
        return pc.first(self.array, skip_nulls=skip_nulls, min_count=min_count).as_py()

    @compute_field
    def last(self, skip_nulls: bool = True, min_count: int = 0) -> T | None:
        return pc.last(self.array, skip_nulls=skip_nulls, min_count=min_count).as_py()

    @compute_field
    def min(self, skip_nulls: bool = True, min_count: int = 0) -> T | None:
        return pc.min(self.array, skip_nulls=skip_nulls, min_count=min_count).as_py()

    @compute_field
    def max(self, skip_nulls: bool = True, min_count: int = 0) -> T | None:
        return pc.max(self.array, skip_nulls=skip_nulls, min_count=min_count).as_py()


@Column.register(str)
@strawberry.type(name='ingColumn', description="column of strings")
class StringColumn(OrdinalColumn[T]): ...


@strawberry.type
class IntervalColumn(OrdinalColumn[T]):
    @compute_field
    def mode(self, n: int = 1, skip_nulls: bool = True, min_count: int = 0) -> Set[T]:
        return Set(*pc.mode(self.array, n, skip_nulls=skip_nulls, min_count=min_count).flatten())

    @compute_field
    def sum(self, skip_nulls: bool = True, min_count: int = 0) -> T | None:
        return pc.sum(self.array, skip_nulls=skip_nulls, min_count=min_count).as_py()

    @compute_field
    def mean(self, skip_nulls: bool = True, min_count: int = 0) -> float | None:
        return pc.mean(self.array, skip_nulls=skip_nulls, min_count=min_count).as_py()


@Column.register(float, Decimal)
@strawberry.type(name='Column', description="column of floats or decimals")
class RatioColumn(IntervalColumn[T]):
    @compute_field
    def stddev(self, ddof: int = 0, skip_nulls: bool = True, min_count: int = 0) -> float | None:
        return pc.stddev(self.array, ddof=ddof, skip_nulls=skip_nulls, min_count=min_count).as_py()

    @compute_field
    def variance(self, ddof: int = 0, skip_nulls: bool = True, min_count: int = 0) -> float | None:
        options = {'skip_nulls': skip_nulls, 'min_count': min_count}
        return pc.variance(self.array, ddof=ddof, **options).as_py()

    @compute_field
    def quantile(
        self,
        q: list[float] = [0.5],
        interpolation: str = 'linear',
        skip_nulls: bool = True,
        min_count: int = 0,
    ) -> list[float | None]:
        options = {'skip_nulls': skip_nulls, 'min_count': min_count}
        return pc.quantile(self.array, q=q, interpolation=interpolation, **options).to_pylist()


@Column.register(bool)
@strawberry.type(name='eanColumn', description="column of booleans")
class BooleanColumn(IntervalColumn[T]):
    @compute_field
    def any(self, skip_nulls: bool = True, min_count: int = 1) -> bool | None:
        return pc.any(self.array, skip_nulls=skip_nulls, min_count=min_count).as_py()

    @compute_field
    def all(self, skip_nulls: bool = True, min_count: int = 1) -> bool | None:
        return pc.all(self.array, skip_nulls=skip_nulls, min_count=min_count).as_py()


@Column.register(int, Long)
@strawberry.type(name='Column', description="column of integers")
class IntColumn(RatioColumn[T]):
    @doc_field
    def take_from(
        self, info: Info, field: str
    ) -> Annotated['Dataset', strawberry.lazy('.interface')] | None:
        """Select indices from a table on the root Query type."""
        root = getattr(info.root_value, field)
        return root.take(info, self.array.combine_chunks())


@Column.register(list)
@strawberry.type(description="column of lists")
class ListColumn(Column):
    @doc_field
    def length(self, index: Long = 0) -> IntColumn[int]:
        """the lengths of the arrays"""
        return self.cast(self.column.length())  # type: ignore

    @doc_field
    def values(self, index: Long = 0) -> Column:
        """values at index"""
        return self.cast(self.column[index])

    @doc_field
    def unnest(self) -> Column:
        """concatenation of all sub-lists"""
        return self.cast(self.column.unnest())


@Column.register(dict)
@strawberry.type(description="column of structs")
class StructColumn(Column):
    @doc_field
    def value(self, index: Long = 0) -> dict | None:
        """scalar json object at index"""
        return self.array[index].as_py()

    @doc_field
    def names(self) -> list[str]:
        """field names"""
        return [field.name for field in self.array.type]

    @doc_field(name="field name(s); multiple names access nested fields")
    def column(self, name: list[str]) -> Column | None:
        """Return struct field as a column."""
        return self.cast(getitems(self.column, *name))
