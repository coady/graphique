"""
GraphQL output types and resolvers.
"""

from __future__ import annotations

import collections
import functools
import inspect
from collections.abc import Callable
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Annotated, Generic, TypeVar, get_args

import ibis
import pyarrow as pa
import strawberry
from strawberry import UNSET, Info
from strawberry.types.field import StrawberryField

from .core import links
from .inputs import optional, provisional
from .scalars import BigInt, py_type, scalar_map

if TYPE_CHECKING:  # pragma: no cover
    from .interface import Dataset
T = TypeVar('T')


def selections(*fields) -> collections.Counter:
    """Return field name selections from strawberry `SelectedField`."""
    counts = collections.Counter()  # type: ignore
    for field in fields:
        for selection in field.selections:
            if hasattr(selection, 'name'):
                counts[selection.name] += 1
            else:
                counts.update(selections(selection))
    return counts


def doc_field(func: Callable | None = None, **kwargs: str) -> StrawberryField:
    """Return strawberry field with argument and docstring descriptions."""
    if func is None:
        return functools.partial(doc_field, **kwargs)  # type: ignore
    parameters = inspect.signature(func).parameters
    for name in kwargs:
        alias = name.strip('_') if name.endswith('_') else None
        directives = [optional()] if parameters[name].default is UNSET else []
        argument = strawberry.argument(name=alias, description=kwargs[name], directives=directives)
        func.__annotations__[name] = Annotated[func.__annotations__[name], argument]
    return strawberry.field(func, description=inspect.getdoc(func))


def col_field(func: Callable):
    """Wrap `Column` method with its description."""
    doc = inspect.getdoc(getattr(ibis.expr.types.BooleanColumn, func.__name__))
    return strawberry.field(func, description=doc.splitlines()[0])  # type: ignore


@strawberry.interface(description="ibis column interface")
class Column:
    registry = {}  # type: ignore

    def __init__(self, column: ibis.Column):
        self.column = column

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

    @strawberry.field(description=links.types)
    def type(self) -> str:
        return str(self.column.type())

    @classmethod
    def cast(cls, column: ibis.Column) -> Column:
        """Return typed column based on array type."""
        return cls.registry[py_type(column.type())](column)

    @col_field
    def count(self) -> BigInt:
        return self.column.count().to_pyarrow().as_py()

    @col_field
    def nunique(self, approx: bool = False) -> BigInt:
        method = self.column.approx_nunique if approx else self.column.nunique
        return method().to_pyarrow().as_py()

    @classmethod
    def resolve_type(cls, obj, info, *_) -> str:
        config = Info(info, None).schema.config
        args = get_args(getattr(obj, '__orig_class__', None))
        return config.name_converter.from_generic(obj.__strawberry_definition__, args)


@strawberry.type(description="distinct values and counts")
class Set(Generic[T]):
    def __init__(self, column, cache=False):
        table = column.value_counts()
        self.table = table.cache() if cache else table

    @doc_field
    def values(self) -> list[T | None]:
        """distinct values"""
        return self.table[0].to_list()

    @doc_field
    def counts(self) -> list[BigInt]:
        """corresponding counts"""
        return self.table[1].to_list()


@Column.register(bytes)
@strawberry.type(name='Column', description=f"[generic column]({links.ref}/expression-generic)")
class GenericColumn(Generic[T], Column):
    @doc_field
    def values(self) -> list[T | None]:
        """list of values"""
        return self.column.to_list()

    @doc_field
    def distinct(self, info: Info) -> Set[T]:
        """distinct values and counts"""
        count = selections(*info.selected_fields).total()
        return Set(self.column, cache=count > 1)

    @col_field
    def first(self) -> T | None:
        return self.column.first().to_pyarrow().as_py()

    @col_field
    def last(self) -> T | None:
        return self.column.last().to_pyarrow().as_py()

    @doc_field
    def drop_null(self) -> list[T]:
        """non-null values"""
        return self.column.as_table().drop_null()[0].to_list()

    @col_field
    def fill_null(self, value: T) -> list[T]:
        return self.column.fill_null(value).to_list()

    @col_field
    def mode(self) -> T | None:
        return self.column.mode().to_pyarrow().as_py()

    @col_field
    def min(self) -> T | None:
        return self.column.min().to_pyarrow().as_py()

    @col_field
    def max(self) -> T | None:
        return self.column.max().to_pyarrow().as_py()

    @col_field
    def quantile(self, q: list[float] = [0.5]) -> list[T] | None:
        return self.column.quantile(q).to_pyarrow().as_py()


@Column.register(date, datetime, time)
@strawberry.type(name='Column', description=f"[temporal column]({links.ref}/expression-temporal)")
class TemporalColumn(GenericColumn[T]): ...  # pragma: no branch


@Column.register(timedelta, pa.MonthDayNano)
@strawberry.type(
    name='Column',
    directives=[provisional()],
    description=f"""provisional [interval column]({links.ref}/expression-temporal#ibis.expr.types.temporal.IntervalValue)

Interval support varies by backend; durations may still be useful for computation and as scalar inputs.""",
)
class DurationColumn(GenericColumn[T]): ...  # pragma: no branch


@Column.register(str)
@strawberry.type(name='ingColumn', description=f"[string column]({links.ref}/expression-strings)")
class StringColumn(GenericColumn[T]): ...  # pragma: no branch


@Column.register(float, Decimal)
@strawberry.type(name='Column', description=f"[numeric column]({links.ref}/expression-numeric)")
class NumericColumn(GenericColumn[T]):
    @col_field
    def sum(self) -> T | None:
        return self.column.sum().to_pyarrow().as_py()

    @col_field
    def mean(self) -> float | None:
        return self.column.mean().to_pyarrow().as_py()

    @col_field
    def std(self, how: str = 'sample') -> float | None:
        return self.column.std(how=how).to_pyarrow().as_py()

    @col_field
    def var(self, how: str = 'sample') -> float | None:
        return self.column.var(how=how).to_pyarrow().as_py()

    @col_field
    def quantile(self, q: list[float] = [0.5], approx: bool = False) -> list[float] | None:
        method = self.column.approx_quantile if approx else self.column.quantile
        return method(q).to_pyarrow().as_py()


@Column.register(bool)
@strawberry.type(
    name='eanColumn',
    description=f"[boolean column]({links.ref}/expression-numeric#ibis.expr.types.logical.BooleanColumn)",
)
class BooleanColumn(NumericColumn[T]):
    @col_field
    def any(self) -> bool | None:
        return self.column.any().to_pyarrow().as_py()

    @col_field
    def all(self) -> bool | None:
        return self.column.all().to_pyarrow().as_py()


@Column.register(int, BigInt)
@strawberry.type(
    name='Column',
    description=f"[integer column]({links.ref}/expression-numeric#ibis.expr.types.numeric.IntegerColumn)",
)
class IntColumn(NumericColumn[T]):
    @doc_field
    def take_from(
        self, info: Info, field: str
    ) -> Annotated['Dataset', strawberry.lazy('.interface')] | None:
        """Select indices from a table on the root Query type."""
        root = getattr(info.root_value, field)
        return root.take(info, self.column.to_list())


@Column.register(list)
@strawberry.type(description=f"[array column]({links.ref}/expression-collections)")
class ArrayColumn(Column): ...  # pragma: no branch


@Column.register(dict)
@strawberry.type(
    name='Column',
    description=f"[struct column]({links.ref}/expression-collections#ibis.expr.types.structs.StructValue)",
)
class StructColumn(GenericColumn[T]):
    @doc_field
    def names(self) -> list[str]:
        """field names"""
        return self.column.names
