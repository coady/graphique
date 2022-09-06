"""
GraphQL output types and resolvers.
"""
import functools
import inspect
import operator
import types
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Callable, Generic, List, Optional, TypeVar
import pyarrow as pa
import pyarrow.compute as pc
import strawberry
from strawberry.field import StrawberryField
from strawberry.types import Info
from typing_extensions import Annotated
from .core import Column as C
from .inputs import links
from .scalars import Long, type_map

T = TypeVar('T')


def selections(*fields) -> set:
    """Return set of field name selections from strawberry `SelectedField`."""
    return {selection.name for field in fields for selection in field.selections}


def doc_field(func: Optional[Callable] = None, **kwargs: str) -> StrawberryField:
    """Return strawberry field with argument and docstring descriptions."""
    if func is None:
        return functools.partial(doc_field, **kwargs)
    for name in kwargs:
        argument = strawberry.argument(description=kwargs[name])
        func.__annotations__[name] = Annotated[func.__annotations__[name], argument]
    return strawberry.field(func, description=inspect.getdoc(func))


@strawberry.interface(description="column interface")
class Column:
    def __init__(self, array):
        self.array = array

    def __init_subclass__(cls):
        cls.__init__ = Column.__init__

    @doc_field
    def type(self) -> str:
        f"""{links.type}"""
        return str(self.array.type)

    @doc_field
    def length(self) -> Long:
        """number of rows"""
        return len(self.array)

    @classmethod
    def cast(cls, array: pa.ChunkedArray) -> 'Column':
        """Return typed column based on array type."""
        return cls.type_map[type_map[C.scalar_type(array).id]](array)  # type: ignore

    def map(self, func: Callable, **kwargs) -> 'Column':
        return self.cast(C.map(self.array, func, **kwargs))

    @classmethod
    def fromscalar(cls, scalar: pa.ListScalar) -> Optional['Column']:
        return None if scalar.values is None else cls.cast(pa.chunked_array([scalar.values]))

    def unique(self, info: Info):
        """unique values and counts"""
        if 'counts' in selections(*info.selected_fields):
            return Set(*self.array.value_counts().flatten())
        return Set(self.array.unique())

    @doc_field
    def count(self, mode: str = 'only_valid') -> Long:
        """Return number of valid or null values."""
        return pc.count(self.array, mode=mode).as_py()

    def index(self, value, start: Long = 0, end: Optional[Long] = None) -> Long:
        """Return first index of occurrence of value; -1 indicates not found.

        May be faster than `count` for membership test.
        """
        return C.index(self.array, value, start, end)

    def values(self):
        """list of values"""
        return self.array.to_pylist()

    @getattr(functools, 'cached_property', property)  # added in Python 3.8
    def min_max(self):
        return C.min_max(self.array)

    def min(self):
        """minimum value"""
        return self.min_max['min']

    def max(self):
        """maximum value"""
        return self.min_max['max']

    def drop_null(self):
        """remove missing values from an array"""
        return self.array.drop_null().to_pylist()


@strawberry.type(description="unique values")
class Set(Generic[T]):
    length = doc_field(Column.length)
    counts: List[Long] = strawberry.field(description="list of counts")

    def __init__(self, array, counts=pa.array([])):
        self.array, self.counts = array, counts.to_pylist()

    @doc_field
    def values(self) -> List[Optional[T]]:
        """list of values"""
        return self.array.to_pylist()


def annotate(func, return_type, **annotations):
    """Return field from an annotated clone of the function."""
    clone = types.FunctionType(func.__code__, func.__globals__)
    annotations['return'] = return_type
    clone.__annotations__.update(func.__annotations__, **annotations)
    clone.__defaults__ = func.__defaults__
    return strawberry.field(clone, description=inspect.getdoc(func))


@strawberry.type
class NumericColumn(Column):
    def sum(self):
        """sum of the values"""
        return pc.sum(self.array).as_py()

    def product(self):
        """product of the values"""
        return pc.product(self.array).as_py()

    @doc_field
    def mean(self) -> Optional[float]:
        """mean of the values"""
        return pc.mean(self.array).as_py()

    @doc_field
    def stddev(self) -> Optional[float]:
        """standard deviation of the values"""
        return pc.stddev(self.array).as_py()

    @doc_field
    def variance(self) -> Optional[float]:
        """variance of the values"""
        return pc.variance(self.array).as_py()

    @doc_field
    def quantile(self, q: List[float] = [0.5], interpolation: str = 'linear') -> List[float]:
        """Return list of quantiles for values, defaulting to the median."""
        return pc.quantile(self.array, q=q, interpolation=interpolation).to_pylist()

    @doc_field
    def tdigest(
        self, q: List[float] = [0.5], delta: int = 100, buffer_size: int = 500
    ) -> List[float]:
        """Return list of approximate quantiles for values, defaulting to the median."""
        return pc.tdigest(self.array, q=q, delta=delta, buffer_size=buffer_size).to_pylist()

    def mode(self, length: int = 1):
        """mode of the values"""
        return Set(*pc.mode(self.array, length).flatten())


@strawberry.type(description="column of booleans")
class BooleanColumn(Column):
    index = annotate(Column.index, Long, value=bool)
    values = annotate(Column.values, List[Optional[bool]])
    unique = annotate(Column.unique, Set[bool])

    @doc_field
    def any(self) -> Optional[bool]:
        """whether any values evaluate to true"""
        return pc.any(self.array).as_py()

    @doc_field
    def all(self) -> Optional[bool]:
        """whether all values evaluate to true"""
        return pc.all(self.array).as_py()


@strawberry.type(description="column of ints")
class IntColumn(NumericColumn):
    index = annotate(Column.index, Long, value=int)
    values = annotate(Column.values, List[Optional[int]])
    unique = annotate(Column.unique, Set[int])
    sum = annotate(NumericColumn.sum, Optional[int])
    product = annotate(NumericColumn.product, Optional[int])
    mode = annotate(NumericColumn.mode, Set[int])
    min = annotate(Column.min, Optional[int])
    max = annotate(Column.max, Optional[int])
    drop_null = annotate(Column.drop_null, List[int])


@strawberry.type(description="column of longs")
class LongColumn(NumericColumn):
    index = annotate(Column.index, Long, value=Long)
    values = annotate(Column.values, List[Optional[Long]])
    unique = annotate(Column.unique, Set[Long])
    sum = annotate(NumericColumn.sum, Optional[Long])
    product = annotate(NumericColumn.product, Optional[Long])
    mode = annotate(NumericColumn.mode, Set[Long])
    min = annotate(Column.min, Optional[Long])
    max = annotate(Column.max, Optional[Long])
    drop_null = annotate(Column.drop_null, List[Long])


@strawberry.type(description="column of floats")
class FloatColumn(NumericColumn):
    index = annotate(Column.index, Long, value=float)
    values = annotate(Column.values, List[Optional[float]])
    unique = annotate(Column.unique, Set[float])
    sum = annotate(NumericColumn.sum, Optional[float])
    product = annotate(NumericColumn.product, Optional[float])
    mode = annotate(NumericColumn.mode, Set[float])
    min = annotate(Column.min, Optional[float])
    max = annotate(Column.max, Optional[float])
    drop_null = annotate(Column.drop_null, List[float])


@strawberry.type(description="column of decimals")
class DecimalColumn(Column):
    values = annotate(Column.values, List[Optional[Decimal]])
    unique = annotate(Column.unique, Set[Decimal])
    min = annotate(Column.min, Optional[Decimal])
    max = annotate(Column.max, Optional[Decimal])


@strawberry.type(description="column of dates")
class DateColumn(Column):
    index = annotate(Column.index, Long, value=date)
    values = annotate(Column.values, List[Optional[date]])
    unique = annotate(Column.unique, Set[date])
    min = annotate(Column.min, Optional[date])
    max = annotate(Column.max, Optional[date])
    drop_null = annotate(Column.drop_null, List[date])


@strawberry.type(description="column of datetimes")
class DateTimeColumn(Column):
    index = annotate(Column.index, Long, value=datetime)
    values = annotate(Column.values, List[Optional[datetime]])
    unique = annotate(Column.unique, Set[datetime])
    min = annotate(Column.min, Optional[datetime])
    max = annotate(Column.max, Optional[datetime])
    drop_null = annotate(Column.drop_null, List[datetime])


@strawberry.type(description="column of times")
class TimeColumn(Column):
    index = annotate(Column.index, Long, value=time)
    values = annotate(Column.values, List[Optional[time]])
    unique = annotate(Column.unique, Set[time])
    min = annotate(Column.min, Optional[time])
    max = annotate(Column.max, Optional[time])
    drop_null = annotate(Column.drop_null, List[time])


@strawberry.type(description="column of durations")
class DurationColumn(Column):
    index = annotate(Column.index, Long, value=timedelta)
    values = annotate(Column.values, List[Optional[timedelta]])


@strawberry.type(description="column of binaries")
class Base64Column(Column):
    index = annotate(Column.index, Long, value=bytes)
    values = annotate(Column.values, List[Optional[bytes]])
    unique = annotate(Column.unique, Set[bytes])
    drop_null = annotate(Column.drop_null, List[bytes])


@strawberry.type(description="column of strings")
class StringColumn(Column):
    index = annotate(Column.index, Long, value=str)
    values = annotate(Column.values, List[Optional[str]])
    unique = annotate(Column.unique, Set[str])
    min = annotate(Column.min, Optional[str])
    max = annotate(Column.max, Optional[str])
    drop_null = annotate(Column.drop_null, List[str])


@strawberry.type(description="column of lists")
class ListColumn(Column):
    @doc_field
    def values(self) -> List[Optional[Column]]:
        """list of columns"""
        return list(map(self.fromscalar, self.array))

    @doc_field
    def flatten(self) -> Column:
        """concatenation of all sub-lists"""
        return self.cast(pc.list_flatten(self.array))


@strawberry.type(description="column of structs")
class StructColumn(Column):
    @doc_field
    def names(self) -> List[str]:
        """field names"""
        return [field.name for field in self.array.type]

    @doc_field
    def column(self, name: str) -> Column:
        """Return struct field as a column."""
        return self.map(operator.methodcaller('field', name))


Column.type_map = {  # type: ignore
    bool: BooleanColumn,
    int: IntColumn,
    Long: LongColumn,
    float: FloatColumn,
    Decimal: DecimalColumn,
    date: DateColumn,
    datetime: DateTimeColumn,
    time: TimeColumn,
    timedelta: DurationColumn,
    bytes: Base64Column,
    str: StringColumn,
    list: ListColumn,
    dict: StructColumn,
}
