"""
GraphQL output types and resolvers.
"""
import contextlib
import functools
import inspect
import operator
import types
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Callable, List, Optional
import pyarrow as pa
import pyarrow.compute as pc
import strawberry
from strawberry.field import StrawberryField
from typing_extensions import Annotated
from .core import Column as C, ListChunk
from .inputs import BooleanQuery, IntQuery, LongQuery, FloatQuery, DecimalQuery, DateQuery
from .inputs import DateTimeQuery, TimeQuery, DurationQuery, Base64Query, StringQuery
from .scalars import Long, type_map


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
        """[arrow type](https://arrow.apache.org/docs/python/api/datatypes.html)"""
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

    def unique(self, info):
        """unique values and counts"""
        if 'counts' in selections(*info.selected_fields):
            return self.Set(*self.array.value_counts().flatten())
        return self.Set(self.array.unique())

    def count(self, **query) -> Long:
        """Return number of matching values.

        Optimized for `null`, and an empty query is equivalent to not `null`.
        """
        if query == {'equal': None}:
            return self.array.null_count
        if query in ({}, {'not_equal': None}):
            return len(self.array) - self.array.null_count
        return C.count(C.mask(self.array, **query), True)

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

    def sort(self, reverse: bool = False, length: Optional[Long] = None):
        """Return sorted values. Optimized for fixed length."""
        return C.sort(self.array, reverse, length).to_pylist()

    def drop_null(self):
        """remove missing values from an array"""
        return type(self)(self.array.drop_null())

    def fill_null(self, value):
        """Return values with null elements replaced."""
        return type(self)(C.fill_null(self.array, value))

    def min_element_wise(self, value, skip_nulls: bool = True):
        """Return element-wise minimum compared to scalar."""
        return type(self)(pc.min_element_wise(self.array, value, skip_nulls=skip_nulls))

    def max_element_wise(self, value, skip_nulls: bool = True):
        """Return element-wise maximum compared to scalar."""
        return type(self)(pc.max_element_wise(self.array, value, skip_nulls=skip_nulls))

    def between(self, unit: str, start=None, end=None) -> 'LongColumn':
        """Return duration between start and end."""
        if [start, end].count(None) != 1:
            raise ValueError("exactly one of `start` or `end` required")
        convert = functools.partial(pa.scalar, type=self.array.type)
        args = (self.array, convert(end)) if start is None else (convert(start), self.array)
        return LongColumn(getattr(pc, f'{unit}_between')(*args))


@strawberry.interface(description="unique values")
class Set:
    length = doc_field(Column.length)
    counts: List[Long] = strawberry.field(description="list of counts")

    def __init__(self, array, counts=pa.array([])):
        self.array, self.counts = array, counts.to_pylist()

    @classmethod
    def subclass(base, cls, name, description):
        namespace = {
            '__init__': Set.__init__,
            'values': annotate(Column.values, List[Optional[cls]]),
        }
        return strawberry.type(description=description)(type(name, (base,), namespace))


def annotate(func, return_type, **annotations):
    """Return field from an annotated clone of the function."""
    clone = types.FunctionType(func.__code__, func.__globals__)
    annotations['return'] = return_type
    clone.__annotations__.update(func.__annotations__, **annotations)
    clone.__defaults__ = func.__defaults__
    return strawberry.field(clone, description=inspect.getdoc(func))


@strawberry.interface(description="numeric column interface")
class NumericColumn:
    @doc_field
    def any(self) -> Optional[bool]:
        """whether any values evaluate to true"""
        return C.any(self.array)  # type: ignore

    @doc_field
    def all(self) -> Optional[bool]:
        """whether all values evaluate to true"""
        return C.all(self.array)  # type: ignore

    def sum(self):
        """sum of the values"""
        return pc.sum(self.array).as_py()

    def product(self):
        """product of the values"""
        return pc.product(self.array).as_py()

    @doc_field
    def mean(self) -> Optional[float]:
        """mean of the values"""
        return pc.mean(self.array).as_py()  # type: ignore

    @doc_field
    def stddev(self) -> Optional[float]:
        """standard deviation of the values"""
        return pc.stddev(self.array).as_py()  # type: ignore

    @doc_field
    def variance(self) -> Optional[float]:
        """variance of the values"""
        return pc.variance(self.array).as_py()  # type: ignore

    @doc_field
    def quantile(self, q: List[float] = [0.5], interpolation: str = 'linear') -> List[float]:
        """Return list of quantiles for values, defaulting to the median."""
        return pc.quantile(self.array, q=q, interpolation=interpolation).to_pylist()  # type: ignore

    @doc_field
    def tdigest(
        self, q: List[float] = [0.5], delta: int = 100, buffer_size: int = 500
    ) -> List[float]:
        """Return list of approximate quantiles for values, defaulting to the median."""
        return pc.tdigest(self.array, q=q, delta=delta, buffer_size=buffer_size).to_pylist()  # type: ignore

    @doc_field
    def logb(self, base: float) -> 'FloatColumn':
        """Return log of values to base."""
        return FloatColumn(pc.logb(self.array, base))  # type: ignore

    def mode(self, length: int = 1):
        """mode of the values"""
        return self.Set(*pc.mode(self.array, length).flatten())  # type: ignore

    def add(self, value):
        """Return values added to scalar."""
        return type(self)(pc.add(value, self.array))

    def subtract(self, value):
        """Return values subtracted *from* scalar."""
        return type(self)(pc.subtract(value, self.array))

    def multiply(self, value):
        """Return values multiplied by scalar."""
        return type(self)(pc.multiply(value, self.array))

    def divide(self, value):
        """Return values divided *into* scalar."""
        return type(self)(pc.divide(value, self.array))

    def power(self, base=None, exponent=None):
        """Return values raised to power."""
        if [base, exponent].count(None) != 1:
            raise ValueError("exactly one of `base` or `exponent` required")
        args = (self.array, exponent) if base is None else (base, self.array)
        return type(self)(pc.power(*args))


@strawberry.type(description="column of booleans")
class BooleanColumn(Column):
    count = BooleanQuery.resolver(Column.count)
    index = annotate(Column.index, Long, value=bool)
    any = doc_field(NumericColumn.any)
    all = doc_field(NumericColumn.all)
    values = annotate(Column.values, List[Optional[bool]])
    Set = Set.subclass(bool, "BooleanSet", "unique booleans")
    unique = annotate(Column.unique, Set)


@strawberry.type(description="column of ints")
class IntColumn(Column, NumericColumn):
    count = IntQuery.resolver(Column.count)
    index = annotate(Column.index, Long, value=int)
    values = annotate(Column.values, List[Optional[int]])
    Set = Set.subclass(int, "IntSet", "unique ints")
    unique = annotate(Column.unique, Set)
    sort = annotate(Column.sort, List[Optional[int]])
    sum = annotate(NumericColumn.sum, Optional[int])
    product = annotate(NumericColumn.product, Optional[int])
    mode = annotate(NumericColumn.mode, Set)
    min = annotate(Column.min, Optional[int])
    max = annotate(Column.max, Optional[int])
    drop_null = annotate(Column.drop_null, 'IntColumn')
    fill_null = annotate(Column.fill_null, 'IntColumn', value=int)
    add = annotate(NumericColumn.add, 'IntColumn', value=int)
    subtract = annotate(NumericColumn.subtract, 'IntColumn', value=int)
    multiply = annotate(NumericColumn.multiply, 'IntColumn', value=int)
    divide = annotate(NumericColumn.divide, 'IntColumn', value=int)
    power = annotate(NumericColumn.power, 'IntColumn', base=Optional[int], exponent=Optional[int])
    min_element_wise = annotate(Column.min_element_wise, 'IntColumn', value=int)
    max_element_wise = annotate(Column.max_element_wise, 'IntColumn', value=int)


@strawberry.type(description="column of longs")
class LongColumn(Column, NumericColumn):
    count = LongQuery.resolver(Column.count)
    index = annotate(Column.index, Long, value=Long)
    values = annotate(Column.values, List[Optional[Long]])
    Set = Set.subclass(Long, "LongSet", "unique longs")
    unique = annotate(Column.unique, Set)
    sort = annotate(Column.sort, List[Optional[Long]])
    sum = annotate(NumericColumn.sum, Optional[Long])
    product = annotate(NumericColumn.product, Optional[Long])
    mode = annotate(NumericColumn.mode, Set)
    min = annotate(Column.min, Optional[Long])
    max = annotate(Column.max, Optional[Long])
    drop_null = annotate(Column.drop_null, 'LongColumn')
    fill_null = annotate(Column.fill_null, 'LongColumn', value=Long)
    add = annotate(NumericColumn.add, 'LongColumn', value=Long)
    subtract = annotate(NumericColumn.subtract, 'LongColumn', value=Long)
    multiply = annotate(NumericColumn.multiply, 'LongColumn', value=Long)
    divide = annotate(NumericColumn.divide, 'LongColumn', value=Long)
    power = annotate(
        NumericColumn.power, 'LongColumn', base=Optional[Long], exponent=Optional[Long]
    )
    min_element_wise = annotate(Column.min_element_wise, 'LongColumn', value=Long)
    max_element_wise = annotate(Column.max_element_wise, 'LongColumn', value=Long)


@strawberry.type(description="column of floats")
class FloatColumn(Column, NumericColumn):
    count = FloatQuery.resolver(Column.count)
    index = annotate(Column.index, Long, value=float)
    values = annotate(Column.values, List[Optional[float]])
    Set = Set.subclass(float, "FloatSet", "unique floats")
    unique = annotate(Column.unique, Set)
    sort = annotate(Column.sort, List[Optional[float]])
    sum = annotate(NumericColumn.sum, Optional[float])
    product = annotate(NumericColumn.product, Optional[float])
    mode = annotate(NumericColumn.mode, Set)
    min = annotate(Column.min, Optional[float])
    max = annotate(Column.max, Optional[float])
    drop_null = annotate(Column.drop_null, 'FloatColumn')
    fill_null = annotate(Column.fill_null, 'FloatColumn', value=float)
    add = annotate(NumericColumn.add, 'FloatColumn', value=float)
    subtract = annotate(NumericColumn.subtract, 'FloatColumn', value=float)
    multiply = annotate(NumericColumn.multiply, 'FloatColumn', value=float)
    divide = annotate(NumericColumn.divide, 'FloatColumn', value=float)
    power = annotate(
        NumericColumn.power, 'FloatColumn', base=Optional[float], exponent=Optional[float]
    )
    min_element_wise = annotate(Column.min_element_wise, 'FloatColumn', value=float)
    max_element_wise = annotate(Column.max_element_wise, 'FloatColumn', value=float)

    @doc_field
    def round(
        self, ndigits: int = 0, multiple: float = 1.0, round_mode: str = 'half_to_even'
    ) -> 'FloatColumn':
        """Return log of values to base."""
        if ndigits != 0 and multiple != 1.0:
            raise ValueError("only one of `ndigits` or `multiple` allowed")
        if multiple == 1:
            array = pc.round(self.array, ndigits=ndigits, round_mode=round_mode)
        else:
            array = pc.round_to_multiple(self.array, multiple=multiple, round_mode=round_mode)
        return FloatColumn(array)


@strawberry.type(description="column of decimals")
class DecimalColumn(Column):
    count = DecimalQuery.resolver(Column.count)
    values = annotate(Column.values, List[Optional[Decimal]])
    Set = Set.subclass(Decimal, "DecimalSet", "unique decimals")
    unique = annotate(Column.unique, Set)
    sort = annotate(Column.sort, List[Optional[Decimal]])
    min = annotate(Column.min, Optional[Decimal])
    max = annotate(Column.max, Optional[Decimal])


class TemporalColumn:
    def floor_temporal(self, unit: str, multiple: int = 1):
        """Round down to nearest multiple and time unit."""
        return type(self)(pc.floor_temporal(self.array, multiple, unit))  # type: ignore

    def round_temporal(self, unit: str, multiple: int = 1):
        """Round to nearest multiple and time unit."""
        return type(self)(pc.round_temporal(self.array, multiple, unit))  # type: ignore

    def ceil_temporal(self, unit: str, multiple: int = 1):
        """Round up to nearest multiple and time unit."""
        return type(self)(pc.ceil_temporal(self.array, multiple, unit))  # type: ignore


@strawberry.type(description="column of dates")
class DateColumn(Column):
    count = DateQuery.resolver(Column.count)
    index = annotate(Column.index, Long, value=date)
    values = annotate(Column.values, List[Optional[date]])
    Set = Set.subclass(date, "DateSet", "unique dates")
    unique = annotate(Column.unique, Set)
    sort = annotate(Column.sort, List[Optional[date]])
    min = annotate(Column.min, Optional[date])
    max = annotate(Column.max, Optional[date])
    drop_null = annotate(Column.drop_null, 'DateColumn')
    fill_null = annotate(Column.fill_null, 'DateColumn', value=date)
    min_element_wise = annotate(Column.min_element_wise, 'DateColumn', value=date)
    max_element_wise = annotate(Column.max_element_wise, 'DateColumn', value=date)
    between = annotate(Column.between, LongColumn, start=Optional[date], end=Optional[date])
    floor_temporal = annotate(TemporalColumn.floor_temporal, 'DateColumn')
    round_temporal = annotate(TemporalColumn.round_temporal, 'DateColumn')
    ceil_temporal = annotate(TemporalColumn.ceil_temporal, 'DateColumn')

    @doc_field
    def strftime(self, format: str = '%Y-%m-%dT%H:%M:%S', locale: str = 'C') -> 'StringColumn':
        """Return formatted temporal values according to a format string."""
        return StringColumn(pc.strftime(self.array, format=format, locale=locale))


@strawberry.type(description="column of datetimes")
class DateTimeColumn(Column):
    count = DateTimeQuery.resolver(Column.count)
    index = annotate(Column.index, Long, value=datetime)
    values = annotate(Column.values, List[Optional[datetime]])
    Set = Set.subclass(datetime, "DatetimeSet", "unique datetimes")
    unique = annotate(Column.unique, Set)
    sort = annotate(Column.sort, List[Optional[datetime]])
    min = annotate(Column.min, Optional[datetime])
    max = annotate(Column.max, Optional[datetime])
    drop_null = annotate(Column.drop_null, 'DateTimeColumn')
    fill_null = annotate(Column.fill_null, 'DateTimeColumn', value=datetime)
    min_element_wise = annotate(Column.min_element_wise, 'DateTimeColumn', value=datetime)
    max_element_wise = annotate(Column.max_element_wise, 'DateTimeColumn', value=datetime)
    between = annotate(Column.between, LongColumn, start=Optional[datetime], end=Optional[datetime])
    strftime = doc_field(DateColumn.strftime)
    floor_temporal = annotate(TemporalColumn.floor_temporal, 'DateTimeColumn')
    round_temporal = annotate(TemporalColumn.round_temporal, 'DateTimeColumn')
    ceil_temporal = annotate(TemporalColumn.ceil_temporal, 'DateTimeColumn')

    @doc_field
    def subtract(self, value: datetime) -> 'DurationColumn':
        """Return values subtracted *from* scalar."""
        return DurationColumn(pc.subtract(value, self.array))

    @doc_field
    def assume_timezone(
        self, timezone: str, ambiguous: str = 'raise', nonexistent: str = 'raise'
    ) -> 'DateTimeColumn':
        """Convert naive timestamps to timezone-aware timestamps."""
        return DateTimeColumn(
            pc.assume_timezone(self.array, timezone, ambiguous=ambiguous, nonexistent=nonexistent)
        )


@strawberry.type(description="column of times")
class TimeColumn(Column):
    count = TimeQuery.resolver(Column.count)
    index = annotate(Column.index, Long, value=time)
    values = annotate(Column.values, List[Optional[time]])
    Set = Set.subclass(time, "TimeSet", "unique times")
    unique = annotate(Column.unique, Set)
    sort = annotate(Column.sort, List[Optional[time]])
    min = annotate(Column.min, Optional[time])
    max = annotate(Column.max, Optional[time])
    drop_null = annotate(Column.drop_null, 'TimeColumn')
    fill_null = annotate(Column.fill_null, 'TimeColumn', value=time)
    min_element_wise = annotate(Column.min_element_wise, 'TimeColumn', value=time)
    max_element_wise = annotate(Column.max_element_wise, 'TimeColumn', value=time)
    between = annotate(Column.between, LongColumn, start=Optional[time], end=Optional[time])
    floor_temporal = annotate(TemporalColumn.floor_temporal, 'TimeColumn')
    round_temporal = annotate(TemporalColumn.round_temporal, 'TimeColumn')
    ceil_temporal = annotate(TemporalColumn.ceil_temporal, 'TimeColumn')


@strawberry.type(description="column of durations")
class DurationColumn(Column):
    count = DurationQuery.resolver(Column.count)
    index = annotate(Column.index, Long, value=timedelta)
    values = annotate(Column.values, List[Optional[timedelta]])


@strawberry.type(description="column of binaries")
class Base64Column(Column):
    count = Base64Query.resolver(Column.count)
    index = annotate(Column.index, Long, value=bytes)
    any = doc_field(NumericColumn.any)
    all = doc_field(NumericColumn.all)
    values = annotate(Column.values, List[Optional[bytes]])
    Set = Set.subclass(bytes, "Base64Set", "unique binaries")
    unique = annotate(Column.unique, Set)
    sort = annotate(Column.sort, List[Optional[bytes]])
    drop_null = annotate(Column.drop_null, 'Base64Column')
    fill_null = annotate(Column.fill_null, 'Base64Column', value=bytes)

    @doc_field
    def binary_replace_slice(self, start: int, stop: int, replacement: str) -> 'Base64Column':
        """Replace a slice of a binary string with `replacement`."""
        kwargs = dict(start=start, stop=stop, replacement=replacement)
        return Base64Column(pc.binary_replace_slice(self.array, **kwargs))


@strawberry.type(description="column of strings")
class StringColumn(Column):
    count = StringQuery.resolver(Column.count)
    index = annotate(Column.index, Long, value=str)
    any = doc_field(NumericColumn.any)
    all = doc_field(NumericColumn.all)
    values = annotate(Column.values, List[Optional[str]])
    Set = Set.subclass(str, "StringSet", "unique strings")
    unique = annotate(Column.unique, Set)
    sort = annotate(Column.sort, List[Optional[str]])
    min = annotate(Column.min, Optional[str])
    max = annotate(Column.max, Optional[str])
    drop_null = annotate(Column.drop_null, 'StringColumn')
    fill_null = annotate(Column.fill_null, 'StringColumn', value=str)

    @doc_field
    def split(
        self, pattern: str = '', max_splits: int = -1, reverse: bool = False, regex: bool = False
    ) -> 'ListColumn':
        """Return strings split on pattern, by default whitespace."""
        kwargs = {'max_splits': max_splits, 'reverse': reverse}
        if pattern:
            func = pc.split_pattern_regex if regex else pc.split_pattern
            return ListColumn(func(self.array, pattern=pattern, **kwargs))
        return ListColumn(pc.utf8_split_whitespace(self.array, **kwargs))

    @doc_field
    def utf8_ltrim(self, characters: str = '') -> 'StringColumn':
        """Trim leading characters, by default whitespace."""
        if characters:
            return StringColumn(pc.utf8_ltrim(self.array, characters=characters))
        return StringColumn(pc.utf8_ltrim_whitespace(self.array))

    @doc_field
    def utf8_rtrim(self, characters: str = '') -> 'StringColumn':
        """Trim trailing characters, by default whitespace."""
        if characters:
            return StringColumn(pc.utf8_rtrim(self.array, characters=characters))
        return StringColumn(pc.utf8_rtrim_whitespace(self.array))

    @doc_field
    def utf8_trim(self, characters: str = '') -> 'StringColumn':
        """Trim trailing characters, by default whitespace."""
        if characters:
            return StringColumn(pc.utf8_trim(self.array, characters=characters))
        return StringColumn(pc.utf8_trim_whitespace(self.array))

    @doc_field
    def utf8_lpad(self, width: int, padding: str = ' ') -> 'StringColumn':
        """Right-align strings by padding with a given character."""
        return StringColumn(pc.utf8_lpad(self.array, width=width, padding=padding))

    @doc_field
    def utf8_rpad(self, width: int, padding: str = ' ') -> 'StringColumn':
        """Left-align strings by padding with a given character."""
        return StringColumn(pc.utf8_rpad(self.array, width=width, padding=padding))

    @doc_field
    def utf8_center(self, width: int, padding: str = ' ') -> 'StringColumn':
        """Center strings by padding with a given character."""
        return StringColumn(pc.utf8_center(self.array, width=width, padding=padding))

    @doc_field
    def utf8_replace_slice(self, start: int, stop: int, replacement: str) -> 'StringColumn':
        """Replace a slice of a string with `replacement`."""
        kwargs = dict(start=start, stop=stop, replacement=replacement)
        return StringColumn(pc.utf8_replace_slice(self.array, **kwargs))

    @doc_field
    def replace_substring(
        self, pattern: str, replacement: str, max_replacements: int = -1
    ) -> 'StringColumn':
        """Replace non-overlapping substrings that match pattern."""
        kwargs = dict(pattern=pattern, replacement=replacement, max_replacements=max_replacements)
        return StringColumn(pc.replace_substring(self.array, **kwargs))

    @doc_field
    def strptime(self, format: str = '%Y-%m-%dT%H:%M:%S', unit: str = 'ms') -> DateTimeColumn:
        """Return parsed timestamps."""
        return DateTimeColumn(pc.strptime(self.array, format=format, unit=unit))

    @doc_field
    def utf8_slice_codeunits(
        self, start: int = 0, stop: Optional[int] = None, step: int = 1
    ) -> 'StringColumn':
        """Return slice strings, measured in utf8 codeunits."""
        return StringColumn(pc.utf8_slice_codeunits(self.array, start, stop, step))


@strawberry.type(description="column of lists")
class ListColumn(Column):
    @doc_field
    def values(self) -> List[Optional[Column]]:
        """list of columns"""
        return list(map(self.fromscalar, self.array))

    @doc_field
    def count(self, mode: str = 'only_valid') -> LongColumn:
        """non-null count of each list scalar"""
        return LongColumn(self.map(ListChunk.count, mode=mode).array)

    @doc_field
    def count_distinct(self, mode: str = 'only_valid') -> LongColumn:
        """non-null distinct count of each list scalar"""
        return LongColumn(self.map(ListChunk.count_distinct, mode=mode).array)

    @doc_field
    def value_length(self) -> LongColumn:
        """length of each list scalar"""
        return LongColumn(pc.list_value_length(self.array))

    @doc_field
    def flatten(self) -> Column:
        """concatenation of all sub-lists"""
        return self.cast(pc.list_flatten(self.array))

    @doc_field
    def unique(self) -> 'ListColumn':
        """unique values within each scalar"""
        return self.map(ListChunk.unique)  # type: ignore

    @doc_field
    def distinct(self, mode: str = 'only_valid') -> 'ListColumn':
        """non-null distinct values within each scalar"""
        return self.map(ListChunk.distinct, mode=mode)  # type: ignore

    @doc_field
    def element(self, index: Long = 0) -> Column:
        """element at index of each list scalar; defaults to null"""
        with contextlib.suppress(ValueError):
            return self.cast(pc.list_element(self.array, index))
        return self.map(ListChunk.element, index=index)

    @doc_field
    def min(self, skip_nulls: bool = True, min_count: int = 1) -> Column:
        """min value of each list scalar"""
        return self.map(ListChunk.min, skip_nulls=skip_nulls, min_count=min_count)

    @doc_field
    def max(self, skip_nulls: bool = True, min_count: int = 1) -> Column:
        """max value of each list scalar"""
        return self.map(ListChunk.max, skip_nulls=skip_nulls, min_count=min_count)

    @doc_field
    def sum(self, skip_nulls: bool = True, min_count: int = 1) -> Column:
        """sum of each list scalar"""
        return self.map(ListChunk.sum, skip_nulls=skip_nulls, min_count=min_count)

    @doc_field
    def product(self, skip_nulls: bool = True, min_count: int = 1) -> Column:
        """product of each list scalar"""
        return self.map(ListChunk.product, skip_nulls=skip_nulls, min_count=min_count)

    @doc_field
    def mean(self, skip_nulls: bool = True, min_count: int = 1) -> FloatColumn:
        """mean of each list scalar"""
        return self.map(ListChunk.mean, skip_nulls=skip_nulls, min_count=min_count)  # type: ignore

    @doc_field
    def mode(self, n: int = 1, skip_nulls: bool = True, min_count: int = 0) -> 'ListColumn':
        """mode of each list scalar"""
        return self.map(ListChunk.mode, n=n, skip_nulls=skip_nulls, min_count=min_count)  # type: ignore

    @doc_field
    def quantile(
        self,
        q: List[float] = [0.5],
        interpolation: str = 'linear',
        skip_nulls: bool = True,
        min_count: int = 0,
    ) -> 'ListColumn':
        """quantile of each list scalar"""
        return self.map(ListChunk.quantile, q=q, interpolation=interpolation, skip_nulls=skip_nulls, min_count=min_count)  # type: ignore

    @doc_field
    def tdigest(
        self,
        q: List[float] = [0.5],
        delta: int = 100,
        buffer_size: int = 500,
        skip_nulls: bool = True,
        min_count: int = 0,
    ) -> 'ListColumn':
        """approximate quantile of each list scalar"""
        return self.map(ListChunk.tdigest, q=q, delta=delta, buffer_size=buffer_size, skip_nulls=skip_nulls, min_count=min_count)  # type: ignore

    @doc_field
    def stddev(self, ddof: int = 0, skip_nulls: bool = True, min_count: int = 0) -> FloatColumn:
        """stddev of each list scalar"""
        return self.map(ListChunk.stddev, ddof=ddof, skip_nulls=skip_nulls, min_count=min_count)  # type: ignore

    @doc_field
    def variance(self, ddof: int = 0, skip_nulls: bool = True, min_count: int = 0) -> FloatColumn:
        """variance of each list scalar"""
        return self.map(ListChunk.variance, ddof=ddof, skip_nulls=skip_nulls, min_count=min_count)  # type: ignore

    @doc_field
    def any(self, skip_nulls: bool = True, min_count: int = 1) -> BooleanColumn:
        """any true of each list scalar"""
        return self.map(ListChunk.any, skip_nulls=skip_nulls, min_count=min_count)  # type: ignore

    @doc_field
    def all(self, skip_nulls: bool = True, min_count: int = 1) -> BooleanColumn:
        """all true of each list scalar"""
        return self.map(ListChunk.all, skip_nulls=skip_nulls, min_count=min_count)  # type: ignore

    @doc_field
    def binary_join(self, separator: bytes) -> Base64Column:
        """Join a list of binary strings together with a `separator` to form a single string."""
        return Base64Column(pc.binary_join(self.array, separator))

    @doc_field
    def string_join(self, separator: str) -> StringColumn:
        """Join a list of strings together with a `separator` to form a single string."""
        return StringColumn(pc.binary_join(self.array, separator))


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
