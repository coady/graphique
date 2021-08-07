"""
GraphQL output types and resolvers.
"""
import functools
import operator
import types
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Callable, List, Optional
import pyarrow as pa
import pyarrow.compute as pc
import strawberry
from cached_property import cached_property
from strawberry.field import StrawberryField
from typing_extensions import Annotated
from .core import Column as C, ListChunk
from .inputs import BooleanQuery, IntQuery, LongQuery, FloatQuery, DecimalQuery, DateQuery
from .inputs import DateTimeQuery, TimeQuery, DurationQuery, BinaryQuery, StringFilter
from .inputs import Field, resolve_annotations
from .scalars import Long, classproperty, type_map


def selections(*nodes):
    """Return set of field name selections."""
    names = set()
    for node in nodes:
        selections = getattr(node.selection_set, 'selections', [])
        names |= {node.name.value for node in selections if hasattr(node, 'name')}
    return names


def doc_field(func: Optional[Callable] = None, **kwargs: str) -> StrawberryField:
    """Return strawberry field with argument and docstring descriptions."""
    if func is None:
        return functools.partial(doc_field, **kwargs)
    for name in kwargs:
        argument = strawberry.argument(description=kwargs[name])
        func.__annotations__[name] = Annotated[func.__annotations__[name], argument]
    return strawberry.field(func, description=func.__doc__)


@strawberry.interface(description="column interface")
class Column:
    def __init__(self, array):
        self.array = array

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

    def map(self, func: Callable) -> 'Column':
        return self.cast(pa.chunked_array(C.map(func, self.array)))

    @classmethod
    def fromscalar(cls, scalar: pa.ListScalar) -> Optional['Column']:
        return None if scalar.values is None else cls.cast(pa.chunked_array([scalar.values]))

    def unique(self, info):
        """unique values and counts"""
        if 'counts' in selections(*info.field_nodes):
            return self.Set(*self.array.value_counts().flatten())
        return self.Set(self.array.unique())

    def count(self, **query) -> Long:
        """Return number of matching values.
        Optimized for `null`, and empty queries will attempt boolean conversion."""
        if query == {'equal': None}:
            return self.array.null_count
        if query == {'not_equal': None}:
            return len(self.array) - self.array.null_count
        return C.count(C.mask(self.array, **query), True)

    def index(self, value, start: Long = 0, end: Optional[Long] = None) -> Long:
        """Return first index of occurrence of value; -1 indicates not found.
        May be faster than `count` for membership test."""
        return C.index(self.array, value, start, end)

    def values(self):
        """list of values"""
        return self.array.to_pylist()

    def min(self):
        """minimum value"""
        return C.min(self.array)

    def max(self):
        """maximum value"""
        return C.max(self.array)

    def sort(self, reverse: bool = False, length: Optional[Long] = None):
        """Return sorted values. Optimized for fixed length."""
        return C.sort(self.array, reverse, length).to_pylist()

    def fill_null(self, value):
        """Return values with null elements replaced."""
        return type(self)(C.fill_null(self.array, value))

    def min_element_wise(self, value, skip_nulls: bool = True):
        """Return element-wise minimum compared to scalar."""
        return type(self)(pc.min_element_wise(self.array, value, skip_nulls=skip_nulls))

    def max_element_wise(self, value, skip_nulls: bool = True):
        """Return element-wise maximum compared to scalar."""
        return type(self)(pc.max_element_wise(self.array, value, skip_nulls=skip_nulls))


@strawberry.interface(description="unique values")
class Set:
    length = doc_field(Column.length)

    def __init__(self, array, count=pa.array([])):
        self.array, self.count = array, count

    @doc_field
    def counts(self) -> List[Long]:
        """list of counts"""
        return self.count.to_pylist()

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
    return strawberry.field(clone, description=func.__doc__)


@strawberry.interface(description="numeric column interface")
class NumericColumn:
    @doc_field
    def any(self) -> bool:
        """whether any values evaluate to true"""
        return C.any(self.array)  # type: ignore

    @doc_field
    def all(self) -> bool:
        """whether all values evaluate to true"""
        return C.all(self.array)  # type: ignore

    def sum(self):
        """sum of the values"""
        return pc.sum(self.array).as_py()

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

    def mode(self, length: int = 1):
        """mode of the values"""
        return self.Set(*pc.mode(self.array, length).flatten())  # type: ignore

    @cached_property
    def min_max(self):
        return pc.min_max(self.array).as_py()

    def min(self):
        """minimum value"""
        return self.min_max['min']

    def max(self):
        """maximum value"""
        return self.min_max['max']

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
    __init__ = Column.__init__  # type: ignore
    count = BooleanQuery.resolver(Column.count)
    index = annotate(Column.index, Long, value=bool)
    any = doc_field(NumericColumn.any)
    all = doc_field(NumericColumn.all)
    values = annotate(Column.values, List[Optional[bool]])
    Set = Set.subclass(bool, "BooleanSet", "unique booleans")
    unique = annotate(Column.unique, Set)


@strawberry.type(description="column of ints")
class IntColumn(Column, NumericColumn):
    __init__ = Column.__init__  # type: ignore
    count = IntQuery.resolver(Column.count)
    index = annotate(Column.index, Long, value=int)
    values = annotate(Column.values, List[Optional[int]])
    Set = Set.subclass(int, "IntSet", "unique ints")
    unique = annotate(Column.unique, Set)
    sort = annotate(Column.sort, List[Optional[int]])
    sum = annotate(NumericColumn.sum, Optional[int])
    mode = annotate(NumericColumn.mode, Set)
    min = annotate(NumericColumn.min, Optional[int])
    max = annotate(NumericColumn.max, Optional[int])
    unique = annotate(Column.unique, Set)
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
    __init__ = Column.__init__  # type: ignore
    count = LongQuery.resolver(Column.count)
    index = annotate(Column.index, Long, value=Long)
    values = annotate(Column.values, List[Optional[Long]])
    Set = Set.subclass(Long, "LongSet", "unique longs")
    unique = annotate(Column.unique, Set)
    sort = annotate(Column.sort, List[Optional[Long]])
    sum = annotate(NumericColumn.sum, Optional[Long])
    mode = annotate(NumericColumn.mode, Set)
    min = annotate(NumericColumn.min, Optional[Long])
    max = annotate(NumericColumn.max, Optional[Long])
    unique = annotate(Column.unique, Set)
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
    __init__ = Column.__init__  # type: ignore
    count = FloatQuery.resolver(Column.count)
    index = annotate(Column.index, Long, value=float)
    values = annotate(Column.values, List[Optional[float]])
    Set = Set.subclass(float, "FloatSet", "unique floats")
    unique = annotate(Column.unique, Set)
    sort = annotate(Column.sort, List[Optional[float]])
    sum = annotate(NumericColumn.sum, Optional[float])
    mode = annotate(NumericColumn.mode, Set)
    min = annotate(NumericColumn.min, Optional[float])
    max = annotate(NumericColumn.max, Optional[float])
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


@strawberry.type(description="column of decimals")
class DecimalColumn(Column):
    __init__ = Column.__init__  # type: ignore
    count = DecimalQuery.resolver(Column.count)
    values = annotate(Column.values, List[Optional[Decimal]])
    Set = Set.subclass(Decimal, "DecimalSet", "unique decimals")
    unique = annotate(Column.unique, Set)
    sort = annotate(Column.sort, List[Optional[Decimal]])
    min = annotate(Column.min, Optional[Decimal])
    max = annotate(Column.max, Optional[Decimal])


@strawberry.type(description="column of dates")
class DateColumn(Column):
    __init__ = Column.__init__  # type: ignore
    count = DateQuery.resolver(Column.count)
    index = annotate(Column.index, Long, value=date)
    values = annotate(Column.values, List[Optional[date]])
    Set = Set.subclass(date, "DateSet", "unique dates")
    unique = annotate(Column.unique, Set)
    sort = annotate(Column.sort, List[Optional[date]])
    min = annotate(Column.min, Optional[date])
    max = annotate(Column.max, Optional[date])
    fill_null = annotate(Column.fill_null, 'DateColumn', value=date)
    min_element_wise = annotate(Column.min_element_wise, 'DateColumn', value=date)
    max_element_wise = annotate(Column.max_element_wise, 'DateColumn', value=date)


@strawberry.type(description="column of datetimes")
class DateTimeColumn(Column):
    __init__ = Column.__init__  # type: ignore
    count = DateTimeQuery.resolver(Column.count)
    index = annotate(Column.index, Long, value=datetime)
    values = annotate(Column.values, List[Optional[datetime]])
    Set = Set.subclass(datetime, "DatetimeSet", "unique datetimes")
    unique = annotate(Column.unique, Set)
    sort = annotate(Column.sort, List[Optional[datetime]])
    min = annotate(Column.min, Optional[datetime])
    max = annotate(Column.max, Optional[datetime])
    fill_null = annotate(Column.fill_null, 'DateTimeColumn', value=datetime)
    min_element_wise = annotate(Column.min_element_wise, 'DateTimeColumn', value=datetime)
    max_element_wise = annotate(Column.max_element_wise, 'DateTimeColumn', value=datetime)

    @doc_field
    def subtract(self, value: datetime) -> 'DurationColumn':
        """Return values subtracted *from* scalar."""
        return DurationColumn(pc.subtract(value, self.array))


@strawberry.type(description="column of times")
class TimeColumn(Column):
    __init__ = Column.__init__  # type: ignore
    count = TimeQuery.resolver(Column.count)
    index = annotate(Column.index, Long, value=time)
    values = annotate(Column.values, List[Optional[time]])
    Set = Set.subclass(time, "TimeSet", "unique times")
    unique = annotate(Column.unique, Set)
    sort = annotate(Column.sort, List[Optional[time]])
    min = annotate(Column.min, Optional[time])
    max = annotate(Column.max, Optional[time])
    fill_null = annotate(Column.fill_null, 'TimeColumn', value=time)
    min_element_wise = annotate(Column.min_element_wise, 'TimeColumn', value=time)
    max_element_wise = annotate(Column.max_element_wise, 'TimeColumn', value=time)


@strawberry.type(description="column of durations")
class DurationColumn(Column):
    __init__ = Column.__init__  # type: ignore
    count = DurationQuery.resolver(Column.count)
    index = annotate(Column.index, Long, value=timedelta)
    values = annotate(Column.values, List[Optional[timedelta]])


@strawberry.type(description="column of binaries")
class BinaryColumn(Column):
    __init__ = Column.__init__  # type: ignore
    count = BinaryQuery.resolver(Column.count)
    index = annotate(Column.index, Long, value=bytes)
    any = doc_field(NumericColumn.any)
    all = doc_field(NumericColumn.all)
    values = annotate(Column.values, List[Optional[bytes]])
    Set = Set.subclass(bytes, "BinarySet", "unique binaries")
    unique = annotate(Column.unique, Set)
    sort = annotate(Column.sort, List[Optional[bytes]])
    fill_null = annotate(Column.fill_null, 'BinaryColumn', value=bytes)

    @doc_field
    def binary_replace_slice(self, start: int, stop: int, replacement: str) -> 'BinaryColumn':
        """Replace a slice of a binary string with `replacement`."""
        kwargs = dict(start=start, stop=stop, replacement=replacement)
        return BinaryColumn(pc.binary_replace_slice(self.array, **kwargs))


@strawberry.type(description="column of strings")
class StringColumn(Column):
    __init__ = Column.__init__  # type: ignore
    count = StringFilter.resolver(Column.count)
    index = annotate(Column.index, Long, value=str)
    any = doc_field(NumericColumn.any)
    all = doc_field(NumericColumn.all)
    values = annotate(Column.values, List[Optional[str]])
    Set = Set.subclass(str, "StringSet", "unique strings")
    unique = annotate(Column.unique, Set)
    sort = annotate(Column.sort, List[Optional[str]])
    min = annotate(Column.min, Optional[str])
    max = annotate(Column.max, Optional[str])
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


@strawberry.type(description="column of lists")
class ListColumn(Column):
    __init__ = Column.__init__  # type: ignore
    aggregates = (
        'count',
        'unique',
        'first',
        'last',
        'min',
        'max',
        'sum',
        'mean',
        'mode',
        'quantile',
        'stddev',
        'variance',
        'any',
        'all',
    )

    @classproperty
    def resolver(cls) -> Callable:
        """a decorator which transforms aggregate functions into arguments"""
        annotations = {}
        for key in cls.aggregates:
            argument = strawberry.argument(description=getattr(cls, key).__doc__)
            annotations[key] = Annotated[List[Field], argument]
        defaults = dict.fromkeys(cls.aggregates, [])  # type: dict
        return functools.partial(resolve_annotations, annotations=annotations, defaults=defaults)

    @doc_field
    def values(self) -> List[Optional[Column]]:
        """list of columns"""
        return list(map(self.fromscalar, self.array))

    @doc_field
    def count(self) -> LongColumn:
        """number of values of each list scalar"""
        return LongColumn(self.map(operator.methodcaller('value_lengths')).array)

    @doc_field
    def flatten(self) -> Column:
        """concatenation of all sub-lists"""
        return self.map(operator.methodcaller('flatten'))

    @doc_field
    def unique(self) -> 'ListColumn':
        """unique values within each scalar"""
        return self.map(ListChunk.unique)  # type: ignore

    @doc_field
    def first(self) -> Column:
        """first value of each list scalar"""
        return self.map(ListChunk.first)

    @doc_field
    def last(self) -> Column:
        """last value of each list scalar"""
        return self.map(ListChunk.last)

    @doc_field
    def min(self) -> Column:
        """min value of each list scalar"""
        return self.map(ListChunk.min)

    @doc_field
    def max(self) -> Column:
        """max value of each list scalar"""
        return self.map(ListChunk.max)

    @doc_field
    def sum(self) -> Column:
        """sum of each list scalar"""
        return self.map(ListChunk.sum)

    @doc_field
    def mean(self) -> FloatColumn:
        """mean of each list scalar"""
        return self.map(ListChunk.mean)  # type: ignore

    @doc_field
    def mode(self, length: int = 1) -> 'ListColumn':
        """mode of each list scalar"""
        return self.map(functools.partial(ListChunk.mode, length=length))  # type: ignore

    @doc_field
    def quantile(self, q: List[float] = [0.5]) -> 'ListColumn':
        """quantile of each list scalar"""
        return self.map(functools.partial(ListChunk.quantile, q=q))  # type: ignore

    @doc_field
    def stddev(self) -> FloatColumn:
        """stddev of each list scalar"""
        return self.map(ListChunk.stddev)  # type: ignore

    @doc_field
    def variance(self) -> FloatColumn:
        """variance of each list scalar"""
        return self.map(ListChunk.variance)  # type: ignore

    @doc_field
    def any(self) -> BooleanColumn:
        """any true of each list scalar"""
        return self.map(ListChunk.any)  # type: ignore

    @doc_field
    def all(self) -> BooleanColumn:
        """all true of each list scalar"""
        return self.map(ListChunk.all)  # type: ignore

    @doc_field
    def binary_join(self, separator: bytes) -> BinaryColumn:
        """Join a list of binary strings together with a `separator` to form a single string."""
        return BinaryColumn(pc.binary_join(self.array, separator))

    @doc_field
    def string_join(self, separator: str) -> StringColumn:
        """Join a list of strings together with a `separator` to form a single string."""
        return StringColumn(pc.binary_join(self.array, separator))


@strawberry.type(description="column of structs")
class StructColumn(Column):
    __init__ = Column.__init__  # type: ignore

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
    bytes: BinaryColumn,
    str: StringColumn,
    list: ListColumn,
    dict: StructColumn,
}
