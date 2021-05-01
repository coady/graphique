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


def selections(node):
    """Return tree of field name selections."""
    nodes = getattr(node.selection_set, 'selections', [])
    return {node.name.value for node in nodes if hasattr(node, 'name')}


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
    @doc_field
    def type(self) -> str:
        """array type"""
        return str(self.array.type)  # type: ignore

    @doc_field
    def length(self) -> Long:
        """number of rows"""
        return len(self.array)  # type: ignore

    @classmethod
    def cast(cls, array: pa.ChunkedArray) -> 'Column':
        """Return typed column based on array type."""
        return cls.type_map[type_map[C.scalar_type(array).id]](array)  # type: ignore

    @classmethod
    def fromlist(cls, scalar: pa.ListScalar) -> Optional['Column']:
        array = scalar.values
        return None if array is None else cls.cast(pa.chunked_array([array]))


@strawberry.interface(description="unique values")
class Set:
    length = doc_field(Column.length)

    @doc_field
    def counts(self) -> List[Long]:
        """list of counts"""
        return self.count.to_pylist()  # type: ignore

    @classmethod
    def subclass(base, cls, name, description):
        namespace = {
            '__init__': resolvers.__init__,
            'values': annotate(resolvers.values, List[Optional[cls]]),
        }
        return strawberry.type(description=description)(type(name, (base,), namespace))


class resolvers:
    def __init__(self, array, count=pa.array([])):
        self.array, self.count = array, count

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
        return C.count(C.mask(self.array, **query), True)  # type: ignore

    def values(self):
        """list of values"""
        return self.array.to_pylist()

    def mode(self, length: int = 1):
        """mode of the values"""
        return self.Set(*pc.mode(self.array, length).flatten())  # type: ignore

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

    def minimum(self, value):
        """Return element-wise minimum compared to scalar."""
        return type(self)(C.minimum(self.array, value))

    def maximum(self, value):
        """Return element-wise maximum compared to scalar."""
        return type(self)(C.maximum(self.array, value))

    def absolute(self):
        """absolute values"""
        return type(self)(C.absolute(self.array))


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
        return type(self)(pc.add(pa.scalar(value, self.array.type), self.array))

    def subtract(self, value):
        """Return values subtracted *from* scalar."""
        return type(self)(pc.subtract(pa.scalar(value, self.array.type), self.array))

    def multiply(self, value):
        """Return values multiplied by scalar."""
        return type(self)(pc.multiply(pa.scalar(value, self.array.type), self.array))

    def divide(self, value):
        """Return values divided *into* scalar."""
        return type(self)(pc.divide(pa.scalar(value, self.array.type), self.array))

    def power(self, base=None, exponent=None):
        """Return values raised to power."""
        assert [base, exponent].count(None) == 1, "exactly one of `base` or `exponent` required"
        if base is None:
            return type(self)(pc.power(self.array, pa.scalar(exponent, self.array.type)))
        return type(self)(pc.power(pa.scalar(base, self.array.type), self.array))


@strawberry.type(description="column of booleans")
class BooleanColumn(Column):
    __init__ = resolvers.__init__  # type: ignore
    count = BooleanQuery.resolver(resolvers.count)
    any = doc_field(NumericColumn.any)
    all = doc_field(NumericColumn.all)
    values = annotate(resolvers.values, List[Optional[bool]])
    Set = Set.subclass(bool, "BooleanSet", "unique booleans")
    unique = annotate(resolvers.unique, Set)


@strawberry.type(description="column of ints")
class IntColumn(Column, NumericColumn):
    __init__ = resolvers.__init__  # type: ignore
    count = IntQuery.resolver(resolvers.count)
    values = annotate(resolvers.values, List[Optional[int]])
    Set = Set.subclass(int, "IntSet", "unique ints")
    unique = annotate(resolvers.unique, Set)
    sort = annotate(resolvers.sort, List[Optional[int]])
    sum = annotate(NumericColumn.sum, Optional[int])
    mode = annotate(resolvers.mode, Set)
    min = annotate(NumericColumn.min, Optional[int])
    max = annotate(NumericColumn.max, Optional[int])
    unique = annotate(resolvers.unique, Set)
    fill_null = annotate(resolvers.fill_null, 'IntColumn', value=int)
    add = annotate(NumericColumn.add, 'IntColumn', value=int)
    subtract = annotate(NumericColumn.subtract, 'IntColumn', value=int)
    multiply = annotate(NumericColumn.multiply, 'IntColumn', value=int)
    divide = annotate(NumericColumn.divide, 'IntColumn', value=int)
    power = annotate(NumericColumn.power, 'IntColumn', base=Optional[int], exponent=Optional[int])
    minimum = annotate(resolvers.minimum, 'IntColumn', value=int)
    maximum = annotate(resolvers.maximum, 'IntColumn', value=int)
    absolute = annotate(resolvers.absolute, 'IntColumn')


@strawberry.type(description="column of longs")
class LongColumn(Column, NumericColumn):
    __init__ = resolvers.__init__  # type: ignore
    count = LongQuery.resolver(resolvers.count)
    values = annotate(resolvers.values, List[Optional[Long]])
    Set = Set.subclass(Long, "LongSet", "unique longs")
    unique = annotate(resolvers.unique, Set)
    sort = annotate(resolvers.sort, List[Optional[Long]])
    sum = annotate(NumericColumn.sum, Optional[Long])
    mode = annotate(resolvers.mode, Set)
    min = annotate(NumericColumn.min, Optional[Long])
    max = annotate(NumericColumn.max, Optional[Long])
    unique = annotate(resolvers.unique, Set)
    fill_null = annotate(resolvers.fill_null, 'LongColumn', value=Long)
    add = annotate(NumericColumn.add, 'LongColumn', value=Long)
    subtract = annotate(NumericColumn.subtract, 'LongColumn', value=Long)
    multiply = annotate(NumericColumn.multiply, 'LongColumn', value=Long)
    divide = annotate(NumericColumn.divide, 'LongColumn', value=Long)
    power = annotate(
        NumericColumn.power, 'LongColumn', base=Optional[Long], exponent=Optional[Long]
    )
    minimum = annotate(resolvers.minimum, 'LongColumn', value=Long)
    maximum = annotate(resolvers.maximum, 'LongColumn', value=Long)
    absolute = annotate(resolvers.absolute, 'LongColumn')


@strawberry.type(description="column of floats")
class FloatColumn(Column, NumericColumn):
    __init__ = resolvers.__init__  # type: ignore
    count = FloatQuery.resolver(resolvers.count)
    values = annotate(resolvers.values, List[Optional[float]])
    Set = Set.subclass(float, "FloatSet", "unique floats")
    unique = annotate(resolvers.unique, Set)
    sort = annotate(resolvers.sort, List[Optional[float]])
    sum = annotate(NumericColumn.sum, Optional[float])
    mode = annotate(resolvers.mode, Set)
    min = annotate(NumericColumn.min, Optional[float])
    max = annotate(NumericColumn.max, Optional[float])
    fill_null = annotate(resolvers.fill_null, 'FloatColumn', value=float)
    add = annotate(NumericColumn.add, 'FloatColumn', value=float)
    subtract = annotate(NumericColumn.subtract, 'FloatColumn', value=float)
    multiply = annotate(NumericColumn.multiply, 'FloatColumn', value=float)
    divide = annotate(NumericColumn.divide, 'FloatColumn', value=float)
    power = annotate(
        NumericColumn.power, 'FloatColumn', base=Optional[float], exponent=Optional[float]
    )
    minimum = annotate(resolvers.minimum, 'FloatColumn', value=float)
    maximum = annotate(resolvers.maximum, 'FloatColumn', value=float)
    absolute = annotate(resolvers.absolute, 'FloatColumn')


@strawberry.type(description="column of decimals")
class DecimalColumn(Column):
    __init__ = resolvers.__init__  # type: ignore
    count = DecimalQuery.resolver(resolvers.count)
    values = annotate(resolvers.values, List[Optional[Decimal]])
    Set = Set.subclass(Decimal, "DecimalSet", "unique decimals")
    unique = annotate(resolvers.unique, Set)
    min = annotate(resolvers.min, Optional[Decimal])
    max = annotate(resolvers.max, Optional[Decimal])
    minimum = annotate(resolvers.minimum, 'DecimalColumn', value=Decimal)
    maximum = annotate(resolvers.maximum, 'DecimalColumn', value=Decimal)


@strawberry.type(description="column of dates")
class DateColumn(Column):
    __init__ = resolvers.__init__  # type: ignore
    count = DateQuery.resolver(resolvers.count)
    values = annotate(resolvers.values, List[Optional[date]])
    Set = Set.subclass(date, "DateSet", "unique dates")
    unique = annotate(resolvers.unique, Set)
    min = annotate(resolvers.min, Optional[date])
    max = annotate(resolvers.max, Optional[date])
    unique = annotate(resolvers.unique, Set)
    fill_null = annotate(resolvers.fill_null, 'DateColumn', value=date)
    minimum = annotate(resolvers.minimum, 'DateColumn', value=date)
    maximum = annotate(resolvers.maximum, 'DateColumn', value=date)


@strawberry.type(description="column of datetimes")
class DateTimeColumn(Column):
    __init__ = resolvers.__init__  # type: ignore
    count = DateTimeQuery.resolver(resolvers.count)
    values = annotate(resolvers.values, List[Optional[datetime]])
    Set = Set.subclass(datetime, "DatetimeSet", "unique datetimes")
    unique = annotate(resolvers.unique, Set)
    min = annotate(resolvers.min, Optional[datetime])
    max = annotate(resolvers.max, Optional[datetime])
    fill_null = annotate(resolvers.fill_null, 'DateTimeColumn', value=datetime)
    minimum = annotate(resolvers.minimum, 'DateTimeColumn', value=datetime)
    maximum = annotate(resolvers.maximum, 'DateTimeColumn', value=datetime)

    @doc_field
    def subtract(self, value: datetime) -> 'DurationColumn':
        """Return values subtracted *from* scalar."""
        return DurationColumn(pc.subtract(pa.scalar(value, self.array.type), self.array))  # type: ignore


@strawberry.type(description="column of times")
class TimeColumn(Column):
    __init__ = resolvers.__init__  # type: ignore
    count = TimeQuery.resolver(resolvers.count)
    values = annotate(resolvers.values, List[Optional[time]])
    Set = Set.subclass(time, "TimeSet", "unique times")
    unique = annotate(resolvers.unique, Set)
    min = annotate(resolvers.min, Optional[time])
    max = annotate(resolvers.max, Optional[time])
    fill_null = annotate(resolvers.fill_null, 'TimeColumn', value=time)
    minimum = annotate(resolvers.minimum, 'TimeColumn', value=time)
    maximum = annotate(resolvers.maximum, 'TimeColumn', value=time)


@strawberry.type(description="column of durations")
class DurationColumn(Column):
    __init__ = resolvers.__init__  # type: ignore
    count = DurationQuery.resolver(resolvers.count)
    values = annotate(resolvers.values, List[Optional[timedelta]])
    minimum = annotate(resolvers.minimum, 'DurationColumn', value=timedelta)
    maximum = annotate(resolvers.maximum, 'DurationColumn', value=timedelta)
    absolute = annotate(resolvers.absolute, 'DurationColumn')


@strawberry.type(description="column of binaries")
class BinaryColumn(Column):
    __init__ = resolvers.__init__  # type: ignore
    count = BinaryQuery.resolver(resolvers.count)
    any = doc_field(NumericColumn.any)
    all = doc_field(NumericColumn.all)
    values = annotate(resolvers.values, List[Optional[bytes]])
    Set = Set.subclass(bytes, "BinarySet", "unique binaries")
    unique = annotate(resolvers.unique, Set)
    fill_null = annotate(resolvers.fill_null, 'BinaryColumn', value=bytes)

    @doc_field
    def binary_length(self) -> 'IntColumn':
        """length of bytes or strings"""
        return IntColumn(pc.binary_length(self.array))  # type: ignore


@strawberry.type(description="column of strings")
class StringColumn(Column):
    __init__ = resolvers.__init__  # type: ignore
    count = StringFilter.resolver(resolvers.count)
    any = doc_field(NumericColumn.any)
    all = doc_field(NumericColumn.all)
    values = annotate(resolvers.values, List[Optional[str]])
    Set = Set.subclass(str, "StringSet", "unique strings")
    unique = annotate(resolvers.unique, Set)
    sort = annotate(resolvers.sort, List[Optional[str]])
    min = annotate(resolvers.min, Optional[str])
    max = annotate(resolvers.max, Optional[str])
    fill_null = annotate(resolvers.fill_null, 'StringColumn', value=str)
    minimum = annotate(resolvers.minimum, 'StringColumn', value=str)
    maximum = annotate(resolvers.maximum, 'StringColumn', value=str)

    @doc_field
    def utf8_length(self) -> 'IntColumn':
        """length of bytes or strings"""
        return IntColumn(pc.utf8_length(self.array))  # type: ignore

    @doc_field
    def utf8_lower(self) -> 'StringColumn':
        """strings converted to lowercase"""
        return StringColumn(pc.utf8_lower(self.array))  # type: ignore

    @doc_field
    def utf8_upper(self) -> 'StringColumn':
        """strings converted to uppercase"""
        return StringColumn(pc.utf8_upper(self.array))  # type: ignore

    @doc_field
    def split(self, pattern: str = '', max_splits: int = -1, reverse: bool = False) -> 'ListColumn':
        """Return strings split on pattern, by default whitespace."""
        kwargs = {'max_splits': max_splits, 'reverse': reverse}
        if pattern:
            return ListColumn(pc.split_pattern(self.array, pattern=pattern, **kwargs))  # type: ignore
        return ListColumn(pc.utf8_split_whitespace(self.array, **kwargs))  # type: ignore


@strawberry.type(description="column of lists")
class ListColumn(Column):
    __init__ = resolvers.__init__  # type: ignore
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
        defaults = dict.fromkeys(cls.aggregates, [])  # type: ignore
        return functools.partial(resolve_annotations, annotations=annotations, defaults=defaults)

    @doc_field
    def values(self) -> List[Optional[Column]]:
        """list of columns"""
        return list(map(self.fromlist, self.array))  # type: ignore

    def map(self, func: Callable) -> Column:
        return self.cast(pa.chunked_array(C.map(func, self.array)))  # type: ignore

    @doc_field
    def count(self) -> IntColumn:
        """number of values of each list scalar"""
        return self.map(pa.ListArray.value_lengths)  # type: ignore

    @doc_field
    def flatten(self) -> Column:
        """concatenation of all sub-lists"""
        return self.map(pa.ListArray.flatten)

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
        """sum each list scalar"""
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


@strawberry.type(description="column of structs")
class StructColumn(Column):
    __init__ = resolvers.__init__  # type: ignore

    @doc_field
    def names(self) -> List[str]:
        """field names"""
        return [field.name for field in self.array.type]  # type: ignore

    @doc_field
    def column(self, name: str) -> Column:
        """Return struct field as a column."""
        return ListColumn.map(self, operator.methodcaller('field', name))


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
