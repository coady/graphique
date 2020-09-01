"""
GraphQL output types and resolvers.
"""
import types
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import List, Optional
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import strawberry
from strawberry.types.type_resolver import resolve_type
from strawberry.types.types import ArgumentDefinition, FieldDefinition, undefined
from strawberry.utils.str_converters import to_camel_case
from .core import Column as C
from .inputs import (
    BooleanQuery,
    IntQuery,
    LongQuery,
    FloatQuery,
    DecimalQuery,
    DateQuery,
    DateTimeQuery,
    TimeQuery,
    DurationQuery,
    BinaryQuery,
    StringFilter,
)
from .scalars import Long


def selections(node):
    """Return tree of field name selections."""
    nodes = getattr(node.selection_set, 'selections', [])
    return {node.name.value for node in nodes if hasattr(node, 'name')}


def doc_field(func):
    return strawberry.field(func, description=func.__doc__)


@strawberry.interface(description="column interface")
class Column:
    @doc_field
    def type(self) -> str:
        """array type"""
        return str(self.array.type)  # type: ignore


@strawberry.interface(description="unique values")
class Set:
    counts: List[Long]

    @doc_field
    def length(self) -> Long:
        """number of rows"""
        return len(self.array)  # type: ignore


class resolvers:
    def __init__(self, array, **attrs):
        self.array = array
        self.__dict__.update(attrs)

    def unique(self, info):
        """unique values and counts"""
        if 'counts' in selections(*info.field_nodes):
            values, counts = C.value_counts(self.array).flatten()
        else:
            values, counts = C.unique(self.array), pa.array([])
        return self.Set(values, counts=counts.to_pylist())

    def count(self, **query) -> Long:
        """Return number of matching values.
        Optimized for `null`, and empty queries will attempt boolean conversion."""
        if query == {'equal': None}:
            return self.array.null_count
        if query == {'not_equal': None}:
            return len(self.array) - self.array.null_count
        query = {
            key: (value.asdict() if hasattr(value, 'asdict') else value)
            for key, value in query.items()
        }
        return C.count(C.mask(self.array, **query), True)  # type: ignore

    def values(self):
        """list of values"""
        return self.array.to_pylist()

    def sum(self, exp: int = 1):
        """Return sum of the values, with optional exponentiation."""
        return C.sum(self.array, exp)

    @doc_field
    def mean(self) -> Optional[float]:
        """mean of the values"""
        return pc.call_function('mean', [self.array]).as_py()

    def min(self):
        """minimum value"""
        return C.min(self.array)

    def max(self):
        """maximum value"""
        return C.max(self.array)

    @doc_field
    def quantile(self, q: List[float]) -> List[float]:
        """Return q-th quantiles for values."""
        return np.nanquantile(self.array, q).tolist()

    def sort(self, reverse: bool = False, length: Optional[Long] = None):
        """Return sorted values. Optimized for fixed length."""
        return C.sort(self.array, reverse, length).to_pylist()

    @doc_field
    def binary_length(self) -> 'IntColumn':
        """length of bytes or strings"""
        return IntColumn(pc.binary_length(self.array))

    def fill_null(self, value):
        """Return values with null elements replaced."""
        return type(self)(self.array.fill_null(value))

    def add(self, value):
        """Return values added to scalar."""
        return type(self)(pc.add(pa.scalar(value, self.array.type), self.array))

    def subtract(self, value):
        """Return values subtracted *from* scalar."""
        return type(self)(pc.subtract(pa.scalar(value, self.array.type), self.array))

    def multiply(self, value):
        """Return values multiplied by scalar."""
        return type(self)(pc.multiply(pa.scalar(value, self.array.type), self.array))

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
    clone = types.FunctionType(func.__code__, func.__globals__)
    annotations['return'] = return_type
    clone.__annotations__.update(func.__annotations__, **annotations)
    clone.__defaults__ = func.__defaults__
    return strawberry.field(clone, description=func.__doc__)


def resolve_arguments(func, arguments):
    for argument in arguments:
        argument.origin = func
        argument.name = to_camel_case(argument.origin_name)
        resolve_type(argument)
    func._field_definition = FieldDefinition(
        name=to_camel_case(func.__name__),
        origin_name=func.__name__,
        type=func.__annotations__['return'],
        origin=func,
        arguments=arguments,
        description=func.__doc__,
        base_resolver=func,
    )
    return func


def query_args(func, query):
    clone = types.FunctionType(func.__code__, func.__globals__)
    clone.__annotations__.update(func.__annotations__)
    arguments = [
        ArgumentDefinition(
            origin_name=name, type=value, default_value=getattr(query, name, undefined)
        )
        for name, value in query.__annotations__.items()
        if name != 'apply'
    ]
    return resolve_arguments(clone, arguments)


@strawberry.type(description="column of booleans")
class BooleanColumn(Column):
    __init__ = resolvers.__init__  # type: ignore
    count = query_args(resolvers.count, BooleanQuery)
    values = annotate(resolvers.values, List[Optional[bool]])


@strawberry.type(description="unique ints")
class IntSet(Set):
    __init__ = resolvers.__init__  # type: ignore
    values = annotate(resolvers.values, List[Optional[int]])


@strawberry.type(description="column of ints")
class IntColumn(Column):
    Set = IntSet
    __init__ = resolvers.__init__  # type: ignore
    count = query_args(resolvers.count, IntQuery)
    values = annotate(resolvers.values, List[Optional[int]])
    sort = annotate(resolvers.sort, List[Optional[int]])
    sum = annotate(resolvers.sum, Optional[int])
    mean = resolvers.mean
    min = annotate(resolvers.min, Optional[int])
    max = annotate(resolvers.max, Optional[int])
    quantile = resolvers.quantile
    unique = annotate(resolvers.unique, Set)
    fill_null = annotate(resolvers.fill_null, 'IntColumn', value=int)
    add = annotate(resolvers.add, 'IntColumn', value=int)
    subtract = annotate(resolvers.subtract, 'IntColumn', value=int)
    multiply = annotate(resolvers.multiply, 'IntColumn', value=int)
    minimum = annotate(resolvers.minimum, 'IntColumn', value=int)
    maximum = annotate(resolvers.maximum, 'IntColumn', value=int)
    absolute = annotate(resolvers.absolute, 'IntColumn')


@strawberry.type(description="unique longs")
class LongSet(Set):
    __init__ = resolvers.__init__  # type: ignore
    values = annotate(resolvers.values, List[Optional[Long]])


@strawberry.type(description="column of longs")
class LongColumn(Column):
    Set = LongSet
    __init__ = resolvers.__init__  # type: ignore
    count = query_args(resolvers.count, LongQuery)
    values = annotate(resolvers.values, List[Optional[Long]])
    sort = annotate(resolvers.sort, List[Optional[Long]])
    sum = annotate(resolvers.sum, Optional[Long])
    mean = resolvers.mean
    min = annotate(resolvers.min, Optional[Long])
    max = annotate(resolvers.max, Optional[Long])
    quantile = resolvers.quantile
    unique = annotate(resolvers.unique, Set)
    fill_null = annotate(resolvers.fill_null, 'LongColumn', value=Long)
    add = annotate(resolvers.add, 'LongColumn', value=Long)
    subtract = annotate(resolvers.subtract, 'LongColumn', value=Long)
    multiply = annotate(resolvers.multiply, 'LongColumn', value=Long)
    minimum = annotate(resolvers.minimum, 'LongColumn', value=Long)
    maximum = annotate(resolvers.maximum, 'LongColumn', value=Long)
    absolute = annotate(resolvers.absolute, 'LongColumn')


@strawberry.type(description="column of floats")
class FloatColumn(Column):
    __init__ = resolvers.__init__  # type: ignore
    count = query_args(resolvers.count, FloatQuery)
    values = annotate(resolvers.values, List[Optional[float]])
    sort = annotate(resolvers.sort, List[Optional[float]])
    sum = annotate(resolvers.sum, Optional[float])
    mean = resolvers.mean
    min = annotate(resolvers.min, Optional[float])
    max = annotate(resolvers.max, Optional[float])
    quantile = resolvers.quantile
    fill_null = annotate(resolvers.fill_null, 'FloatColumn', value=float)
    add = annotate(resolvers.add, 'FloatColumn', value=float)
    subtract = annotate(resolvers.subtract, 'FloatColumn', value=float)
    multiply = annotate(resolvers.multiply, 'FloatColumn', value=float)
    minimum = annotate(resolvers.minimum, 'FloatColumn', value=float)
    maximum = annotate(resolvers.maximum, 'FloatColumn', value=float)
    absolute = annotate(resolvers.absolute, 'FloatColumn')


@strawberry.type(description="column of decimals")
class DecimalColumn(Column):
    __init__ = resolvers.__init__  # type: ignore
    count = query_args(resolvers.count, DecimalQuery)
    values = annotate(resolvers.values, List[Optional[Decimal]])
    min = annotate(resolvers.min, Optional[Decimal])
    max = annotate(resolvers.max, Optional[Decimal])
    minimum = annotate(resolvers.minimum, 'DecimalColumn', value=Decimal)
    maximum = annotate(resolvers.maximum, 'DecimalColumn', value=Decimal)


@strawberry.type(description="unique dates")
class DateSet(Set):
    __init__ = resolvers.__init__  # type: ignore
    values = annotate(resolvers.values, List[Optional[date]])


@strawberry.type(description="column of dates")
class DateColumn(Column):
    Set = DateSet
    __init__ = resolvers.__init__  # type: ignore
    count = query_args(resolvers.count, DateQuery)
    values = annotate(resolvers.values, List[Optional[date]])
    min = annotate(resolvers.min, Optional[date])
    max = annotate(resolvers.max, Optional[date])
    unique = annotate(resolvers.unique, Set)
    fill_null = annotate(resolvers.fill_null, 'DateColumn', value=date)
    minimum = annotate(resolvers.minimum, 'DateColumn', value=date)
    maximum = annotate(resolvers.maximum, 'DateColumn', value=date)


@strawberry.type(description="column of datetimes")
class DateTimeColumn(Column):
    __init__ = resolvers.__init__  # type: ignore
    count = query_args(resolvers.count, DateTimeQuery)
    values = annotate(resolvers.values, List[Optional[datetime]])
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
    count = query_args(resolvers.count, TimeQuery)
    values = annotate(resolvers.values, List[Optional[time]])
    min = annotate(resolvers.min, Optional[time])
    max = annotate(resolvers.max, Optional[time])
    fill_null = annotate(resolvers.fill_null, 'TimeColumn', value=time)
    minimum = annotate(resolvers.minimum, 'TimeColumn', value=time)
    maximum = annotate(resolvers.maximum, 'TimeColumn', value=time)


@strawberry.type(description="column of durations")
class DurationColumn(Column):
    __init__ = resolvers.__init__  # type: ignore
    count = query_args(resolvers.count, DurationQuery)
    values = annotate(resolvers.values, List[Optional[timedelta]])
    min = annotate(resolvers.min, Optional[timedelta])
    max = annotate(resolvers.max, Optional[timedelta])
    quantile = annotate(resolvers.quantile, 'DurationColumn')
    minimum = annotate(resolvers.minimum, 'DurationColumn', value=timedelta)
    maximum = annotate(resolvers.maximum, 'DurationColumn', value=timedelta)
    absolute = annotate(resolvers.absolute, 'DurationColumn')


@strawberry.type(description="column of binaries")
class BinaryColumn(Column):
    __init__ = resolvers.__init__  # type: ignore
    count = query_args(resolvers.count, BinaryQuery)
    values = annotate(resolvers.values, List[Optional[bytes]])
    binary_length = resolvers.binary_length


@strawberry.type(description="unique strings")
class StringSet(Set):
    __init__ = resolvers.__init__  # type: ignore
    values = annotate(resolvers.values, List[Optional[str]])


@strawberry.type(description="column of strings")
class StringColumn(Column):
    Set = StringSet
    __init__ = resolvers.__init__  # type: ignore
    count = query_args(resolvers.count, StringFilter)
    values = annotate(resolvers.values, List[Optional[str]])
    sort = annotate(resolvers.sort, List[Optional[str]])
    min = annotate(resolvers.min, Optional[str])
    max = annotate(resolvers.max, Optional[str])
    unique = annotate(resolvers.unique, Set)
    binary_length = resolvers.binary_length
    minimum = annotate(resolvers.minimum, 'StringColumn', value=str)
    maximum = annotate(resolvers.maximum, 'StringColumn', value=str)

    @doc_field
    def utf8_lower(self) -> 'StringColumn':
        """strings converted to lowercase"""
        return StringColumn(pc.utf8_lower(self.array))  # type: ignore

    @doc_field
    def utf8_upper(self) -> 'StringColumn':
        """strings converted to uppercase"""
        return StringColumn(pc.utf8_upper(self.array))  # type: ignore


column_map = {
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
}
