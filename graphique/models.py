import base64
import types
from datetime import date, datetime, time
from decimal import Decimal
from typing import List, NewType, Optional
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import strawberry
from strawberry.types.type_resolver import resolve_type
from strawberry.types.types import ArgumentDefinition, FieldDefinition, undefined
from strawberry.utils.str_converters import to_camel_case
from .core import Column as C


Long = NewType('Long', int)
strawberry.scalar(Long, description="64-bit int")
strawberry.scalar(
    bytes,
    name='Binary',
    description="base64 encoded bytes",
    serialize=lambda b: base64.b64encode(b).decode('utf8'),
    parse_value=base64.b64decode,
)

type_map = {
    pa.lib.Type_BOOL: bool,
    pa.lib.Type_UINT8: int,
    pa.lib.Type_INT8: int,
    pa.lib.Type_UINT16: int,
    pa.lib.Type_INT16: int,
    pa.lib.Type_UINT32: Long,
    pa.lib.Type_INT32: int,
    pa.lib.Type_UINT64: Long,
    pa.lib.Type_INT64: Long,
    pa.lib.Type_FLOAT: float,
    pa.lib.Type_DOUBLE: float,
    pa.lib.Type_DECIMAL: Decimal,
    pa.lib.Type_DATE32: date,
    pa.lib.Type_DATE64: date,
    pa.lib.Type_TIMESTAMP: datetime,
    pa.lib.Type_TIME32: time,
    pa.lib.Type_TIME64: time,
    pa.lib.Type_BINARY: bytes,
    pa.lib.Type_STRING: str,
}
ops = 'equal', 'not_equal', 'less', 'less_equal', 'greater', 'greater_equal'


@strawberry.input(description="nominal predicates projected across two columns")
class Nominal:
    equal: Optional[str] = undefined
    not_equal: Optional[str] = undefined

    def asdict(self):
        return {name: value for name, value in self.__dict__.items() if value is not undefined}


@strawberry.input(description="ordinal predicates projected across two columns")
class Ordinal(Nominal):
    less: Optional[str] = undefined
    less_equal: Optional[str] = undefined
    greater: Optional[str] = undefined
    greater_equal: Optional[str] = undefined
    minimum: Optional[str] = undefined
    maximum: Optional[str] = undefined


@strawberry.input(description="ratio predicates projected across two columns")
class Ratio(Ordinal):
    add: Optional[str] = undefined
    subtract: Optional[str] = undefined
    multiply: Optional[str] = undefined


class Query:
    """base class for predicates"""

    locals().update(dict.fromkeys(ops, undefined))

    def asdict(self):
        return {
            name: (value.asdict() if hasattr(value, 'asdict') else value)
            for name, value in Nominal.asdict(self).items()
        }


@strawberry.input(description="predicates for booleans")
class BooleanQuery(Query):
    __annotations__ = dict.fromkeys(['equal', 'not_equal'], Optional[bool])


@strawberry.input(description="predicates for ints")
class IntQuery(Query):
    __annotations__ = dict.fromkeys(ops, Optional[int])
    is_in: Optional[List[int]] = undefined


@strawberry.input(description="predicates for longs")
class LongQuery(Query):
    __annotations__ = dict.fromkeys(ops, Optional[Long])
    is_in: Optional[List[Long]] = undefined


@strawberry.input(description="predicates for floats")
class FloatQuery(Query):
    __annotations__ = dict.fromkeys(ops, Optional[float])
    is_in: Optional[List[float]] = undefined


@strawberry.input(description="predicates for decimals")
class DecimalQuery(Query):
    __annotations__ = dict.fromkeys(ops, Optional[Decimal])
    is_in: Optional[List[Decimal]] = undefined


@strawberry.input(description="predicates for dates")
class DateQuery(Query):
    __annotations__ = dict.fromkeys(ops, Optional[date])
    is_in: Optional[List[date]] = undefined


@strawberry.input(description="predicates for datetimes")
class DateTimeQuery(Query):
    __annotations__ = dict.fromkeys(ops, Optional[datetime])
    is_in: Optional[List[datetime]] = undefined


@strawberry.input(description="predicates for times")
class TimeQuery(Query):
    __annotations__ = dict.fromkeys(ops, Optional[time])
    is_in: Optional[List[time]] = undefined


@strawberry.input(description="predicates for binaries")
class BinaryQuery(Query):
    __annotations__ = dict.fromkeys(['equal', 'not_equal'], Optional[bytes])
    is_in: Optional[List[bytes]] = undefined


@strawberry.input(description="predicates for strings")
class StringQuery(Query):
    __annotations__ = dict.fromkeys(ops, Optional[str])
    is_in: Optional[List[str]] = undefined


query_map = {
    bool: BooleanQuery,
    int: IntQuery,
    Long: LongQuery,
    float: FloatQuery,
    Decimal: DecimalQuery,
    date: DateQuery,
    datetime: DateTimeQuery,
    time: TimeQuery,
    bytes: BinaryQuery,
    str: StringQuery,
}


@strawberry.input(description="predicates for booleans")
class BooleanFilter(BooleanQuery):
    project: Optional[Nominal] = undefined


@strawberry.input(description="predicates for ints")
class IntFilter(IntQuery):
    project: Optional[Ratio] = undefined


@strawberry.input(description="predicates for longs")
class LongFilter(LongQuery):
    project: Optional[Ratio] = undefined


@strawberry.input(description="predicates for floats")
class FloatFilter(FloatQuery):
    project: Optional[Ratio] = undefined


@strawberry.input(description="predicates for decimals")
class DecimalFilter(DecimalQuery):
    project: Optional[Ordinal] = undefined


@strawberry.input(description="predicates for dates")
class DateFilter(DateQuery):
    project: Optional[Ordinal] = undefined


@strawberry.input(description="predicates for datetimes")
class DateTimeFilter(DateTimeQuery):
    project: Optional[Ordinal] = undefined


@strawberry.input(description="predicates for times")
class TimeFilter(TimeQuery):
    project: Optional[Ordinal] = undefined


@strawberry.input(description="predicates for binaries")
class BinaryFilter(BinaryQuery):
    project: Optional[Nominal] = undefined


@strawberry.input(description="predicates for strings")
class StringFilter(StringQuery):
    __annotations__ = dict(StringQuery.__annotations__)  # used for `count` interface
    match_substring: Optional[str] = undefined
    binary_length: Optional[IntQuery] = undefined
    utf8_lower: Optional['StringFilter'] = undefined
    utf8_upper: Optional['StringFilter'] = undefined
    string_is_ascii: bool = False
    utf8_is_alnum: bool = False
    utf8_is_alpha: bool = False
    utf8_is_digit: bool = False
    utf8_is_lower: bool = False
    utf8_is_title: bool = False
    utf8_is_upper: bool = False
    project: Optional[Ordinal] = undefined


filter_map = {
    bool: BooleanFilter,
    int: IntFilter,
    Long: LongFilter,
    float: FloatFilter,
    Decimal: DecimalFilter,
    date: DateFilter,
    datetime: DateTimeFilter,
    time: TimeFilter,
    bytes: BinaryFilter,
    str: StringFilter,
}


def selections(node):
    """Return tree of field name selections."""
    nodes = getattr(node.selection_set, 'selections', [])
    return {node.name.value: selections(node) for node in nodes}


def doc_field(func):
    return strawberry.field(func, description=func.__doc__)


class resolvers:
    array: pa.ChunkedArray

    def __init__(self, array, **attrs):
        self.array = array
        self.__dict__.update(attrs)

    @doc_field
    def length(self) -> Long:
        """number of rows"""
        return len(self.array)  # type: ignore

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
        if name != 'project'
    ]
    return resolve_arguments(clone, arguments)


@strawberry.type(description="unique booleans")
class BooleanSet:
    counts: List[Long]
    __init__ = resolvers.__init__  # type: ignore
    length = resolvers.length
    values = annotate(resolvers.values, List[Optional[bool]])


@strawberry.type(description="column of booleans")
class BooleanColumn:
    Set = BooleanSet
    __init__ = resolvers.__init__  # type: ignore
    count = query_args(resolvers.count, BooleanQuery)
    values = annotate(resolvers.values, List[Optional[bool]])
    unique = annotate(resolvers.unique, Set)


@strawberry.type(description="unique ints")
class IntSet:
    counts: List[Long]
    __init__ = resolvers.__init__  # type: ignore
    length = resolvers.length
    values = annotate(resolvers.values, List[Optional[int]])


@strawberry.type(description="column of ints")
class IntColumn:
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


@strawberry.type(description="unique longs")
class LongSet:
    counts: List[Long]
    __init__ = resolvers.__init__  # type: ignore
    length = resolvers.length
    values = annotate(resolvers.values, List[Optional[Long]])


@strawberry.type(description="column of longs")
class LongColumn:
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


@strawberry.type(description="column of floats")
class FloatColumn:
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


@strawberry.type(description="column of decimals")
class DecimalColumn:
    __init__ = resolvers.__init__  # type: ignore
    count = query_args(resolvers.count, DecimalQuery)
    values = annotate(resolvers.values, List[Optional[Decimal]])
    sort = annotate(resolvers.sort, List[Optional[Decimal]])
    min = annotate(resolvers.min, Optional[Decimal])
    max = annotate(resolvers.max, Optional[Decimal])
    minimum = annotate(resolvers.minimum, 'DecimalColumn', value=Decimal)
    maximum = annotate(resolvers.maximum, 'DecimalColumn', value=Decimal)


@strawberry.type(description="unique dates")
class DateSet:
    counts: List[Long]
    __init__ = resolvers.__init__  # type: ignore
    length = resolvers.length
    values = annotate(resolvers.values, List[Optional[date]])


@strawberry.type(description="column of dates")
class DateColumn:
    Set = DateSet
    __init__ = resolvers.__init__  # type: ignore
    count = query_args(resolvers.count, DateQuery)
    values = annotate(resolvers.values, List[Optional[date]])
    sort = annotate(resolvers.sort, List[Optional[date]])
    min = annotate(resolvers.min, Optional[date])
    max = annotate(resolvers.max, Optional[date])
    unique = annotate(resolvers.unique, Set)
    fill_null = annotate(resolvers.fill_null, 'DateColumn', value=date)
    minimum = annotate(resolvers.minimum, 'DateColumn', value=date)
    maximum = annotate(resolvers.maximum, 'DateColumn', value=date)


@strawberry.type(description="column of datetimes")
class DateTimeColumn:
    __init__ = resolvers.__init__  # type: ignore
    count = query_args(resolvers.count, DateTimeQuery)
    values = annotate(resolvers.values, List[Optional[datetime]])
    sort = annotate(resolvers.sort, List[Optional[datetime]])
    min = annotate(resolvers.min, Optional[datetime])
    max = annotate(resolvers.max, Optional[datetime])
    fill_null = annotate(resolvers.fill_null, 'DateTimeColumn', value=datetime)
    minimum = annotate(resolvers.minimum, 'DateTimeColumn', value=datetime)
    maximum = annotate(resolvers.maximum, 'DateTimeColumn', value=datetime)


@strawberry.type(description="column of times")
class TimeColumn:
    __init__ = resolvers.__init__  # type: ignore
    count = query_args(resolvers.count, TimeQuery)
    values = annotate(resolvers.values, List[Optional[time]])
    sort = annotate(resolvers.sort, List[Optional[time]])
    min = annotate(resolvers.min, Optional[time])
    max = annotate(resolvers.max, Optional[time])
    fill_null = annotate(resolvers.fill_null, 'TimeColumn', value=time)
    minimum = annotate(resolvers.minimum, 'TimeColumn', value=time)
    maximum = annotate(resolvers.maximum, 'TimeColumn', value=time)


@strawberry.type(description="column of binaries")
class BinaryColumn:
    __init__ = resolvers.__init__  # type: ignore
    count = query_args(resolvers.count, BinaryQuery)
    values = annotate(resolvers.values, List[Optional[bytes]])
    binary_length = resolvers.binary_length


@strawberry.type(description="unique strings")
class StringSet:
    counts: List[Long]
    __init__ = resolvers.__init__  # type: ignore
    length = resolvers.length
    values = annotate(resolvers.values, List[Optional[str]])


@strawberry.type(description="column of strings")
class StringColumn:
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
    bytes: BinaryColumn,
    str: StringColumn,
}
