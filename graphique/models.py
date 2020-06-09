import base64
import copy
import decimal
from datetime import date, datetime, time
from typing import List, NewType, Optional
import numpy as np
import pyarrow as pa
import strawberry
from strawberry.utils.str_converters import to_snake_case
from .core import Column as C


Long = NewType('Long', int)
Decimal = NewType('Decimal', str)
strawberry.scalar(Long, description="64-bit int")
strawberry.scalar(
    Decimal, description="fixed point decimal", serialize=str, parse_value=decimal.Decimal,
)
strawberry.scalar(
    datetime,
    name='Timestamp',
    description="datetime (isoformat)",
    serialize=datetime.isoformat,
    parse_value=datetime.fromisoformat,
)
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


@strawberry.input
class BooleanQuery:
    """predicates for booleans"""

    __annotations__ = dict.fromkeys(['equal', 'not_equal'], Optional[bool])


@strawberry.input
class IntQuery:
    """predicates for ints"""

    __annotations__ = dict.fromkeys(ops, Optional[int])
    isin: Optional[List[int]]


@strawberry.input
class LongQuery:
    """predicates for longs"""

    __annotations__ = dict.fromkeys(ops, Optional[Long])
    isin: Optional[List[Long]]


@strawberry.input
class FloatQuery:
    """predicates for floats"""

    __annotations__ = dict.fromkeys(ops, Optional[float])
    isin: Optional[List[float]]


@strawberry.input
class DecimalQuery:
    """predicates for decimals"""

    __annotations__ = dict.fromkeys(ops, Optional[Decimal])
    isin: Optional[List[Decimal]]


@strawberry.input
class DateQuery:
    """predicates for dates"""

    __annotations__ = dict.fromkeys(ops, Optional[date])
    isin: Optional[List[date]]


@strawberry.input
class TimestampQuery:
    """predicates for timestamps"""

    __annotations__ = dict.fromkeys(ops, Optional[datetime])
    isin: Optional[List[datetime]]


@strawberry.input
class TimeQuery:
    """predicates for times"""

    __annotations__ = dict.fromkeys(ops, Optional[time])
    isin: Optional[List[time]]


@strawberry.input
class BinaryQuery:
    """predicates for binaries"""

    __annotations__ = dict.fromkeys(['equal', 'not_equal'], Optional[bytes])
    isin: Optional[List[bytes]]


@strawberry.input
class StringQuery:
    """predicates for strings"""

    __annotations__ = dict.fromkeys(ops, Optional[str])
    isin: Optional[List[str]]


query_map = {
    bool: BooleanQuery,
    int: IntQuery,
    Long: LongQuery,
    float: FloatQuery,
    Decimal: DecimalQuery,
    date: DateQuery,
    datetime: TimestampQuery,
    time: TimeQuery,
    bytes: BinaryQuery,
    str: StringQuery,
}


def selections(node):
    """Return tree of field name selections."""
    nodes = getattr(node.selection_set, 'selections', [])
    return {node.name.value: selections(node) for node in nodes}


class resolvers:
    array: pa.ChunkedArray

    def __init__(self, array, **attrs):
        self.array = array
        self.__dict__.update(attrs)

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

    def predicate(**query):
        return C.predicate(**{to_snake_case(op): query[op] for op in query})

    def count(self, **query) -> Long:
        """Return number of matching values.
Optimized for `null`, and empty queries are implicitly boolean."""
        if query == {'equal': None}:
            return self.array.null_count
        if query == {'notEqual': None}:
            return len(self.array) - self.array.null_count
        mask = C.mask(self.array, resolvers.predicate(**query)) if query else self.array
        return C.count(mask, True)  # type: ignore

    def any(self, **query) -> bool:
        """Return whether any value evaluates to `true`.
Optimized for `null`, and empty queries are implicitly boolean."""
        if query in ({'equal': None}, {'notEqual': None}):
            return bool(resolvers.count(self, **query))
        return C.any(self.array, resolvers.predicate(**query))

    def all(self, **query) -> bool:
        """Return whether all values evaluate to `true`.
Optimized for `null`, and empty queries are implicitly boolean."""
        if query in ({'equal': None}, {'notEqual': None}):
            (op,) = {'equal', 'notEqual'} - set(query)
            return not resolvers.count(self, **{op: None})
        return C.all(self.array, resolvers.predicate(**query))

    def item(self, index: Long = 0):  # type: ignore
        """Return scalar value at index."""
        return self.array[index].as_py()

    def values(self):
        """list of values"""
        return self.array.to_pylist()

    def sum(self, exp: int = 1):
        """Return sum of the values, with optional exponentiation."""
        return C.sum(self.array, exp)

    def min(self):
        """minimum value"""
        return C.min(self.array)

    def max(self):
        """maximum value"""
        return C.max(self.array)

    def quantile(self, q: List[float]) -> List[float]:
        """Return q-th quantiles for values."""
        return np.nanquantile(self.array, q).tolist()

    def sort(self, reverse: bool = False, length: Long = None):
        """Return sorted values. Optimized for fixed length."""
        return C.sort(self.array, reverse, length).to_pylist()


def annotate(func, return_type):
    func = copy.copy(func)
    func.__annotations__ = dict(func.__annotations__, **{'return': return_type})
    field = strawberry.field(func)
    field.graphql_type  # force evaluation
    return field


def query_args(func, query):
    field = strawberry.field(func)
    field.graphql_type.args.update(query.graphql_type.fields)
    return field


@strawberry.type
class BooleanSet:
    """unique booleans"""

    counts: List[Long]
    __init__ = resolvers.__init__  # type: ignore
    length = strawberry.field(resolvers.length)
    values = annotate(resolvers.values, List[Optional[bool]])


@strawberry.type
class BooleanColumn:
    """column of booleans"""

    Set = BooleanSet
    __init__ = resolvers.__init__  # type: ignore
    count = query_args(resolvers.count, BooleanQuery)
    any = query_args(resolvers.any, BooleanQuery)
    all = query_args(resolvers.all, BooleanQuery)
    item = annotate(resolvers.item, Optional[bool])
    values = annotate(resolvers.values, List[Optional[bool]])
    unique = annotate(resolvers.unique, Set)


@strawberry.type
class IntSet:
    """unique ints"""

    counts: List[Long]
    __init__ = resolvers.__init__  # type: ignore
    length = strawberry.field(resolvers.length)
    values = annotate(resolvers.values, List[Optional[int]])


@strawberry.type
class IntColumn:
    """column of ints"""

    Set = IntSet
    __init__ = resolvers.__init__  # type: ignore
    count = query_args(resolvers.count, IntQuery)
    any = query_args(resolvers.any, IntQuery)
    all = query_args(resolvers.all, IntQuery)
    item = annotate(resolvers.item, Optional[int])
    values = annotate(resolvers.values, List[Optional[int]])
    sort = annotate(resolvers.sort, List[Optional[int]])
    sum = annotate(resolvers.sum, int)
    min = annotate(resolvers.min, int)
    max = annotate(resolvers.max, int)
    quantile = strawberry.field(resolvers.quantile)
    unique = annotate(resolvers.unique, Set)


@strawberry.type
class LongSet:
    """unique longs"""

    counts: List[Long]
    __init__ = resolvers.__init__  # type: ignore
    length = strawberry.field(resolvers.length)
    values = annotate(resolvers.values, List[Optional[Long]])


@strawberry.type
class LongColumn:
    """column of longs"""

    Set = LongSet
    __init__ = resolvers.__init__  # type: ignore
    count = query_args(resolvers.count, LongQuery)
    any = query_args(resolvers.any, LongQuery)
    all = query_args(resolvers.all, LongQuery)
    item = annotate(resolvers.item, Optional[Long])
    values = annotate(resolvers.values, List[Optional[Long]])
    sort = annotate(resolvers.sort, List[Optional[Long]])
    sum = annotate(resolvers.sum, Long)
    min = annotate(resolvers.min, Long)
    max = annotate(resolvers.max, Long)
    quantile = strawberry.field(resolvers.quantile)
    unique = annotate(resolvers.unique, Set)


@strawberry.type
class FloatColumn:
    """column of floats"""

    __init__ = resolvers.__init__  # type: ignore
    count = query_args(resolvers.count, FloatQuery)
    any = query_args(resolvers.any, FloatQuery)
    all = query_args(resolvers.all, FloatQuery)
    item = annotate(resolvers.item, Optional[float])
    values = annotate(resolvers.values, List[Optional[float]])
    sort = annotate(resolvers.sort, List[Optional[float]])
    sum = annotate(resolvers.sum, float)
    min = annotate(resolvers.min, float)
    max = annotate(resolvers.max, float)
    quantile = strawberry.field(resolvers.quantile)


@strawberry.type
class DecimalColumn:
    """column of decimals"""

    __init__ = resolvers.__init__  # type: ignore
    count = query_args(resolvers.count, DecimalQuery)
    any = query_args(resolvers.any, DecimalQuery)
    all = query_args(resolvers.all, DecimalQuery)
    item = annotate(resolvers.item, Optional[Decimal])
    values = annotate(resolvers.values, List[Optional[Decimal]])
    sort = annotate(resolvers.sort, List[Optional[Decimal]])
    min = annotate(resolvers.min, Decimal)
    max = annotate(resolvers.max, Decimal)


@strawberry.type
class DateSet:
    """unique dates"""

    counts: List[Long]
    __init__ = resolvers.__init__  # type: ignore
    length = strawberry.field(resolvers.length)
    values = annotate(resolvers.values, List[Optional[date]])


@strawberry.type
class DateColumn:
    """column of dates"""

    Set = DateSet
    __init__ = resolvers.__init__  # type: ignore
    count = query_args(resolvers.count, DateQuery)
    any = query_args(resolvers.any, DateQuery)
    all = query_args(resolvers.all, DateQuery)
    item = annotate(resolvers.item, Optional[date])
    values = annotate(resolvers.values, List[Optional[date]])
    sort = annotate(resolvers.sort, List[Optional[date]])
    min = annotate(resolvers.min, date)
    max = annotate(resolvers.max, date)
    unique = annotate(resolvers.unique, Set)


@strawberry.type
class TimestampColumn:
    """column of timestamps"""

    __init__ = resolvers.__init__  # type: ignore
    count = query_args(resolvers.count, TimestampQuery)
    any = query_args(resolvers.any, TimestampQuery)
    all = query_args(resolvers.all, TimestampQuery)
    item = annotate(resolvers.item, Optional[datetime])
    values = annotate(resolvers.values, List[Optional[datetime]])
    sort = annotate(resolvers.sort, List[Optional[datetime]])
    min = annotate(resolvers.min, datetime)
    max = annotate(resolvers.max, datetime)


@strawberry.type
class TimeColumn:
    """column of times"""

    __init__ = resolvers.__init__  # type: ignore
    count = query_args(resolvers.count, TimeQuery)
    any = query_args(resolvers.any, TimeQuery)
    all = query_args(resolvers.all, TimeQuery)
    item = annotate(resolvers.item, Optional[time])
    values = annotate(resolvers.values, List[Optional[time]])
    sort = annotate(resolvers.sort, List[Optional[time]])
    min = annotate(resolvers.min, time)
    max = annotate(resolvers.max, time)


@strawberry.type
class BinaryColumn:
    """column of binaries"""

    __init__ = resolvers.__init__  # type: ignore
    count = query_args(resolvers.count, BinaryQuery)
    any = query_args(resolvers.any, BinaryQuery)
    all = query_args(resolvers.all, BinaryQuery)
    item = annotate(resolvers.item, Optional[bytes])
    values = annotate(resolvers.values, List[Optional[bytes]])


@strawberry.type
class StringSet:
    """unique strings"""

    counts: List[Long]
    __init__ = resolvers.__init__  # type: ignore
    length = strawberry.field(resolvers.length)
    values = annotate(resolvers.values, List[Optional[str]])


@strawberry.type
class StringColumn:
    """column of strings"""

    Set = StringSet
    __init__ = resolvers.__init__  # type: ignore
    count = query_args(resolvers.count, StringQuery)
    any = query_args(resolvers.any, StringQuery)
    all = query_args(resolvers.all, StringQuery)
    item = annotate(resolvers.item, Optional[str])
    values = annotate(resolvers.values, List[Optional[str]])
    sort = annotate(resolvers.sort, List[Optional[str]])
    min = annotate(resolvers.min, str)
    max = annotate(resolvers.max, str)
    unique = annotate(resolvers.unique, Set)


column_map = {
    bool: BooleanColumn,
    int: IntColumn,
    Long: LongColumn,
    float: FloatColumn,
    Decimal: DecimalColumn,
    date: DateColumn,
    datetime: TimestampColumn,
    time: TimeColumn,
    bytes: BinaryColumn,
    str: StringColumn,
}
