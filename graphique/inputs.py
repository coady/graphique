"""
GraphQL input types.
"""
import functools
import inspect
import operator
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Callable, Generic, List, Optional, TypeVar, no_type_check
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import strawberry
from strawberry import UNSET
from strawberry.annotation import StrawberryAnnotation
from strawberry.arguments import StrawberryArgument
from strawberry.field import StrawberryField
from strawberry.scalars import Base64, JSON
from strawberry.types.fields.resolver import StrawberryResolver
from typing_extensions import Annotated
from .core import ListChunk, Column
from .scalars import Duration, Long, classproperty

T = TypeVar('T')


class links:
    compute = 'https://arrow.apache.org/docs/python/api/compute.html'
    type = '[arrow type](https://arrow.apache.org/docs/python/api/datatypes.html)'


class Input:
    """Common utilities for input types."""

    nullables: dict = {}

    def __init_subclass__(cls):
        for name in set(cls.nullables) & set(cls.__annotations__):
            setattr(cls, name, default_field(description=cls.nullables[name]))

    def keys(self):
        for name, value in self.__dict__.items():
            if value is None and name not in self.nullables:
                raise TypeError(f"`{self.__class__.__name__}.{name}` is optional, not nullable")
            if value is not UNSET:
                yield name

    def __getitem__(self, name):
        value = getattr(self, name)
        return dict(value) if hasattr(value, 'keys') else value

    @classproperty
    def resolver(cls) -> Callable:
        """a decorator which flattens an input type's fields into arguments"""
        annotations = dict(cls.__annotations__)
        defaults = {name: getattr(cls, name) for name in annotations if hasattr(cls, name)}
        for field in cls._type_definition.fields:  # type: ignore
            argument = strawberry.argument(description=field.description)
            annotations[field.name] = Annotated[annotations[field.name], argument]
            defaults[field.name] = field.default_factory()
        return functools.partial(resolve_annotations, annotations=annotations, defaults=defaults)


def resolve_annotations(func: Callable, annotations: dict, defaults: dict = {}) -> StrawberryField:
    """Return field by transforming annotations into function arguments."""
    resolver = StrawberryResolver(func)
    resolver.arguments = [
        StrawberryArgument(
            python_name=name,
            graphql_name=name,
            type_annotation=StrawberryAnnotation(annotation),
            default=defaults.get(name, UNSET),
        )
        for name, annotation in annotations.items()
    ]
    return StrawberryField(
        python_name=func.__name__,
        type_annotation=StrawberryAnnotation(func.__annotations__['return']),
        description=inspect.getdoc(func),
        base_resolver=resolver,
    )


def default_field(
    default_factory: Callable = lambda: UNSET, func: Callable = None, **kwargs
) -> StrawberryField:
    """Use dataclass `default_factory` for `UNSET` or mutables."""
    if func is not None:
        kwargs['description'] = inspect.getdoc(func).splitlines()[0]  # type: ignore
    return strawberry.field(default_factory=default_factory, **kwargs)


@strawberry.input(description="predicates for scalars")
class Query(Generic[T], Input):
    eq: Optional[List[Optional[T]]] = UNSET
    ne: Optional[T] = UNSET
    lt: Optional[T] = default_field(description="<")
    le: Optional[T] = default_field(description="<=")
    gt: Optional[T] = default_field(description=r"\>")
    ge: Optional[T] = default_field(description=r"\>=")

    nullables = {
        'eq': "== or `isin`; `null` is equivalent to arrow `is_null`.",
        'ne': "!=; `null` is equivalent to arrow `is_valid`.",
    }

    @classmethod
    @no_type_check
    def resolve_types(cls, types: dict) -> Callable:
        """Return a decorator which transforms the type map into arguments."""
        defaults = dict.fromkeys(types, {})
        annotations = {
            name: Query[types[name]] for name in types if types[name] not in (list, dict)
        }  # type: ignore
        return functools.partial(resolve_annotations, annotations=annotations, defaults=defaults)


@strawberry.input
class Names:
    name: List[Optional[str]] = strawberry.field(description="column name(s)")

    def serialize(self, table):
        """Return (name, args, kwargs) suitable for computing."""
        exclude = {'name', 'alias'}
        return (
            self.alias or self.name[0],
            map(table.column, self.name),
            {name: value for name, value in self.__dict__.items() if name not in exclude},
        )


@strawberry.input(description="positional function arguments without scalars")
class Fields(Names):
    alias: str = strawberry.field(default='', description="output column name")


@strawberry.input(description="positional function arguments with typed scalar")
class Arguments(Fields, Generic[T]):
    value: List[Optional[T]] = default_field(list, description="scalar value(s)")
    cast: str = strawberry.field(default='', description=f"cast scalar to {links.type}")

    def serialize(self, table):
        """Return (name, args, kwargs) suitable for computing."""
        exclude = {'name', 'alias', 'value', 'cast'}
        values = (pa.scalar(value, self.cast) if self.cast else value for value in self.value)
        args = [table[name] if name else next(values) for name in self.name]
        return (
            self.alias or next(filter(None, self.name)),
            args + list(values),
            {name: value for name, value in self.__dict__.items() if name not in exclude},
        )


@strawberry.input(description=f"applied [functions]({links.compute})")
class Function(Generic[T], Input):
    coalesce: Optional[Arguments[T]] = default_field(func=pc.coalesce)
    fill_null_backward: Optional[Fields] = default_field(func=pc.fill_null_backward)
    fill_null_forward: Optional[Fields] = default_field(func=pc.fill_null_forward)
    if_else: Optional[Arguments[T]] = default_field(func=pc.if_else)


@strawberry.input
class _Function(Input):
    name: str
    alias: str = ''
    coalesce: Optional[List[str]] = UNSET
    fill_null_backward: bool = False
    fill_null_forward: bool = False


@strawberry.input
class ElementWiseAggregate(Arguments[T]):
    skip_nulls: bool = True


@strawberry.input
class OrdinalFunction(Function[T]):
    min_element_wise: Optional[ElementWiseAggregate[T]] = default_field(func=pc.min_element_wise)
    max_element_wise: Optional[ElementWiseAggregate[T]] = default_field(func=pc.max_element_wise)


@strawberry.input
class _OrdinalFunction(_Function):
    min_element_wise: Optional[str] = UNSET
    max_element_wise: Optional[str] = UNSET


@strawberry.input
class NumericFunction(_OrdinalFunction):
    checked: bool = strawberry.field(default=False, description="check math functions for overlow")
    power: Optional[str] = UNSET
    atan2: Optional[str] = UNSET
    abs: bool = False
    negate: bool = False
    sign: bool = False
    ln: bool = False
    log1p: bool = False
    log10: bool = False
    log2: bool = False
    floor: bool = False
    ceil: bool = False
    trunc: bool = False
    round: bool = False
    sin: bool = False
    asin: bool = False
    cos: bool = False
    acos: bool = False
    tan: bool = False
    atan: bool = False
    sqrt: bool = False


@operator.itemgetter(bool)
@strawberry.input(
    name='Function', description=f"[functions]({links.compute}#selecting-multiplexing) for booleans"
)
class BooleanFunction(Function[T]):
    and_kleene: Optional[Arguments[T]] = default_field(
        description="logical 'and' boolean values (Kleene logic)"
    )
    and_not: Optional[Arguments[T]] = default_field(description="logical 'and not' boolean values")
    and_not_kleene: Optional[Arguments[T]] = default_field(
        description="logical 'and not' boolean values (Kleene logic)"
    )
    or_kleene: Optional[Arguments[T]] = default_field(
        description="logical 'or' boolean values (Kleene logic)"
    )
    xor: Optional[Arguments[T]] = default_field(description="logical 'xor' boolean values")


@strawberry.input(description=f"[functions]({links.compute}#arithmetic-functions) for ints")
class IntFunction(NumericFunction):
    bit_wise_or: Optional[str] = UNSET
    bit_wise_and: Optional[str] = UNSET
    bit_wise_xor: Optional[str] = UNSET
    shift_left: Optional[str] = UNSET
    shift_right: Optional[str] = UNSET
    bit_wise_not: bool = False
    fill_null: Optional[int] = UNSET
    digitize: Optional[List[int]] = default_field(description=inspect.getdoc(Column.digitize))


@strawberry.input(description=f"[functions]({links.compute}#arithmetic-functions) for longs")
class LongFunction(NumericFunction):
    fill_null: Optional[Long] = UNSET
    digitize: Optional[List[Long]] = default_field(description=inspect.getdoc(Column.digitize))


@strawberry.input(description=f"[functions]({links.compute}#arithmetic-functions) for floats")
class FloatFunction(NumericFunction):
    fill_null: Optional[float] = UNSET
    digitize: Optional[List[float]] = default_field(description=inspect.getdoc(Column.digitize))
    is_finite: bool = False
    is_inf: bool = False
    is_nan: bool = False


DecimalFunction = Function[Decimal]


@strawberry.input(
    description=f"[functions]({links.compute}#temporal-component-extraction) for dates"
)
class DateFunction(_OrdinalFunction):
    fill_null: Optional[date] = UNSET
    years_between: Optional[str] = UNSET
    quarters_between: Optional[str] = UNSET
    weeks_between: Optional[str] = UNSET
    days_between: Optional[str] = UNSET
    hours_between: Optional[str] = UNSET
    minutes_between: Optional[str] = UNSET
    seconds_between: Optional[str] = UNSET
    milliseconds_between: Optional[str] = UNSET
    microseconds_between: Optional[str] = UNSET
    nanoseconds_between: Optional[str] = UNSET
    year: bool = False
    quarter: bool = False
    month: bool = False
    week: bool = False
    us_week: bool = False
    day: bool = False
    day_of_week: bool = False
    day_of_year: bool = False
    strftime: bool = False


@strawberry.input(
    description=f"[functions]({links.compute}#temporal-component-extraction) for datetimes"
)
class DateTimeFunction(_OrdinalFunction):
    fill_null: Optional[datetime] = UNSET
    years_between: Optional[str] = UNSET
    quarters_between: Optional[str] = UNSET
    weeks_between: Optional[str] = UNSET
    days_between: Optional[str] = UNSET
    hours_between: Optional[str] = UNSET
    minutes_between: Optional[str] = UNSET
    seconds_between: Optional[str] = UNSET
    milliseconds_between: Optional[str] = UNSET
    microseconds_between: Optional[str] = UNSET
    nanoseconds_between: Optional[str] = UNSET
    year: bool = False
    quarter: bool = False
    month: bool = False
    week: bool = False
    us_week: bool = False
    day: bool = False
    day_of_week: bool = False
    day_of_year: bool = False
    hour: bool = False
    minute: bool = False
    second: bool = False
    subsecond: bool = False
    millisecond: bool = False
    microsecond: bool = False
    nanosecond: bool = False
    strftime: bool = False


@strawberry.input(
    description=f"[functions]({links.compute}#temporal-component-extraction) for times"
)
class TimeFunction(_OrdinalFunction):
    fill_null: Optional[time] = UNSET
    hours_between: Optional[str] = UNSET
    minutes_between: Optional[str] = UNSET
    seconds_between: Optional[str] = UNSET
    milliseconds_between: Optional[str] = UNSET
    microseconds_between: Optional[str] = UNSET
    nanoseconds_between: Optional[str] = UNSET
    hour: bool = False
    minute: bool = False
    second: bool = False
    subsecond: bool = False
    millisecond: bool = False
    microsecond: bool = False
    nanosecond: bool = False


DurationFunction = Function[Duration]


@strawberry.input
class Join(Arguments[T]):
    null_handling: str = 'emit_null'
    null_replacement: str = ''


@strawberry.input
class ReplaceSlice(Names, Generic[T]):
    start: int
    stop: int
    replacement: T
    alias: str = strawberry.field(default='', description="output column name")


@operator.itemgetter(Base64)
@strawberry.input(
    name='Function', description=f"[functions]({links.compute}#string-transforms) for binaries"
)
class Base64Function(Function[T]):
    binary_join_element_wise: Optional[Join[T]] = default_field(func=pc.binary_join_element_wise)
    binary_length: Optional[Fields] = default_field(func=pc.binary_length)
    binary_replace_slice: Optional[ReplaceSlice] = default_field(func=pc.binary_replace_slice)


@strawberry.input
class MatchSubstring(Arguments[T]):
    ignore_case: bool = False


@strawberry.input
class Split(Arguments[T]):
    max_splits: Optional[int] = None
    reverse: bool = False


@strawberry.input
class Pad(Names):
    width: int
    padding: str = ''
    alias: str = strawberry.field(default='', description="output column name")


@strawberry.input
class ReplaceSubstring(Names):
    pattern: str
    replacement: str
    max_replacements: Optional[int] = None
    alias: str = strawberry.field(default='', description="output column name")


@strawberry.input
class Strptime(Names):
    format: str
    unit: str
    error_is_null: bool = False
    alias: str = strawberry.field(default='', description="output column name")


@strawberry.input
class Slice(Names):
    start: int
    stop: Optional[int] = None
    step: int = 1
    alias: str = strawberry.field(default='', description="output column name")


@operator.itemgetter(str)
@strawberry.input(
    name='Function', description=f"[functions]({links.compute}#string-transforms) for strings"
)
class StringFunction(OrdinalFunction[T]):
    binary_join_element_wise: Optional[Join[T]] = default_field(func=pc.binary_join_element_wise)
    binary_length: Optional[Fields] = default_field(func=pc.binary_length)

    find_substring: Optional[MatchSubstring[T]] = default_field(func=pc.find_substring)
    count_substring: Optional[MatchSubstring[T]] = default_field(func=pc.count_substring)
    match_substring: Optional[MatchSubstring[T]] = default_field(func=pc.match_substring)
    match_substring_regex: Optional[MatchSubstring[T]] = default_field(
        func=pc.match_substring_regex
    )
    match_like: Optional[MatchSubstring[T]] = default_field(func=pc.match_like)

    utf8_capitalize: Optional[Fields] = default_field(func=pc.utf8_capitalize)
    utf8_length: Optional[Fields] = default_field(func=pc.utf8_length)
    utf8_lower: Optional[Fields] = default_field(func=pc.utf8_lower)
    utf8_upper: Optional[Fields] = default_field(func=pc.utf8_upper)
    utf8_swapcase: Optional[Fields] = default_field(func=pc.utf8_swapcase)
    utf8_reverse: Optional[Fields] = default_field(func=pc.utf8_reverse)

    utf8_replace_slice: Optional[ReplaceSlice] = default_field(func=pc.utf8_replace_slice)
    utf8_split_whitespace: Optional[Split] = default_field(func=pc.utf8_split_whitespace)
    split_pattern: Optional[Split] = default_field(func=pc.split_pattern)
    split_pattern_regex: Optional[Split] = default_field(func=pc.split_pattern_regex)
    utf8_ltrim: Optional[Arguments[T]] = default_field(func=pc.utf8_ltrim)
    utf8_ltrim_whitespace: Optional[Fields] = default_field(func=pc.utf8_ltrim_whitespace)
    utf8_rtrim: Optional[Arguments[T]] = default_field(func=pc.utf8_rtrim)
    utf8_rtrim_whitespace: Optional[Fields] = default_field(func=pc.utf8_rtrim_whitespace)
    utf8_trim: Optional[Arguments[T]] = default_field(func=pc.utf8_trim)
    utf8_trim_whitespace: Optional[Fields] = default_field(func=pc.utf8_trim_whitespace)
    utf8_center: Optional[Pad] = default_field(func=pc.utf8_center)
    utf8_lpad: Optional[Pad] = default_field(func=pc.utf8_lpad)
    utf8_rpad: Optional[Pad] = default_field(func=pc.utf8_rpad)
    replace_substring: Optional[ReplaceSubstring] = default_field(func=pc.replace_substring)
    strptime: Optional[Strptime] = default_field(func=pc.strptime)
    utf8_slice_codeunits: Optional[Slice] = default_field(func=pc.utf8_slice_codeunits)

    string_is_ascii: Optional[Fields] = default_field(func=pc.string_is_ascii)
    utf8_is_alnum: Optional[Fields] = default_field(func=pc.utf8_is_alnum)
    utf8_is_alpha: Optional[Fields] = default_field(func=pc.utf8_is_alpha)
    utf8_is_decimal: Optional[Fields] = default_field(func=pc.utf8_is_decimal)
    utf8_is_digit: Optional[Fields] = default_field(func=pc.utf8_is_digit)
    utf8_is_lower: Optional[Fields] = default_field(func=pc.utf8_is_lower)
    utf8_is_numeric: Optional[Fields] = default_field(func=pc.utf8_is_numeric)
    utf8_is_printable: Optional[Fields] = default_field(func=pc.utf8_is_printable)
    utf8_is_space: Optional[Fields] = default_field(func=pc.utf8_is_space)
    utf8_is_title: Optional[Fields] = default_field(func=pc.utf8_is_title)
    utf8_is_upper: Optional[Fields] = default_field(func=pc.utf8_is_upper)


@strawberry.input
class StructField(Names):
    indices: List[int]
    alias: str = strawberry.field(default='', description="output column name")


@strawberry.input(description=f"[functions]({links.compute}#selecting-multiplexing) for structs")
class StructFunction(Input):
    fill_null_backward: Optional[Fields] = default_field(func=pc.fill_null_backward)
    fill_null_forward: Optional[Fields] = default_field(func=pc.fill_null_forward)
    case_when: Optional[Fields] = default_field(func=pc.case_when)
    struct_field: Optional[StructField] = default_field(func=pc.struct_field)


@strawberry.input(description=f"[functions]({links.compute}#structural-transforms) for list")
class ListFunction(Input):
    name: str = ''
    alias: str = ''
    filter: 'Expression' = default_field(dict, description="filter within list scalars")
    mode: bool = strawberry.field(default=False, description=inspect.getdoc(ListChunk.mode))
    quantile: bool = strawberry.field(default=False, description=inspect.getdoc(ListChunk.quantile))
    value_length: bool = strawberry.field(
        default=False, description="faster than `count` aggregation"
    )


@strawberry.input(
    description=f"names and optional aliases for [aggregation]({links.compute}#aggregations)"
)
class Aggregate(Input):
    name: str
    alias: str = ''


@strawberry.input(
    description=f"options for count [aggregation]({links.compute}#grouped-aggregations)"
)
class CountAggregate(Aggregate):
    mode: str = 'only_valid'


@strawberry.input(
    description=f"options for scalar [aggregation]({links.compute}#grouped-aggregations)"
)
class ScalarAggregate(Aggregate):
    skip_nulls: bool = True
    min_count: int = 1


@strawberry.input(
    description=f"options for variance [aggregation]({links.compute}#grouped-aggregations)"
)
class VarianceAggregate(ScalarAggregate):
    ddof: int = 0


@strawberry.input(
    description=f"options for tdigest [aggregation]({links.compute}#grouped-aggregations)"
)
class TDigestAggregate(ScalarAggregate):
    q: List[float] = (0.5,)  # type: ignore
    delta: int = 100
    buffer_size: int = 500


@strawberry.input
class Aggregations(Input):
    all: List[ScalarAggregate] = default_field(list, func=pc.all)
    any: List[ScalarAggregate] = default_field(list, func=pc.any)
    count: List[CountAggregate] = default_field(list, func=pc.count)
    count_distinct: List[CountAggregate] = default_field(list, func=pc.count_distinct)
    distinct: List[CountAggregate] = default_field(
        list, description="distinct values within each scalar"
    )
    first: List[Aggregate] = default_field(list, func=ListChunk.first)
    last: List[Aggregate] = default_field(list, func=ListChunk.last)
    max: List[ScalarAggregate] = default_field(list, func=pc.max)
    mean: List[ScalarAggregate] = default_field(list, func=pc.mean)
    min: List[ScalarAggregate] = default_field(list, func=pc.min)
    one: List[Aggregate] = default_field(list, description="arbitrary value within each scalar")
    product: List[ScalarAggregate] = default_field(list, func=pc.product)
    stddev: List[VarianceAggregate] = default_field(list, func=pc.stddev)
    sum: List[ScalarAggregate] = default_field(list, func=pc.sum)
    approximate_median: List[ScalarAggregate] = default_field(list, func=pc.approximate_median)
    tdigest: List[TDigestAggregate] = default_field(list, func=pc.tdigest)
    variance: List[VarianceAggregate] = default_field(list, func=pc.variance)


@strawberry.input(description="discrete difference predicates; durations may be in float seconds")
class Diff(Input):
    name: str
    lt: Optional[float] = UNSET
    le: Optional[float] = UNSET
    gt: Optional[float] = UNSET
    ge: Optional[float] = UNSET
    predicates = {
        'ne': pc.not_equal,
        'lt': pc.less,
        'le': pc.less_equal,
        'gt': pc.greater,
        'ge': pc.greater_equal,
    }
    nullables = dict.fromkeys(
        predicates,
        "`null` compares the arrays element-wise. A float computes the discrete difference first.",
    )


@strawberry.input(
    description=f"[functions]({links.compute}#arithmetic-functions) projected across two columns"
)
class Projections(Input):
    fill_null: Optional[List[str]] = UNSET
    binary_join_element_wise: Optional[List[str]] = UNSET
    min_element_wise: Optional[str] = UNSET
    max_element_wise: Optional[str] = UNSET
    power: Optional[str] = UNSET
    atan2: Optional[str] = UNSET
    bit_wise_or: Optional[str] = UNSET
    bit_wise_and: Optional[str] = UNSET
    bit_wise_xor: Optional[str] = UNSET
    shift_left: Optional[str] = UNSET
    shift_right: Optional[str] = UNSET
    years_between: Optional[str] = UNSET
    quarters_between: Optional[str] = UNSET
    weeks_between: Optional[str] = UNSET
    days_between: Optional[str] = UNSET
    hours_between: Optional[str] = UNSET
    minutes_between: Optional[str] = UNSET
    seconds_between: Optional[str] = UNSET
    milliseconds_between: Optional[str] = UNSET
    microseconds_between: Optional[str] = UNSET
    nanoseconds_between: Optional[str] = UNSET


@strawberry.input(
    description="""
[Dataset expression](https://arrow.apache.org/docs/python/generated/pyarrow.dataset.Expression.html)
used for [scanning](https://arrow.apache.org/docs/python/generated/pyarrow.dataset.Scanner.html).

Expects one of: a field `name`, a scalar, or an operator with expressions. Single values can be passed for an
[input `List`](https://spec.graphql.org/October2021/#sec-List.Input-Coercion).
* `eq` with a list scalar is equivalent to `isin`
* `eq` with a `null` scalars is equivalent `is_null`
* `ne` with a `null` scalar is equivalent to `is_valid`
"""
)
class Expression:
    name: str = strawberry.field(default='', description="field name")
    alias: str = strawberry.field(default='', description="name for outermost columns")
    cast: str = strawberry.field(default='', description=f"cast as {links.type}")
    value: Optional[JSON] = default_field(description="JSON scalar; also see typed scalars")

    base64: List[bytes] = default_field(list)
    boolean: List[bool] = default_field(list)
    date_: List[date] = default_field(list, name='date')
    datetime_: List[datetime] = default_field(list, name='datetime')
    decimal: List[Decimal] = default_field(list)
    duration: List[timedelta] = default_field(list)
    float_: List[float] = default_field(list, name='float')
    int_: List[int] = default_field(list, name='int')
    long: List[Long] = default_field(list)
    string: List[str] = default_field(list)
    time_: List[time] = default_field(list, name='time')

    eq: List['Expression'] = default_field(list, description="==")
    ne: List['Expression'] = default_field(list, description="!=")
    lt: List['Expression'] = default_field(list, description="<")
    le: List['Expression'] = default_field(list, description="<=")
    gt: List['Expression'] = default_field(list, description=r"\>")
    ge: List['Expression'] = default_field(list, description=r"\>=")

    add: List['Expression'] = default_field(list, description=r"\+")
    mul: List['Expression'] = default_field(list, description=r"\*")
    sub: List['Expression'] = default_field(list, description=r"\-")
    truediv: List['Expression'] = default_field(list, name='div', description='/')

    and_: List['Expression'] = default_field(list, name='and', description="&")
    or_: List['Expression'] = default_field(list, name='or', description="|")
    inv: Optional['Expression'] = default_field(description="~")

    ops = ('eq', 'ne', 'lt', 'le', 'gt', 'ge', 'add', 'mul', 'sub', 'truediv', 'and_', 'or_')
    scalars = (
        'base64',
        'boolean',
        'date_',
        'datetime_',
        'decimal',
        'duration',
        'float_',
        'int_',
        'long',
        'string',
        'time_',
    )

    def to_arrow(self) -> Optional[ds.Expression]:
        """Transform GraphQL expression into a dataset expression."""
        fields = []
        if self.name:
            field = ds.field(self.name)
            fields.append(field)
        for name in self.scalars:
            scalars = getattr(self, name)
            if self.cast:
                scalars = [pa.scalar(scalar, self.cast) for scalar in scalars]
            if scalars:
                fields.append(scalars[0] if len(scalars) == 1 else scalars)
        if self.value is not UNSET:
            fields.append(self.value)
        for op in self.ops:
            exprs = [expr.to_arrow() for expr in getattr(self, op)]
            if exprs:
                if op == 'eq' and isinstance(exprs[-1], list):
                    field = ds.Expression.isin(*exprs)
                elif exprs[-1] is None and op in ('eq', 'ne'):
                    field, _ = exprs
                    field = field.is_null() if op == 'eq' else field.is_valid()
                else:
                    field = functools.reduce(getattr(operator, op), exprs)
                fields.append(field)
        if self.inv is not UNSET:
            fields.append(~self.inv.to_arrow())  # type: ignore
        if not fields:
            return None
        if len(fields) > 1:
            raise ValueError(f"conflicting inputs: {', '.join(map(str, fields))}")
        (field,) = fields
        return field.cast(self.cast) if self.cast and isinstance(field, ds.Expression) else field

    @classmethod
    @no_type_check
    def from_query(cls, **queries: Query) -> 'Expression':
        """Transform query syntax into an Expression input."""
        exprs = []
        for name, query in queries.items():
            field = cls(name=name)
            exprs += (cls(**{op: [field, cls(value=value)]}) for op, value in dict(query).items())
        return cls(and_=exprs)
