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
import pyarrow.dataset as ds
import strawberry
from strawberry import UNSET
from strawberry.annotation import StrawberryAnnotation
from strawberry.arguments import StrawberryArgument
from strawberry.field import StrawberryField
from strawberry.scalars import JSON
from strawberry.types.fields.resolver import StrawberryResolver
from typing_extensions import Annotated
from .core import ListChunk, Column
from .scalars import Long, classproperty

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


def default_field(default_factory: Callable = lambda: UNSET, **kwargs) -> StrawberryField:
    """Use dataclass `default_factory` for `UNSET` or mutables."""
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
class Function(Input):
    name: str
    alias: str = ''
    coalesce: Optional[List[str]] = UNSET
    fill_null_backward: bool = False
    fill_null_forward: bool = False


@strawberry.input
class OrdinalFunction(Function):
    min_element_wise: Optional[str] = UNSET
    max_element_wise: Optional[str] = UNSET


@strawberry.input
class NumericFunction(OrdinalFunction):
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


@strawberry.input(description=f"[functions]({links.compute}#selecting-multiplexing) for booleans")
class BooleanFunction(Function):
    if_else: Optional[List[str]] = UNSET
    and_: Optional[str] = default_field(name='and')
    or_: Optional[str] = default_field(name='or')
    xor: Optional[str] = UNSET
    and_not: Optional[str] = UNSET
    kleene: bool = False


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


@strawberry.input(description="functions for decimals")
class DecimalFunction(Function):
    fill_null: Optional[Decimal] = UNSET


@strawberry.input(
    description=f"[functions]({links.compute}#temporal-component-extraction) for dates"
)
class DateFunction(OrdinalFunction):
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
class DateTimeFunction(OrdinalFunction):
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
class TimeFunction(OrdinalFunction):
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


@strawberry.input(description="functions for durations")
class DurationFunction(Function):
    fill_null: Optional[timedelta] = UNSET


@strawberry.input(description=f"[functions]({links.compute}#string-transforms) for binaries")
class Base64Function(Function):
    binary_join_element_wise: Optional[List[str]] = UNSET
    fill_null: Optional[bytes] = UNSET
    binary_length: bool = False


@strawberry.input(description=f"[functions]({links.compute}#string-transforms) for strings")
class StringFunction(Function):
    binary_join_element_wise: Optional[List[str]] = UNSET
    find_substring: Optional[str] = UNSET
    count_substring: Optional[str] = UNSET
    match_substring: Optional[str] = UNSET
    match_like: Optional[str] = UNSET
    ignore_case: bool = strawberry.field(default=False, description="case option for substrings")
    regex: bool = strawberry.field(default=False, description="regex option for substrings")
    fill_null: Optional[str] = UNSET
    utf8_capitalize: bool = False
    utf8_length: bool = False
    utf8_lower: bool = False
    utf8_upper: bool = False
    utf8_swapcase: bool = False
    utf8_reverse: bool = False
    string_is_ascii: bool = False
    utf8_is_alnum: bool = False
    utf8_is_alpha: bool = False
    utf8_is_decimal: bool = False
    utf8_is_digit: bool = False
    utf8_is_lower: bool = False
    utf8_is_numeric: bool = False
    utf8_is_printable: bool = False
    utf8_is_space: bool = False
    utf8_is_title: bool = False
    utf8_is_upper: bool = False


@strawberry.input(description=f"[functions]({links.compute}#selecting-multiplexing) for structs")
class StructFunction(Function):
    case_when: Optional[List[str]] = UNSET


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


@strawberry.input
class Aggregations(Input):
    all: List[ScalarAggregate] = default_field(list, description=inspect.getdoc(ListChunk.all))
    any: List[ScalarAggregate] = default_field(list, description=inspect.getdoc(ListChunk.any))
    count: List[CountAggregate] = default_field(list, description=inspect.getdoc(ListChunk.count))
    count_distinct: List[CountAggregate] = default_field(
        list, description=inspect.getdoc(ListChunk.count_distinct)
    )
    distinct: List[CountAggregate] = default_field(
        list, description=inspect.getdoc(ListChunk.distinct)
    )
    first: List[Aggregate] = default_field(list, description=inspect.getdoc(ListChunk.first))
    last: List[Aggregate] = default_field(list, description=inspect.getdoc(ListChunk.last))
    max: List[ScalarAggregate] = default_field(list, description=inspect.getdoc(ListChunk.max))
    mean: List[ScalarAggregate] = default_field(list, description=inspect.getdoc(ListChunk.mean))
    min: List[ScalarAggregate] = default_field(list, description=inspect.getdoc(ListChunk.min))
    one: List[Aggregate] = default_field(list, description=inspect.getdoc(ListChunk.one))
    product: List[ScalarAggregate] = default_field(
        list, description=inspect.getdoc(ListChunk.product)
    )
    stddev: List[ScalarAggregate] = default_field(
        list, description=inspect.getdoc(ListChunk.stddev)
    )
    sum: List[ScalarAggregate] = default_field(list, description=inspect.getdoc(ListChunk.sum))
    tdigest: List[ScalarAggregate] = default_field(
        list, description=inspect.getdoc(ListChunk.tdigest)
    )
    variance: List[ScalarAggregate] = default_field(
        list, description=inspect.getdoc(ListChunk.variance)
    )


@strawberry.input(description="discrete difference predicates; durations may be in float seconds")
class Diff(Input):
    name: str
    less: Optional[float] = UNSET
    less_equal: Optional[float] = UNSET
    greater: Optional[float] = UNSET
    greater_equal: Optional[float] = UNSET
    nullables = dict.fromkeys(
        ['less', 'less_equal', 'greater', 'greater_equal'],
        "`null` compares the arrays element-wise. A float computes the discrete difference first.",
    )


@strawberry.input(
    description=f"[functions]({links.compute}#arithmetic-functions) projected across two columns"
)
class Projections(Input):
    coalesce: Optional[List[str]] = UNSET
    fill_null: Optional[List[str]] = UNSET
    binary_join_element_wise: Optional[List[str]] = UNSET
    if_else: Optional[List[str]] = UNSET
    case_when: Optional[List[str]] = UNSET
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
