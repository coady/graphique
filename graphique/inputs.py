"""
GraphQL input types.
"""

import functools
import inspect
import operator
from collections.abc import Callable, Iterable
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Generic, Optional, TypeVar, no_type_check
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import strawberry
from strawberry import UNSET
from strawberry.annotation import StrawberryAnnotation
from strawberry.types.arguments import StrawberryArgument
from strawberry.schema_directive import Location
from strawberry.types.field import StrawberryField
from strawberry.scalars import JSON
from typing_extensions import Self
from .core import Agg
from .scalars import Long

T = TypeVar('T')


class links:
    compute = 'https://arrow.apache.org/docs/python/api/compute.html'
    type = '[arrow type](https://arrow.apache.org/docs/python/api/datatypes.html)'


class Input:
    """Common utilities for input types."""

    nullables: set = set()

    def keys(self):
        for name, value in self.__dict__.items():
            if value is None and name not in self.nullables:
                raise TypeError(f"`{self.__class__.__name__}.{name}` is optional, not nullable")
            if value is not UNSET:
                yield name

    def __getitem__(self, name):
        value = getattr(self, name)
        return dict(value) if hasattr(value, 'keys') else value


def use_doc(decorator: Callable, **kwargs):
    return lambda func: decorator(description=inspect.getdoc(func), **kwargs)(func)


@use_doc(
    strawberry.schema_directive,
    locations=[Location.ARGUMENT_DEFINITION, Location.INPUT_FIELD_DEFINITION],
)
class optional:
    """This input is optional, not nullable.
    If the client insists on sending an explicit null value, the behavior is undefined.
    """


@use_doc(
    strawberry.schema_directive,
    locations=[Location.ARGUMENT_DEFINITION, Location.FIELD_DEFINITION],
)
class provisional:
    """Provisional feature; subject to change in the future."""


def default_field(
    default=UNSET, func: Optional[Callable] = None, nullable: bool = False, **kwargs
) -> StrawberryField:
    """Use dataclass `default_factory` for `UNSET` or mutables."""
    if func is not None:
        kwargs['description'] = inspect.getdoc(func).splitlines()[0]  # type: ignore
    if not nullable and default is UNSET:
        kwargs.setdefault('directives', []).append(optional())
    return strawberry.field(default_factory=type(default), **kwargs)


@strawberry.input(description="predicates for scalars")
class Filter(Generic[T], Input):
    eq: Optional[list[Optional[T]]] = default_field(
        description="== or `isin`; `null` is equivalent to arrow `is_null`.", nullable=True
    )
    ne: Optional[T] = default_field(
        description="!=; `null` is equivalent to arrow `is_valid`.", nullable=True
    )
    lt: Optional[T] = default_field(description="<")
    le: Optional[T] = default_field(description="<=")
    gt: Optional[T] = default_field(description=r"\>")
    ge: Optional[T] = default_field(description=r"\>=")

    nullables = {'eq', 'ne'}

    @classmethod
    def resolve_args(cls, types: dict) -> Iterable[StrawberryArgument]:
        """Generate dynamically resolved arguments for filter field."""
        for name in types:
            annotation = StrawberryAnnotation(cls[types[name]])  # type: ignore
            if types[name] not in (list, dict):
                yield StrawberryArgument(name, name, annotation, default={})


@strawberry.input(description=f"name and optional alias for [compute functions]({links.compute})")
class Field(Agg):
    name: str = strawberry.field(description="column name")
    alias: str = strawberry.field(default='', description="output column name")

    __init__ = Agg.__init__

    def __init_subclass__(cls):
        cls.__init__ = cls.__init__


@strawberry.input
class Cumulative(Field):
    start: float = 0.0
    skip_nulls: bool = False
    checked: bool = False


@strawberry.input
class Pairwise(Field):
    period: int = 1
    checked: bool = False


@strawberry.input
class Index(Field):
    value: JSON
    start: Long = 0
    end: Optional[Long] = None


@strawberry.input
class Mode(Field):
    n: int = 1
    skip_nulls: bool = True
    min_count: int = 1


@strawberry.input
class Quantile(Field):
    q: list[float] = (0.5,)  # type: ignore
    interpolation: str = 'linear'
    skip_nulls: bool = True
    min_count: int = 1


@strawberry.input
class Rank(Field):
    sort_keys: str = 'ascending'
    null_placement: str = 'at_end'
    tiebreaker: str = 'first'


@strawberry.input
class Sort:
    by: list[str]
    length: Optional[Long] = None


@strawberry.input
class Ranked:
    by: list[str]
    max: int = 1


@strawberry.input(description=f"[functions]({links.compute}#structural-transforms) for lists")
class ListFunction(Input):
    deprecation = "List scalar functions will be moved to `scan(...: {list: ...})`"

    filter: 'Expression' = default_field({}, description="filter within list scalars")
    sort: Optional[Sort] = default_field(description="sort within list scalars")
    rank: Optional[Ranked] = default_field(description="select by dense rank within list scalars")
    index: Optional[Index] = default_field(func=pc.index, deprecation_reason=deprecation)
    mode: Optional[Mode] = default_field(func=pc.mode, deprecation_reason=deprecation)
    quantile: Optional[Quantile] = default_field(func=pc.quantile, deprecation_reason=deprecation)

    def keys(self):
        return set(super().keys()) - {'filter', 'sort', 'rank'}


@strawberry.input(description=f"options for count [aggregation]({links.compute}#aggregations)")
class CountAggregate(Field):
    mode: str = 'only_valid'


@strawberry.input(description=f"options for scalar [aggregation]({links.compute}#aggregations)")
class ScalarAggregate(Field):
    skip_nulls: bool = True


@strawberry.input(description=f"options for variance [aggregation]({links.compute}#aggregations)")
class VarianceAggregate(ScalarAggregate):
    ddof: int = 0


@strawberry.input(description=f"options for tdigest [aggregation]({links.compute}#aggregations)")
class TDigestAggregate(ScalarAggregate):
    q: list[float] = (0.5,)  # type: ignore
    delta: int = 100
    buffer_size: int = 500


@strawberry.input
class ScalarAggregates(Input):
    all: list[ScalarAggregate] = default_field([], func=pc.all)
    any: list[ScalarAggregate] = default_field([], func=pc.any)
    approximate_median: list[ScalarAggregate] = default_field([], func=pc.approximate_median)
    count: list[CountAggregate] = default_field([], func=pc.count)
    count_distinct: list[CountAggregate] = default_field([], func=pc.count_distinct)
    first: list[ScalarAggregate] = default_field([], func=pc.first)
    first_last: list[ScalarAggregate] = default_field([], func=pc.first_last)
    last: list[ScalarAggregate] = default_field([], func=pc.last)
    max: list[ScalarAggregate] = default_field([], func=pc.max)
    mean: list[ScalarAggregate] = default_field([], func=pc.mean)
    min: list[ScalarAggregate] = default_field([], func=pc.min)
    min_max: list[ScalarAggregate] = default_field([], func=pc.min_max)
    product: list[ScalarAggregate] = default_field([], func=pc.product)
    stddev: list[VarianceAggregate] = default_field([], func=pc.stddev)
    sum: list[ScalarAggregate] = default_field([], func=pc.sum)
    tdigest: list[TDigestAggregate] = default_field([], func=pc.tdigest)
    variance: list[VarianceAggregate] = default_field([], func=pc.variance)

    def keys(self):
        return (key.rstrip('_') for key in super().keys() if self[key])

    def __getitem__(self, name):
        return super().__getitem__('list_' if name == 'list' else name)


@strawberry.input
class HashAggregates(ScalarAggregates):
    distinct: list[CountAggregate] = default_field(
        [], description="distinct values within each scalar"
    )
    list_: list[Field] = default_field([], name='list', description="all values within each scalar")
    one: list[Field] = default_field([], description="arbitrary value within each scalar")


@use_doc(strawberry.input)
class Diff(Input):
    """Discrete difference predicates, applied in forwards direction (array[i + 1] ? array[i]).

    By default compares by not equal. Specifying `null` with a predicate compares pairwise.
    A float computes the discrete difference first; durations may be in float seconds.
    """

    name: str
    less: Optional[float] = default_field(name='lt', description="<", nullable=True)
    less_equal: Optional[float] = default_field(name='le', description="<=", nullable=True)
    greater: Optional[float] = default_field(name='gt', description=r"\>", nullable=True)
    greater_equal: Optional[float] = default_field(name='ge', description=r"\>=", nullable=True)

    nullables = {'less', 'less_equal', 'greater', 'greater_equal'}


@use_doc(strawberry.input)
class Expression:
    """[Dataset expression](https://arrow.apache.org/docs/python/generated/pyarrow.dataset.Expression.html)
    used for [scanning](https://arrow.apache.org/docs/python/generated/pyarrow.dataset.Scanner.html).

    Expects one of: a field `name`, a scalar, or an operator with expressions. Single values can be passed for an
    [input `List`](https://spec.graphql.org/October2021/#sec-List.Input-Coercion).
    * `eq` with a list scalar is equivalent to `isin`
    * `eq` with a `null` scalar is equivalent `is_null`
    * `ne` with a `null` scalar is equivalent to `is_valid`
    """

    name: list[str] = default_field([], description="field name(s)")
    cast: str = strawberry.field(default='', description=f"cast as {links.type}")
    safe: bool = strawberry.field(default=True, description="check for conversion errors on cast")
    value: Optional[JSON] = default_field(
        description="JSON scalar; also see typed scalars", nullable=True
    )
    kleene: bool = strawberry.field(default=False, description="use kleene logic for booleans")
    checked: bool = strawberry.field(default=False, description="check for overflow errors")

    base64: list[bytes] = default_field([])
    date_: list[date] = default_field([], name='date')
    datetime_: list[datetime] = default_field([], name='datetime')
    decimal: list[Decimal] = default_field([])
    duration: list[timedelta] = default_field([])
    time_: list[time] = default_field([], name='time')

    eq: list['Expression'] = default_field([], description="==")
    ne: list['Expression'] = default_field([], description="!=")
    lt: list['Expression'] = default_field([], description="<")
    le: list['Expression'] = default_field([], description="<=")
    gt: list['Expression'] = default_field([], description=r"\>")
    ge: list['Expression'] = default_field([], description=r"\>=")
    inv: Optional['Expression'] = default_field(description="~")

    abs: Optional['Expression'] = default_field(func=pc.abs)
    add: list['Expression'] = default_field([], func=pc.add)
    divide: list['Expression'] = default_field([], func=pc.divide)
    multiply: list['Expression'] = default_field([], func=pc.multiply)
    negate: Optional['Expression'] = default_field(func=pc.negate)
    power: list['Expression'] = default_field([], func=pc.power)
    sign: Optional['Expression'] = default_field(func=pc.sign)
    subtract: list['Expression'] = default_field([], func=pc.subtract)

    bit_wise: Optional['BitWise'] = default_field(description="bit-wise functions")
    rounding: Optional['Rounding'] = default_field(description="rounding functions")
    log: Optional['Log'] = default_field(description="logarithmic functions")
    trig: Optional['Trig'] = default_field(description="trigonometry functions")
    element_wise: Optional['ElementWise'] = default_field(
        description="element-wise aggregate functions"
    )

    and_: list['Expression'] = default_field([], name='and', description="&")
    and_not: list['Expression'] = default_field([], func=pc.and_not)
    or_: list['Expression'] = default_field([], name='or', description="|")
    xor: list['Expression'] = default_field([], func=pc.xor)

    utf8: Optional['Utf8'] = default_field(description="utf8 string functions")
    string_is_ascii: Optional['Expression'] = default_field(func=pc.string_is_ascii)
    substring: Optional['MatchSubstring'] = default_field(description="match substring functions")

    binary: Optional['Binary'] = default_field(description="binary functions")
    set_lookup: Optional['SetLookup'] = default_field(description="set lookup functions")

    is_finite: Optional['Expression'] = default_field(func=pc.is_finite)
    is_inf: Optional['Expression'] = default_field(func=pc.is_inf)
    is_nan: Optional['Expression'] = default_field(func=pc.is_nan)
    true_unless_null: Optional['Expression'] = default_field(func=pc.true_unless_null)

    case_when: list['Expression'] = default_field([], func=pc.case_when)
    choose: list['Expression'] = default_field([], func=pc.choose)
    coalesce: list['Expression'] = default_field([], func=pc.coalesce)
    if_else: list['Expression'] = default_field([], func=pc.if_else)

    temporal: Optional['Temporal'] = default_field(description="temporal functions")

    replace_with_mask: list['Expression'] = default_field([], func=pc.replace_with_mask)

    list: Optional['Lists'] = default_field(description="list array functions")

    unaries = ('inv', 'abs', 'negate', 'sign', 'string_is_ascii', 'is_finite', 'is_inf', 'is_nan')
    associatives = ('add', 'multiply', 'and_', 'or_', 'xor')
    variadics = ('eq', 'ne', 'lt', 'le', 'gt', 'ge', 'divide', 'power', 'subtract', 'and_not')
    variadics += ('case_when', 'choose', 'coalesce', 'if_else', 'replace_with_mask')  # type: ignore
    scalars = ('base64', 'date_', 'datetime_', 'decimal', 'duration', 'time_')
    groups = ('bit_wise', 'rounding', 'log', 'trig', 'element_wise', 'utf8', 'substring', 'binary')
    groups += ('set_lookup', 'temporal', 'list')  # type: ignore

    def to_arrow(self) -> Optional[ds.Expression]:
        """Transform GraphQL expression into a dataset expression."""
        fields = []
        if self.name:
            fields.append(pc.field(*self.name))
        for name in self.scalars:
            scalars = list(map(self.getscalar, getattr(self, name)))
            if scalars:
                fields.append(scalars[0] if len(scalars) == 1 else scalars)
        if self.value is not UNSET:
            fields.append(self.getscalar(self.value))
        for op in self.associatives:
            exprs = [expr.to_arrow() for expr in getattr(self, op)]
            if exprs:
                fields.append(functools.reduce(self.getfunc(op), exprs))
        for op in self.variadics:
            exprs = [expr.to_arrow() for expr in getattr(self, op)]
            if exprs:
                if op == 'eq' and isinstance(exprs[-1], list):
                    field = ds.Expression.isin(*exprs)
                elif exprs[-1] is None and op in ('eq', 'ne'):
                    field, _ = exprs
                    field = field.is_null() if op == 'eq' else field.is_valid()
                else:
                    field = self.getfunc(op)(*exprs)
                fields.append(field)
        for group in operator.attrgetter(*self.groups)(self):
            if group is not UNSET:
                fields += group.to_fields()
        for op in self.unaries:
            expr = getattr(self, op)
            if expr is not UNSET:
                fields.append(self.getfunc(op)(expr.to_arrow()))
        if not fields:
            return None
        if len(fields) > 1:
            raise ValueError(f"conflicting inputs: {', '.join(map(str, fields))}")
        (field,) = fields
        cast = self.cast and isinstance(field, ds.Expression)
        return field.cast(self.cast, self.safe) if cast else field

    def getscalar(self, value):
        return pa.scalar(value, self.cast) if self.cast else value

    def getfunc(self, name):
        if self.kleene:
            name = name.rstrip('_') + '_kleene'
        if self.checked:
            name += '_checked'
        if name.endswith('_'):  # `and_` and `or_` functions differ from operators
            return getattr(operator, name)
        return getattr(pc if hasattr(pc, name) else operator, name)

    @classmethod
    @no_type_check
    def from_query(cls, **queries: Filter) -> Self:
        """Transform query syntax into an Expression input."""
        exprs = []
        for name, query in queries.items():
            field = cls(name=[name])
            exprs += (cls(**{op: [field, cls(value=value)]}) for op, value in dict(query).items())
        return cls(and_=exprs)


@strawberry.input(description="an `Expression` with an optional alias")
class Projection(Expression):
    alias: str = strawberry.field(default='', description="name of projected column")


class Fields:
    """Fields grouped by naming conventions or common options."""

    prefix: str = ''

    def to_fields(self) -> Iterable[ds.Expression]:
        funcs, arguments, options = [], [], {}
        for field in self.__strawberry_definition__.fields:  # type: ignore
            value = getattr(self, field.name)
            if isinstance(value, Expression):
                value = [value]
            if not isinstance(value, (list, type(UNSET))):
                options[field.name] = value
            elif value:
                funcs.append(self.getfunc(field.name))
                arguments.append([expr.to_arrow() for expr in value])
        for func, args in zip(funcs, arguments):
            keys = set(options) & set(inspect.signature(func).parameters)
            yield func(*args, **{key: options[key] for key in keys})

    def getfunc(self, name):
        return getattr(pc, self.prefix + name)


@strawberry.input(description="Bit-wise functions.")
class BitWise(Fields):
    and_: list[Expression] = default_field([], name='and', func=pc.bit_wise_and)
    not_: list[Expression] = default_field([], name='not', func=pc.bit_wise_not)
    or_: list[Expression] = default_field([], name='or', func=pc.bit_wise_or)
    xor: list[Expression] = default_field([], func=pc.bit_wise_xor)
    shift_left: list[Expression] = default_field([], func=pc.shift_left)
    shift_right: list[Expression] = default_field([], func=pc.shift_right)

    def getfunc(self, name):
        return getattr(pc, name if name.startswith('shift') else 'bit_wise_' + name.rstrip('_'))


@strawberry.input(description="Rounding functions.")
class Rounding(Fields):
    ceil: Optional[Expression] = default_field(func=pc.ceil)
    floor: Optional[Expression] = default_field(func=pc.floor)
    trunc: Optional[Expression] = default_field(func=pc.trunc)

    round: Optional[Expression] = default_field(func=pc.round)
    ndigits: int = 0
    round_mode: str = 'half_to_even'
    multiple: float = 1.0

    def getfunc(self, name):
        if name == 'round' and self.multiple != 1.0:
            name = 'round_to_multiple'
        return getattr(pc, name)


@strawberry.input(description="Logarithmic functions.")
class Log(Fields):
    ln: Optional[Expression] = default_field(func=pc.ln)
    log1p: Optional[Expression] = default_field(func=pc.log1p)
    logb: list[Expression] = default_field([], func=pc.logb)


@strawberry.input(description="Trigonometry functions.")
class Trig(Fields):
    checked: bool = strawberry.field(default=False, description="check for overflow errors")

    acos: Optional[Expression] = default_field(func=pc.acos)
    asin: Optional[Expression] = default_field(func=pc.asin)
    atan: Optional[Expression] = default_field(func=pc.atan)
    atan2: list[Expression] = default_field([], func=pc.atan2)
    cos: Optional[Expression] = default_field(func=pc.cos)
    sin: Optional[Expression] = default_field(func=pc.sin)
    tan: Optional[Expression] = default_field(func=pc.tan)

    def getfunc(self, name):
        return getattr(pc, name + ('_checked' * self.checked))


@strawberry.input(description="Element-wise aggregate functions.")
class ElementWise(Fields):
    min_element_wise: list[Expression] = default_field([], name='min', func=pc.min_element_wise)
    max_element_wise: list[Expression] = default_field([], name='max', func=pc.max_element_wise)
    skip_nulls: bool = True


@strawberry.input(description="Utf8 string functions.")
class Utf8(Fields):
    is_alnum: Optional[Expression] = default_field(func=pc.utf8_is_alnum)
    is_alpha: Optional[Expression] = default_field(func=pc.utf8_is_alpha)
    is_decimal: Optional[Expression] = default_field(func=pc.utf8_is_decimal)
    is_digit: Optional[Expression] = default_field(func=pc.utf8_is_digit)
    is_lower: Optional[Expression] = default_field(func=pc.utf8_is_lower)
    is_numeric: Optional[Expression] = default_field(func=pc.utf8_is_numeric)
    is_printable: Optional[Expression] = default_field(func=pc.utf8_is_printable)
    is_space: Optional[Expression] = default_field(func=pc.utf8_is_space)
    is_title: Optional[Expression] = default_field(func=pc.utf8_is_title)
    is_upper: Optional[Expression] = default_field(func=pc.utf8_is_upper)

    capitalize: Optional[Expression] = default_field(func=pc.utf8_capitalize)
    length: Optional[Expression] = default_field(func=pc.utf8_length)
    lower: Optional[Expression] = default_field(func=pc.utf8_lower)
    reverse: Optional[Expression] = default_field(func=pc.utf8_reverse)
    swapcase: Optional[Expression] = default_field(func=pc.utf8_swapcase)
    title: Optional[Expression] = default_field(func=pc.utf8_title)
    upper: Optional[Expression] = default_field(func=pc.utf8_upper)

    ltrim: Optional[Expression] = default_field(func=pc.utf8_ltrim)
    rtrim: Optional[Expression] = default_field(func=pc.utf8_rtrim)
    trim: Optional[Expression] = default_field(func=pc.utf8_trim)
    characters: str = default_field('', description="trim options; by default trims whitespace")

    replace_slice: Optional[Expression] = default_field(func=pc.utf8_replace_slice)
    slice_codeunits: Optional[Expression] = default_field(func=pc.utf8_slice_codeunits)
    start: int = 0
    stop: Optional[int] = UNSET
    step: int = 1
    replacement: str = ''

    center: Optional[Expression] = default_field(func=pc.utf8_center)
    lpad: Optional[Expression] = default_field(func=pc.utf8_lpad)
    rpad: Optional[Expression] = default_field(func=pc.utf8_rpad)
    width: int = 0
    padding: str = ''

    def getfunc(self, name):
        if name.endswith('trim') and not self.characters:
            name += '_whitespace'
        return getattr(pc, 'utf8_' + name)


@strawberry.input(description="Binary functions.")
class Binary(Fields):
    length: Optional[Expression] = default_field(func=pc.binary_length)
    repeat: list[Expression] = default_field([], func=pc.binary_repeat)
    reverse: Optional[Expression] = default_field(func=pc.binary_reverse)

    join: list[Expression] = default_field([], func=pc.binary_join)
    join_element_wise: list[Expression] = default_field([], func=pc.binary_join_element_wise)
    null_handling: str = 'emit_null'
    null_replacement: str = ''

    replace_slice: Optional[Expression] = default_field(func=pc.binary_replace_slice)
    start: int = 0
    stop: int = 0
    replacement: bytes = b''

    prefix = 'binary_'


@strawberry.input(description="Match substring functions.")
class MatchSubstring(Fields):
    count_substring: Optional[Expression] = default_field(name='count', func=pc.count_substring)
    ends_with: Optional[Expression] = default_field(func=pc.ends_with)
    find_substring: Optional[Expression] = default_field(name='find', func=pc.find_substring)
    match_substring: Optional[Expression] = default_field(name='match', func=pc.match_substring)
    starts_with: Optional[Expression] = default_field(func=pc.starts_with)
    replace_substring: Optional[Expression] = default_field(
        name='replace', func=pc.replace_substring
    )
    split_pattern: Optional[Expression] = default_field(name='split', func=pc.split_pattern)
    extract: Optional[Expression] = default_field(func=pc.extract_regex)
    pattern: str = ''
    ignore_case: bool = False
    regex: bool = False
    replacement: str = ''
    max_replacements: Optional[int] = None
    max_splits: Optional[int] = None
    reverse: bool = False

    def getfunc(self, name):
        if name == 'split_pattern' and not self.pattern:
            name = 'utf8_split_whitespace'
        return getattr(pc, name + ('_regex' * self.regex))


@strawberry.input(description="Temporal functions.")
class Temporal(Fields):
    day: Optional[Expression] = default_field(func=pc.day)
    day_of_year: Optional[Expression] = default_field(func=pc.day_of_year)
    hour: Optional[Expression] = default_field(func=pc.hour)
    iso_week: Optional[Expression] = default_field(func=pc.iso_week)
    iso_year: Optional[Expression] = default_field(func=pc.iso_year)
    iso_calendar: Optional[Expression] = default_field(func=pc.iso_calendar)
    is_leap_year: Optional[Expression] = default_field(func=pc.is_leap_year)

    microsecond: Optional[Expression] = default_field(func=pc.microsecond)
    millisecond: Optional[Expression] = default_field(func=pc.millisecond)
    minute: Optional[Expression] = default_field(func=pc.minute)
    month: Optional[Expression] = default_field(func=pc.month)
    nanosecond: Optional[Expression] = default_field(func=pc.nanosecond)
    quarter: Optional[Expression] = default_field(func=pc.quarter)
    second: Optional[Expression] = default_field(func=pc.second)
    subsecond: Optional[Expression] = default_field(func=pc.subsecond)
    us_week: Optional[Expression] = default_field(func=pc.us_week)
    us_year: Optional[Expression] = default_field(func=pc.us_year)
    year: Optional[Expression] = default_field(func=pc.year)
    year_month_day: Optional[Expression] = default_field(func=pc.year_month_day)

    day_time_interval_between: list[Expression] = default_field(
        [], func=pc.day_time_interval_between
    )
    days_between: list[Expression] = default_field([], func=pc.days_between)
    hours_between: list[Expression] = default_field([], func=pc.hours_between)
    microseconds_between: list[Expression] = default_field([], func=pc.microseconds_between)
    milliseconds_between: list[Expression] = default_field([], func=pc.milliseconds_between)
    minutes_between: list[Expression] = default_field([], func=pc.minutes_between)
    month_day_nano_interval_between: list[Expression] = default_field(
        [], func=pc.month_day_nano_interval_between
    )
    month_interval_between: list[Expression] = default_field([], func=pc.month_interval_between)
    nanoseconds_between: list[Expression] = default_field([], func=pc.nanoseconds_between)
    quarters_between: list[Expression] = default_field([], func=pc.quarters_between)
    seconds_between: list[Expression] = default_field([], func=pc.seconds_between)
    weeks_between: list[Expression] = default_field([], func=pc.weeks_between)
    years_between: list[Expression] = default_field([], func=pc.years_between)

    ceil_temporal: Optional[Expression] = default_field(name='ceil', func=pc.ceil_temporal)
    floor_temporal: Optional[Expression] = default_field(name='floor', func=pc.floor_temporal)
    round_temporal: Optional[Expression] = default_field(name='round', func=pc.round_temporal)
    multiple: int = 1
    unit: str = 'day'
    week_starts_monday: bool = True
    ceil_is_strictly_greater: bool = False
    calendar_based_origin: bool = False

    week: Optional[Expression] = default_field(func=pc.week)
    count_from_zero: Optional[bool] = UNSET
    first_week_is_fully_in_year: bool = False

    day_of_week: Optional[Expression] = default_field(func=pc.day_of_week)
    week_start: int = 1

    strftime: Optional[Expression] = default_field(func=pc.strftime)
    strptime: Optional[Expression] = default_field(func=pc.strptime)
    format: str = '%Y-%m-%dT%H:%M:%S'
    locale: str = 'C'
    error_is_null: bool = False

    assume_timezone: Optional[Expression] = default_field(func=pc.assume_timezone)
    timezone: str = ''
    ambiguous: str = 'raise'
    nonexistent: str = 'raise'


@strawberry.input(description="Set lookup functions.")
class SetLookup(Fields):
    index_in: list[Expression] = default_field([], func=pc.index_in)
    digitize: list[Expression] = default_field(
        [],
        description="numpy [digitize](https://numpy.org/doc/stable/reference/generated/numpy.digitize.html)",
    )
    skip_nulls: bool = False
    right: bool = False

    def to_fields(self) -> Iterable[ds.Expression]:
        if self.index_in:
            values, value_set = [expr.to_arrow() for expr in self.index_in]
            yield pc.index_in(values, pa.array(value_set), skip_nulls=self.skip_nulls)
        if self.digitize:
            values, value_set = [expr.to_arrow() for expr in self.digitize]
            args = values.cast('float64'), list(map(float, value_set)), self.right  # type: ignore
            yield ds.Expression._call('digitize', list(args))


@strawberry.input(description="List array functions.")
class Lists(Fields):
    element: list[Expression] = default_field([], func=pc.list_element)
    value_length: Optional[Expression] = default_field(func=pc.list_value_length)
    # user defined functions
    all: Optional[Expression] = default_field(func=pc.all)
    any: Optional[Expression] = default_field(func=pc.any)

    slice: Optional[Expression] = default_field(func=pc.list_slice)
    start: int = 0
    stop: Optional[int] = None
    step: int = 1
    return_fixed_size_list: Optional[bool] = None

    prefix = 'list_'

    def getfunc(self, name):
        if name in ('element', 'value_length', 'slice'):  # built-ins
            return super().getfunc(name)
        return lambda *args: ds.Expression._call(self.prefix + name, list(args))
