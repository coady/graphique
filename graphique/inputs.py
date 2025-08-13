"""
GraphQL input types.
"""

from __future__ import annotations
import functools
import inspect
import operator
from collections.abc import Callable, Iterable
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Generic, TypeVar
import ibis
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

T = TypeVar('T')


class links:
    compute = 'https://arrow.apache.org/docs/python/api/compute.html'
    type = '[arrow type](https://arrow.apache.org/docs/python/api/datatypes.html)'


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
    default=UNSET, func: Callable | None = None, nullable: bool = False, **kwargs
) -> StrawberryField:
    """Use dataclass `default_factory` for `UNSET` or mutables."""
    if func is not None:
        kwargs['description'] = inspect.getdoc(func).splitlines()[0]  # type: ignore
    if not nullable and default is UNSET:
        kwargs.setdefault('directives', []).append(optional())
    return strawberry.field(default_factory=type(default), **kwargs)


@strawberry.input(description="predicates for scalars")
class Filter(Generic[T]):
    eq: list[T | None] | None = default_field(description="== or `isin`", nullable=True)
    ne: T | None = default_field(description="!=", nullable=True)
    lt: T | None = default_field(description="<")
    le: T | None = default_field(description="<=")
    gt: T | None = default_field(description=r"\>")
    ge: T | None = default_field(description=r"\>=")

    def __iter__(self) -> Iterable[tuple]:
        for name, value in self.__dict__.items():
            if value is not UNSET:
                yield name, value

    @classmethod
    def resolve_args(cls, types: dict) -> Iterable[StrawberryArgument]:
        """Generate dynamically resolved arguments for filter field."""
        for name in types:
            annotation = StrawberryAnnotation(cls[types[name]])  # type: ignore
            if types[name] not in (list, dict):
                yield StrawberryArgument(name, name, annotation, default={})
        annotation = StrawberryAnnotation(IExpression | None)
        yield StrawberryArgument('where', None, annotation, default=None)

    @staticmethod
    def to_exprs(**queries: Filter) -> Iterable[ibis.Deferred]:
        """Transform query syntax into ibis expressions."""
        for name, query in queries.items():
            field = ibis._[name]
            for op, value in query:  # type: ignore
                isin = op == 'eq' and isinstance(value, list)
                yield field.isin(value) if isin else getattr(operator, op)(field, value)

    @staticmethod
    def to_arrow(**queries: Filter | dict) -> ds.Expression | None:
        """Transform query syntax into an arrow expression."""
        exprs = []
        for name, query in queries.items():
            field = pc.field(name)
            for op, value in dict(query).items():  # type: ignore
                if op == 'eq' and isinstance(value, list):
                    exprs.append(pc.is_in(field, pa.array(value)))
                else:
                    exprs.append(getattr(operator, op)(field, value))
        return functools.reduce(operator.and_, exprs or [None])


@strawberry.input(description=f"name and optional alias for [compute functions]({links.compute})")
class Aggregate:
    name: str = strawberry.field(description="column name")
    alias: str = strawberry.field(default='', description="output column name")

    def to_ibis(self, func: str, **options) -> tuple:
        return (self.alias or self.name), getattr(ibis._[self.name], func)(**options)


@strawberry.input(description=f"options for tdigest [aggregation]({links.compute}#aggregations)")
class CollectAggregate(Aggregate):
    distinct: bool = False

    def to_ibis(self, func: str) -> tuple:  # type: ignore
        return super().to_ibis(func, distinct=self.distinct)


@strawberry.input(description="Aggregation functions.")
class Aggregates:
    all: list[Aggregate] = default_field([], func=ibis.expr.types.BooleanColumn.all)
    any: list[Aggregate] = default_field([], func=ibis.expr.types.BooleanColumn.any)
    collect: list[CollectAggregate] = default_field([], func=ibis.expr.types.Column.collect)
    first: list[Aggregate] = default_field([], func=ibis.expr.types.Column.first)
    last: list[Aggregate] = default_field([], func=ibis.expr.types.Column.last)
    max: list[Aggregate] = default_field([], func=ibis.expr.types.Column.max)
    mean: list[Aggregate] = default_field([], func=ibis.expr.types.NumericColumn.mean)
    median: list[Aggregate] = default_field([], func=ibis.expr.types.Column.median)
    min: list[Aggregate] = default_field([], func=ibis.expr.types.Column.min)
    std: list[Aggregate] = default_field([], func=ibis.expr.types.NumericColumn.std)
    sum: list[Aggregate] = default_field([], func=ibis.expr.types.NumericColumn.sum)
    var: list[Aggregate] = default_field([], func=ibis.expr.types.NumericColumn.var)

    def __iter__(self) -> Iterable[tuple]:
        for name, aggs in self.__dict__.items():
            for agg in aggs:
                yield agg.to_ibis(name)


@use_doc(strawberry.input)
class Expression:
    """[Dataset expression](https://arrow.apache.org/docs/python/generated/pyarrow.dataset.Expression.html)
    used for scanning.

    Expects one of: a field `name`, a scalar, or an operator with expressions. Single values can be passed for an
    [input `List`](https://spec.graphql.org/October2021/#sec-List.Input-Coercion).
    * `eq` with a list scalar is equivalent to `isin`
    * `eq` with a `null` scalar is equivalent `is_null`
    * `ne` with a `null` scalar is equivalent to `is_valid`
    """

    name: list[str] = default_field([], description="field name(s)")
    cast: str = strawberry.field(default='', description=f"cast as {links.type}")
    safe: bool = strawberry.field(default=True, description="check for conversion errors on cast")
    value: JSON | None = default_field(
        description="JSON scalar; also see typed scalars", nullable=True
    )

    def to_arrow(self) -> ds.Expression | None:
        """Transform GraphQL expression into a dataset expression."""
        fields = []
        if self.name:
            fields.append(pc.field(*self.name))
        if not fields:
            return None
        (field,) = fields
        cast = self.cast and isinstance(field, ds.Expression)
        return field.cast(self.cast, self.safe) if cast else field


@strawberry.input(description="an `Expression` with an optional alias")
class Projection(Expression):
    alias: str = strawberry.field(default='', description="name of projected column")


@use_doc(strawberry.input)
class IExpression:
    """[Ibis expression](https://ibis-project.org/reference/#expression-api)."""

    name: str = strawberry.field(default='', description="field name")
    value: JSON | None = default_field(description="JSON scalar", nullable=True)
    row_number: None = default_field(func=ibis.row_number)

    base64: bytes | None = default_field(description="binary scalar")
    date_: date | None = default_field(description="date scalar", name='date')
    datetime_: datetime | None = default_field(description="datetime scalar", name='datetime')
    decimal: Decimal | None = default_field(description="decimal scalar")
    duration: timedelta | None = default_field(description="duration scalar")
    time_: time | None = default_field(description="time scalar", name='time')

    eq: list[IExpression] = default_field([], description="==")
    ne: list[IExpression] = default_field([], description="!=")
    lt: list[IExpression] = default_field([], description="<")
    le: list[IExpression] = default_field([], description="<=")
    gt: list[IExpression] = default_field([], description=r"\>")
    ge: list[IExpression] = default_field([], description=r"\>=")
    isin: list[IExpression] = default_field([], func=ibis.expr.types.Column.isin)

    inv: IExpression | None = default_field(description="~")
    and_: list[IExpression] = default_field([], name='and', description="&")
    or_: list[IExpression] = default_field([], name='or', description="|")
    xor: list[IExpression] = default_field([], description="^")

    add: list[IExpression] = default_field([], description="+")
    sub: list[IExpression] = default_field([], description="-")
    mul: list[IExpression] = default_field([], description="*")
    truediv: list[IExpression] = default_field([], name='div', description="/")

    coalesce: list[IExpression] = default_field([], func=ibis.expr.types.Column.coalesce)
    cume_dist: IExpression | None = default_field(func=ibis.expr.types.Column.cume_dist)
    cummax: IExpression | None = default_field(func=ibis.expr.types.Column.cummax)
    cummin: IExpression | None = default_field(func=ibis.expr.types.Column.cummin)
    dense_rank: IExpression | None = default_field(func=ibis.expr.types.Column.dense_rank)
    ifelse: list[IExpression] = default_field([], func=ibis.expr.types.BooleanColumn.ifelse)
    percent_rank: IExpression | None = default_field(func=ibis.expr.types.Column.percent_rank)
    rank: IExpression | None = default_field(func=ibis.expr.types.Column.rank)

    array: Arrays | None = default_field(description="array value functions")
    numeric: Numeric | None = default_field(description="numeric functions")
    string: Strings | None = default_field(description="string functions")
    temporal: Temporal | None = default_field(description="temporal functions")

    def items(self) -> Iterable[tuple]:
        for name, value in self.__dict__.items():
            if isinstance(value, IExpression):
                yield name, [value.to_ibis()]
            elif isinstance(value, list) and value and isinstance(value[0], IExpression):
                yield name, map(IExpression.to_ibis, value)

    def __iter__(self) -> Iterable[ibis.Deferred]:
        if self.name:
            yield ibis._[self.name]
        scalars = self.base64, self.date_, self.datetime_, self.decimal, self.duration, self.time_
        for scalar in (self.value,) + scalars:
            if scalar is not UNSET:
                yield scalar
        if self.row_number is not UNSET:
            yield ibis.row_number()
        for name, (expr, *args) in self.items():
            match name:
                case 'eq' | 'ne' | 'lt' | 'le' | 'gt' | 'ge' | 'inv' | 'and_' | 'or_' | 'xor':
                    yield getattr(operator, name)(expr, *args)
                case 'add' | 'sub' | 'mul' | 'truediv':
                    yield getattr(operator, name)(expr, *args)
                case _:
                    yield getattr(expr, name)(*args)
        for field in (self.array, self.numeric, self.string, self.temporal):
            if field:
                yield from field  # type: ignore

    def to_ibis(self) -> ibis.Deferred | None:
        fields = list(self) or [None]  # type: ignore
        if len(fields) == 1:
            return fields[0]
        raise ValueError(f"conflicting inputs: {', '.join(map(str, fields))}")


@strawberry.input(description="an `IExpression` with an optional alias")
class IProjection(IExpression):
    alias: str = strawberry.field(default='', description="name of projected column")


@strawberry.input(description="Array value functions.")
class Arrays:
    alls: IExpression | None = default_field(func=ibis.expr.types.ArrayValue.alls)
    anys: IExpression | None = default_field(func=ibis.expr.types.ArrayValue.anys)
    length: IExpression | None = default_field(func=ibis.expr.types.ArrayValue.length)
    maxs: IExpression | None = default_field(func=ibis.expr.types.ArrayValue.maxs)
    means: IExpression | None = default_field(func=ibis.expr.types.ArrayValue.means)
    modes: IExpression | None = default_field(func=ibis.expr.types.ArrayValue.modes)
    mins: IExpression | None = default_field(func=ibis.expr.types.ArrayValue.mins)
    sort: IExpression | None = default_field(func=ibis.expr.types.ArrayValue.sort)
    sums: IExpression | None = default_field(func=ibis.expr.types.ArrayValue.sums)
    unique: IExpression | None = default_field(func=ibis.expr.types.ArrayValue.unique)

    index: list[IExpression] = default_field([], func=ibis.expr.types.ArrayValue.index)

    slice: IExpression | None = default_field(description="array slice")
    value: IExpression | None = default_field(description="value at offset")
    offset: int = 0
    limit: int | None = None

    def __iter__(self) -> Iterable[ibis.Deferred]:
        for name, (expr, *args) in IExpression.items(self):  # type: ignore
            match name:
                case 'slice':
                    yield expr[self.offset :][: self.limit]
                case 'value':
                    yield expr[self.offset]
                case _:
                    yield getattr(expr, name)(*args)


@strawberry.input(description="Numeric functions.")
class Numeric:
    abs: IExpression | None = default_field(func=ibis.expr.types.NumericColumn.abs)
    acos: IExpression | None = default_field(func=ibis.expr.types.NumericColumn.acos)
    asin: IExpression | None = default_field(func=ibis.expr.types.NumericColumn.asin)
    atan: IExpression | None = default_field(func=ibis.expr.types.NumericColumn.atan)
    atan2: list[IExpression] = default_field([], func=ibis.expr.types.NumericColumn.atan2)
    ceil: IExpression | None = default_field(func=ibis.expr.types.NumericColumn.ceil)
    cos: IExpression | None = default_field(func=ibis.expr.types.NumericColumn.cos)
    exp: IExpression | None = default_field(func=ibis.expr.types.NumericColumn.exp)
    floor: IExpression | None = default_field(func=ibis.expr.types.NumericColumn.floor)
    isinf: IExpression | None = default_field(func=ibis.expr.types.FloatingColumn.isinf)
    isnan: IExpression | None = default_field(func=ibis.expr.types.FloatingColumn.isnan)
    log: list[IExpression] = default_field([], func=ibis.expr.types.NumericColumn.log)
    negate: IExpression | None = default_field(func=ibis.expr.types.NumericColumn.negate)
    round: list[IExpression] = default_field([], func=ibis.expr.types.NumericColumn.round)
    sign: IExpression | None = default_field(func=ibis.expr.types.NumericColumn.sign)
    sin: IExpression | None = default_field(func=ibis.expr.types.NumericColumn.sin)
    sqrt: IExpression | None = default_field(func=ibis.expr.types.NumericColumn.sqrt)
    tan: IExpression | None = default_field(func=ibis.expr.types.NumericColumn.tan)

    bucket: IExpression | None = default_field(func=ibis.expr.types.NumericColumn.bucket)
    cummean: IExpression | None = default_field(func=ibis.expr.types.NumericColumn.cummean)
    cumsum: IExpression | None = default_field(func=ibis.expr.types.NumericColumn.cumsum)

    buckets: list[JSON] = default_field([])
    closed: str = 'left'
    close_extreme: bool = True
    include_under: bool = False
    include_over: bool = False

    def __iter__(self) -> Iterable[ibis.Deferred]:
        for name, (expr, *args) in IExpression.items(self):  # type: ignore
            match name:
                case 'bucket':
                    yield expr.bucket(
                        self.buckets,
                        closed=self.closed,
                        close_extreme=self.close_extreme,
                        include_under=self.include_under,
                        include_over=self.include_over,
                    )
                case _:
                    yield getattr(expr, name)(*args)


@strawberry.input(description="String functions.")
class Strings:
    capitalize: IExpression | None = default_field(func=ibis.expr.types.StringColumn.capitalize)
    contains: list[IExpression] = default_field([], func=ibis.expr.types.StringColumn.contains)
    endswith: list[IExpression] = default_field([], func=ibis.expr.types.StringColumn.endswith)
    find: list[IExpression] = default_field([], func=ibis.expr.types.StringColumn.find)
    length: IExpression | None = default_field(func=ibis.expr.types.StringColumn.length)
    lower: IExpression | None = default_field(func=ibis.expr.types.StringColumn.lower)
    lpad: list[IExpression] = default_field([], func=ibis.expr.types.StringColumn.lpad)
    lstrip: list[IExpression] = default_field([], func=ibis.expr.types.StringColumn.lstrip)
    re_extract: list[IExpression] = default_field([], func=ibis.expr.types.StringColumn.re_extract)
    re_search: list[IExpression] = default_field([], func=ibis.expr.types.StringColumn.re_search)
    re_split: list[IExpression] = default_field([], func=ibis.expr.types.StringColumn.re_split)
    replace: list[IExpression] = default_field([], func=ibis.expr.types.StringColumn.replace)
    reverse: IExpression | None = default_field(func=ibis.expr.types.StringColumn.reverse)
    rpad: list[IExpression] = default_field([], func=ibis.expr.types.StringColumn.rpad)
    rstrip: list[IExpression] = default_field([], func=ibis.expr.types.StringColumn.rstrip)
    split: list[IExpression] = default_field([], func=ibis.expr.types.StringColumn.split)
    startswith: list[IExpression] = default_field([], func=ibis.expr.types.StringColumn.startswith)
    strip: list[IExpression] = default_field([], func=ibis.expr.types.StringColumn.strip)
    upper: IExpression | None = default_field(func=ibis.expr.types.StringColumn.upper)

    def __iter__(self) -> Iterable[ibis.Deferred]:
        for name, (expr, *args) in IExpression.items(self):  # type: ignore
            yield getattr(expr, name)(*args)


@strawberry.input(description="Temporal functions.")
class Temporal:
    date: IExpression | None = default_field(func=ibis.expr.types.TimestampColumn.date)
    day: IExpression | None = default_field(func=ibis.expr.types.TimestampColumn.day)
    day_of_year: IExpression | None = default_field(
        func=ibis.expr.types.TimestampColumn.day_of_year
    )
    delta: list[IExpression] = default_field([], func=ibis.expr.types.TimestampColumn.delta)
    epoch_seconds: IExpression | None = default_field(
        func=ibis.expr.types.TimestampColumn.epoch_seconds
    )
    hour: IExpression | None = default_field(func=ibis.expr.types.TimestampColumn.hour)
    microsecond: IExpression | None = default_field(
        func=ibis.expr.types.TimestampColumn.microsecond
    )
    millisecond: IExpression | None = default_field(
        func=ibis.expr.types.TimestampColumn.millisecond
    )
    minute: IExpression | None = default_field(func=ibis.expr.types.TimestampColumn.minute)
    month: IExpression | None = default_field(func=ibis.expr.types.TimestampColumn.month)
    quarter: IExpression | None = default_field(func=ibis.expr.types.TimestampColumn.quarter)
    second: IExpression | None = default_field(func=ibis.expr.types.TimestampColumn.second)
    strftime: IExpression | None = default_field(func=ibis.expr.types.TimestampColumn.strftime)
    time: IExpression | None = default_field(func=ibis.expr.types.TimestampColumn.time)
    truncate: IExpression | None = default_field(func=ibis.expr.types.TimestampColumn.truncate)
    week_of_year: IExpression | None = default_field(
        func=ibis.expr.types.TimestampColumn.week_of_year
    )
    year: IExpression | None = default_field(func=ibis.expr.types.TimestampColumn.year)

    format_str: str = ''
    unit: str = ''

    def __iter__(self) -> Iterable[ibis.Deferred]:
        for name, (expr, *args) in IExpression.items(self):  # type: ignore
            match name:
                case 'delta':
                    yield expr.delta(*args, unit=self.unit)
                case 'truncate':
                    yield expr.truncate(self.unit)
                case 'strftime':
                    yield expr.strftime(self.format_str)
                case _:
                    yield getattr(expr, name)(*args)
