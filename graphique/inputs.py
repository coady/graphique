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
from strawberry.scalars import JSON
from strawberry.schema_directive import Location
from strawberry.types.arguments import StrawberryArgument
from strawberry.types.field import StrawberryField

from .core import getitems, links, order_key

T = TypeVar('T')


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


@use_doc(strawberry.schema_directive, locations=[Location.OBJECT, Location.FIELD_DEFINITION])
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


@strawberry.input(description="a schema field")
class Field:
    name: str
    type: str


@strawberry.input(description="predicates for scalars")
class Filter(Generic[T]):
    eq: list[T] | None = default_field(description="== or `isin`", nullable=True)
    ne: list[T] | None = default_field(description="!= or `notin`", nullable=True)
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
        annotation = StrawberryAnnotation(Expression | None)
        yield StrawberryArgument('where', None, annotation, default=None)

    @staticmethod
    def to_exprs(**queries: Filter) -> Iterable[ibis.Deferred]:
        """Transform query syntax into ibis expressions."""
        for name, query in queries.items():
            field = ibis._[name]
            for op, value in query:  # type: ignore
                match value, op:
                    case list(), 'eq':
                        yield field.isin(value)
                    case list(), 'ne':
                        yield field.notin(value)
                    case _:
                        yield getattr(operator, op)(field, value)

    @staticmethod
    def to_arrow(**queries: Filter | dict) -> ds.Expression | None:
        """Transform query syntax into an arrow expression."""
        exprs = []
        for name, query in queries.items():
            field = pc.field(name)
            for op, value in dict(query).items():  # type: ignore
                if isinstance(value, list):
                    expr = pc.is_in(field, pa.array(value))
                    exprs.append(pc.invert(expr) if op == 'ne' else expr)
                else:
                    exprs.append(getattr(operator, op)(field, value))
        return functools.reduce(operator.and_, exprs or [None])


@strawberry.input(description="name and optional alias for aggregation")
class Aggregate:
    name: str = strawberry.field(description="column name")
    alias: str = strawberry.field(default='', description="output column name")
    where: Expression | None = None

    def to_ibis(self, func: str, **options) -> tuple:
        options['where'] = self.where and self.where.to_ibis()
        return (self.alias or self.name), getattr(ibis._[self.name], func)(**options)


@strawberry.input
class UniqueAggregate(Aggregate):
    approx: bool = False

    def to_ibis(self, func: str, **options) -> tuple:  # type: ignore
        return super().to_ibis('approx_' + func if self.approx else func, **options)


@strawberry.input
class OrderAggregate(Aggregate):
    order_by: list[str] = default_field([])
    include_null: bool = False

    def to_ibis(self, func: str, **options) -> tuple:
        options.update(
            order_by=list(map(order_key, self.order_by)) or None,
            include_null=self.include_null,
        )
        return super().to_ibis(func, **options)


@strawberry.input
class VarAggregate(Aggregate):
    how: str = 'sample'

    def to_ibis(self, func: str) -> tuple:  # type: ignore
        return super().to_ibis(func, how=self.how)


@strawberry.input
class QuantileAggregate(UniqueAggregate):
    q: float = 0.5

    def to_ibis(self, func: str) -> tuple:  # type: ignore
        return super().to_ibis(func, quantile=self.q)


@strawberry.input
class CollectAggregate(OrderAggregate):
    distinct: bool = False

    def to_ibis(self, func: str) -> tuple:  # type: ignore
        return super().to_ibis(func, distinct=self.distinct)


@strawberry.input(description=f"aggregation [expressions]({links.ref}/expression-generic)")
class Aggregates:
    all: list[Aggregate] = default_field([], func=ibis.expr.types.BooleanColumn.all)
    any: list[Aggregate] = default_field([], func=ibis.expr.types.BooleanColumn.any)
    collect: list[CollectAggregate] = default_field([], func=ibis.Column.collect)
    count: list[Aggregate] = default_field([], func=ibis.Column.count)
    first: list[OrderAggregate] = default_field([], func=ibis.Column.first)
    last: list[OrderAggregate] = default_field([], func=ibis.Column.last)
    max: list[Aggregate] = default_field([], func=ibis.Column.max)
    mean: list[Aggregate] = default_field([], func=ibis.expr.types.NumericColumn.mean)
    min: list[Aggregate] = default_field([], func=ibis.Column.min)
    nunique: list[UniqueAggregate] = default_field([], func=ibis.Column.nunique)
    quantile: list[QuantileAggregate] = default_field([], func=ibis.Column.quantile)
    std: list[VarAggregate] = default_field([], func=ibis.expr.types.NumericColumn.std)
    sum: list[Aggregate] = default_field([], func=ibis.expr.types.NumericColumn.sum)
    var: list[VarAggregate] = default_field([], func=ibis.expr.types.NumericColumn.var)

    def __iter__(self) -> Iterable[tuple]:
        for name, aggs in self.__dict__.items():
            for agg in aggs:
                yield agg.to_ibis(name)


@strawberry.input(description="typed scalars")
class Scalars:
    base64: bytes | None = default_field(description="binary scalar")
    date: date | None = default_field(description="date scalar")
    datetime: datetime | None = default_field(description="datetime scalar")
    decimal: Decimal | None = default_field(description="decimal scalar")
    duration: timedelta | None = default_field(description="duration scalar")
    time: time | None = default_field(description="time scalar")

    def __iter__(self):
        for value in self.__dict__.values():
            if value is not UNSET:
                yield value


@strawberry.input(description=f"[expression API]({links.ref}/#expression-api)")
class Expression:
    name: list[str] = default_field([], description="column name(s)")
    value: JSON | None = default_field(description="JSON scalar", nullable=True)
    scalar: Scalars | None = default_field(description="typed scalar")
    row_number: None = default_field(func=ibis.row_number)

    eq: list[Expression] = default_field([], description="==")
    ne: list[Expression] = default_field([], description="!=")
    lt: list[Expression] = default_field([], description="<")
    le: list[Expression] = default_field([], description="<=")
    gt: list[Expression] = default_field([], description=r"\>")
    ge: list[Expression] = default_field([], description=r"\>=")
    isin: list[Expression] = default_field([], func=ibis.Column.isin)
    notin: list[Expression] = default_field([], func=ibis.Column.notin)

    inv: Expression | None = default_field(description="~")
    and_: list[Expression] = default_field([], name='and', description="&")
    or_: list[Expression] = default_field([], name='or', description="|")
    xor: list[Expression] = default_field([], description="^")

    add: list[Expression] = default_field([], description="+")
    sub: list[Expression] = default_field([], description="-")
    mul: list[Expression] = default_field([], description="*")
    truediv: list[Expression] = default_field([], name='div', description="/")

    coalesce: list[Expression] = default_field([], func=ibis.Column.coalesce)
    cume_dist: Expression | None = default_field(func=ibis.Column.cume_dist)
    cummax: Expression | None = default_field(func=ibis.Column.cummax)
    cummin: Expression | None = default_field(func=ibis.Column.cummin)
    dense_rank: Expression | None = default_field(func=ibis.Column.dense_rank)
    ifelse: list[Expression] = default_field([], func=ibis.expr.types.BooleanColumn.ifelse)
    percent_rank: Expression | None = default_field(func=ibis.Column.percent_rank)
    rank: Expression | None = default_field(func=ibis.Column.rank)

    array: Arrays | None = default_field(description="array value functions")
    numeric: Numeric | None = default_field(description="numeric functions")
    string: Strings | None = default_field(description="string functions")
    temporal: Temporal | None = default_field(description="temporal functions")
    window: Window | None = default_field(description="window functions")

    def items(self) -> Iterable[tuple]:
        for name, value in self.__dict__.items():
            if isinstance(value, Expression):
                yield name, [value.to_ibis()]
            elif isinstance(value, list) and value and isinstance(value[0], Expression):
                yield name, map(Expression.to_ibis, value)

    def __iter__(self) -> Iterable[ibis.Deferred]:
        if self.name:
            yield getitems(ibis._, *self.name)
        if self.value is not UNSET:
            yield self.value
        if self.scalar:
            yield from self.scalar
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
        for field in (self.array, self.numeric, self.string, self.temporal, self.window):
            if field:
                yield from field  # type: ignore

    def to_ibis(self) -> ibis.Deferred | None:
        fields = list(self) or [None]  # type: ignore
        if len(fields) == 1:
            return fields[0]
        raise ValueError(f"conflicting inputs: {', '.join(map(str, fields))}")


@strawberry.input(description="an `Expression` with an optional alias")
class Projection(Expression):
    alias: str = strawberry.field(default='', description="name of projected column")

    def to_ibis(self) -> tuple:
        name = self.alias or '.'.join(self.name)
        if name:
            return name, super().to_ibis()
        raise ValueError(f"projected fields require a name or alias: {self}")


@strawberry.input(description=f"array [expressions]({links.ref}/expression-collections)")
class Arrays:
    alls: Expression | None = default_field(func=ibis.expr.types.ArrayValue.alls)
    anys: Expression | None = default_field(func=ibis.expr.types.ArrayValue.anys)
    length: Expression | None = default_field(func=ibis.expr.types.ArrayValue.length)
    maxs: Expression | None = default_field(func=ibis.expr.types.ArrayValue.maxs)
    means: Expression | None = default_field(func=ibis.expr.types.ArrayValue.means)
    modes: Expression | None = default_field(func=ibis.expr.types.ArrayValue.modes)
    mins: Expression | None = default_field(func=ibis.expr.types.ArrayValue.mins)
    sort: Expression | None = default_field(func=ibis.expr.types.ArrayValue.sort)
    sums: Expression | None = default_field(func=ibis.expr.types.ArrayValue.sums)
    unique: Expression | None = default_field(func=ibis.expr.types.ArrayValue.unique)

    index: list[Expression] = default_field([], func=ibis.expr.types.ArrayValue.index)

    slice: Expression | None = default_field(description="array slice")
    value: Expression | None = default_field(description="value at offset")
    offset: int = 0
    limit: int | None = None

    def __iter__(self) -> Iterable[ibis.Deferred]:
        for name, (expr, *args) in Expression.items(self):  # type: ignore
            match name:
                case 'slice':
                    yield expr[self.offset :][: self.limit]
                case 'value':
                    yield expr[self.offset]
                case _:
                    yield getattr(expr, name)(*args)


@strawberry.input(description=f"numeric [expressions]({links.ref}/expression-numeric)")
class Numeric:
    abs: Expression | None = default_field(func=ibis.expr.types.NumericColumn.abs)
    acos: Expression | None = default_field(func=ibis.expr.types.NumericColumn.acos)
    asin: Expression | None = default_field(func=ibis.expr.types.NumericColumn.asin)
    atan: Expression | None = default_field(func=ibis.expr.types.NumericColumn.atan)
    atan2: list[Expression] = default_field([], func=ibis.expr.types.NumericColumn.atan2)
    ceil: Expression | None = default_field(func=ibis.expr.types.NumericColumn.ceil)
    cos: Expression | None = default_field(func=ibis.expr.types.NumericColumn.cos)
    exp: Expression | None = default_field(func=ibis.expr.types.NumericColumn.exp)
    floor: Expression | None = default_field(func=ibis.expr.types.NumericColumn.floor)
    isinf: Expression | None = default_field(func=ibis.expr.types.FloatingColumn.isinf)
    isnan: Expression | None = default_field(func=ibis.expr.types.FloatingColumn.isnan)
    log: list[Expression] = default_field([], func=ibis.expr.types.NumericColumn.log)
    negate: Expression | None = default_field(func=ibis.expr.types.NumericColumn.negate)
    round: list[Expression] = default_field([], func=ibis.expr.types.NumericColumn.round)
    sign: Expression | None = default_field(func=ibis.expr.types.NumericColumn.sign)
    sin: Expression | None = default_field(func=ibis.expr.types.NumericColumn.sin)
    sqrt: Expression | None = default_field(func=ibis.expr.types.NumericColumn.sqrt)
    tan: Expression | None = default_field(func=ibis.expr.types.NumericColumn.tan)

    bucket: Expression | None = default_field(func=ibis.expr.types.NumericColumn.bucket)
    cummean: Expression | None = default_field(func=ibis.expr.types.NumericColumn.cummean)
    cumsum: Expression | None = default_field(func=ibis.expr.types.NumericColumn.cumsum)

    buckets: list[JSON] = default_field([])
    closed: str = 'left'
    close_extreme: bool = True
    include_under: bool = False
    include_over: bool = False

    def __iter__(self) -> Iterable[ibis.Deferred]:
        for name, (expr, *args) in Expression.items(self):  # type: ignore
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


@strawberry.input(description=f"string [expressions]({links.ref}/expression-strings)")
class Strings:
    capitalize: Expression | None = default_field(func=ibis.expr.types.StringColumn.capitalize)
    contains: list[Expression] = default_field([], func=ibis.expr.types.StringColumn.contains)
    endswith: list[Expression] = default_field([], func=ibis.expr.types.StringColumn.endswith)
    find: list[Expression] = default_field([], func=ibis.expr.types.StringColumn.find)
    length: Expression | None = default_field(func=ibis.expr.types.StringColumn.length)
    lower: Expression | None = default_field(func=ibis.expr.types.StringColumn.lower)
    lpad: list[Expression] = default_field([], func=ibis.expr.types.StringColumn.lpad)
    lstrip: list[Expression] = default_field([], func=ibis.expr.types.StringColumn.lstrip)
    re_extract: list[Expression] = default_field([], func=ibis.expr.types.StringColumn.re_extract)
    re_search: list[Expression] = default_field([], func=ibis.expr.types.StringColumn.re_search)
    re_split: list[Expression] = default_field([], func=ibis.expr.types.StringColumn.re_split)
    replace: list[Expression] = default_field([], func=ibis.expr.types.StringColumn.replace)
    reverse: Expression | None = default_field(func=ibis.expr.types.StringColumn.reverse)
    rpad: list[Expression] = default_field([], func=ibis.expr.types.StringColumn.rpad)
    rstrip: list[Expression] = default_field([], func=ibis.expr.types.StringColumn.rstrip)
    split: list[Expression] = default_field([], func=ibis.expr.types.StringColumn.split)
    startswith: list[Expression] = default_field([], func=ibis.expr.types.StringColumn.startswith)
    strip: list[Expression] = default_field([], func=ibis.expr.types.StringColumn.strip)
    upper: Expression | None = default_field(func=ibis.expr.types.StringColumn.upper)

    def __iter__(self) -> Iterable[ibis.Deferred]:
        for name, (expr, *args) in Expression.items(self):  # type: ignore
            yield getattr(expr, name)(*args)


@strawberry.input(description=f"temporal [expressions]({links.ref}/expression-temporal)")
class Temporal:
    date: Expression | None = default_field(func=ibis.expr.types.TimestampColumn.date)
    day: Expression | None = default_field(func=ibis.expr.types.TimestampColumn.day)
    day_of_year: Expression | None = default_field(func=ibis.expr.types.TimestampColumn.day_of_year)
    delta: list[Expression] = default_field([], func=ibis.expr.types.TimestampColumn.delta)
    epoch_seconds: Expression | None = default_field(
        func=ibis.expr.types.TimestampColumn.epoch_seconds
    )
    hour: Expression | None = default_field(func=ibis.expr.types.TimestampColumn.hour)
    microsecond: Expression | None = default_field(func=ibis.expr.types.TimestampColumn.microsecond)
    millisecond: Expression | None = default_field(func=ibis.expr.types.TimestampColumn.millisecond)
    minute: Expression | None = default_field(func=ibis.expr.types.TimestampColumn.minute)
    month: Expression | None = default_field(func=ibis.expr.types.TimestampColumn.month)
    quarter: Expression | None = default_field(func=ibis.expr.types.TimestampColumn.quarter)
    second: Expression | None = default_field(func=ibis.expr.types.TimestampColumn.second)
    strftime: Expression | None = default_field(func=ibis.expr.types.TimestampColumn.strftime)
    time: Expression | None = default_field(func=ibis.expr.types.TimestampColumn.time)
    truncate: Expression | None = default_field(func=ibis.expr.types.TimestampColumn.truncate)
    week_of_year: Expression | None = default_field(
        func=ibis.expr.types.TimestampColumn.week_of_year
    )
    year: Expression | None = default_field(func=ibis.expr.types.TimestampColumn.year)

    format_str: str = ''
    unit: str = ''

    def __iter__(self) -> Iterable[ibis.Deferred]:
        for name, (expr, *args) in Expression.items(self):  # type: ignore
            match name:
                case 'delta':
                    yield expr.delta(*args, unit=self.unit)
                case 'truncate':
                    yield expr.truncate(self.unit)
                case 'strftime':
                    yield expr.strftime(self.format_str)
                case _:
                    yield getattr(expr, name)(*args)


@strawberry.input(
    description=f"window [expressions]({links.ref}/expression-tables.html#ibis.window)"
)
class Window:
    lag: Expression | None = default_field(func=ibis.Column.lag)
    lead: Expression | None = default_field(func=ibis.Column.lead)

    eq: Expression | None = default_field(description="pairwise ==")
    ne: Expression | None = default_field(description="pairwise !=")
    lt: Expression | None = default_field(description="pairwise <")
    le: Expression | None = default_field(description="pairwise <=")
    gt: Expression | None = default_field(description=r"pairwise \>")
    ge: Expression | None = default_field(description=r"pairwise \>=")
    sub: Expression | None = default_field(description="pairwise -")

    offset: int = 1
    default: JSON | None = default_field(None, description="default JSON scalar")
    scalar: Scalars | None = default_field(description="default typed scalar")

    def __iter__(self) -> Iterable[ibis.Deferred]:
        (default,) = self.scalar or [self.default]
        for name, (expr,) in Expression.items(self):  # type: ignore
            match name:
                case 'lag' | 'lead':
                    yield getattr(expr, name)(self.offset, default)
                case _:
                    yield getattr(operator, name)(expr, expr.lag(self.offset)).fill_null(default)
