"""
Service related utilities which don't require knowledge of the schema.
"""
import itertools
from datetime import datetime
import strawberry.asgi
from starlette.middleware import base
from .models import Column


def references(node):
    """Generate every possible column reference."""
    if hasattr(node, 'name'):
        yield node.name.value
    value = getattr(node, 'value', None)
    yield getattr(value, 'value', value)
    nodes = itertools.chain(
        getattr(node, 'arguments', []),
        getattr(node, 'fields', []),
        getattr(value, 'fields', []),
        getattr(value, 'values', []),
        getattr(getattr(node, 'selection_set', None), 'selections', []),
    )
    for node in nodes:
        yield from references(node)


class TimingMiddleware(base.BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):  # pragma: no cover
        start = datetime.now()
        try:
            return await call_next(request)
        finally:
            end = datetime.now()
            print(f"[{end.replace(microsecond=0)}]: {end - start}")


class GraphQL(strawberry.asgi.GraphQL):
    def __init__(self, root_value, **kwargs):
        schema = strawberry.Schema(type(root_value), types=Column.__subclasses__())
        super().__init__(schema, **kwargs)
        self.root_value = root_value

    async def get_root_value(self, request):
        return self.root_value
