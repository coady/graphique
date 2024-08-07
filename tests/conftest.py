import json
import os
import sys
from importlib import metadata
from pathlib import Path
import pyarrow.dataset as ds
import pytest

fixtures = Path(__file__).parent / 'fixtures'


def pytest_report_header(config):
    return [f'{name}: {metadata.version(name)}' for name in ('pyarrow', 'strawberry-graphql')]


class TestClient:
    def __init__(self, app):
        self.app = app

    def _execute(self, query):
        root_value = self.app.root_value
        return self.app.schema.execute_sync(query, root_value=root_value, context_value={})

    def execute(self, query):
        result = self._execute(query)
        for error in result.errors or []:
            raise ValueError(error)
        return result.data


def load(path, **vars):
    os.environ.update(vars, PARQUET_PATH=str(fixtures / path))
    sys.modules.pop('graphique.service', None)
    sys.modules.pop('graphique.settings', None)
    from graphique.service import app

    for var in vars:
        del os.environ[var]
    return app


@pytest.fixture(scope='module')
def table():
    return ds.dataset(fixtures / 'zipcodes.parquet').to_table()


@pytest.fixture(scope='module')
def client():
    filters = json.dumps({'zipcode': {'gt': 0}})
    app = load('zipcodes.parquet', FILTERS=filters)
    return TestClient(app)


@pytest.fixture(params=[None, ['zipcode', 'state', 'county']], scope='module')
def dsclient(request):
    app = load('zipcodes.parquet', COLUMNS=json.dumps(request.param))
    return TestClient(app)


@pytest.fixture(scope='module')
def partclient(request):
    app = load('partitioned')
    return TestClient(app)


@pytest.fixture(scope='module')
def fedclient():
    from .federated import app

    return TestClient(app)


@pytest.fixture(scope='module')
def aliasclient():
    columns = {'snakeId': 'snake_id', 'camelId': 'camelId'}
    app = load('alltypes.parquet', COLUMNS=json.dumps(columns))
    return TestClient(app)


@pytest.fixture(scope='module')
def executor():
    app = load('alltypes.parquet', FILTERS='{}')
    return TestClient(app).execute
