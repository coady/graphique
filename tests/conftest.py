import json
import os
import sys
from pathlib import Path
import pyarrow as pa
import pyarrow.dataset as ds
import pytest
from starlette import testclient

fixtures = Path(__file__).parent / 'fixtures'


def pytest_report_header(config):
    return f'pyarrow {pa.__version__}'


class TestClient(testclient.TestClient):
    def execute(self, query, **variables):
        response = self.post('/graphql', json={'query': query, 'variables': variables})
        response.raise_for_status()
        result = response.json()
        for error in result.get('errors', []):
            raise ValueError(error)
        return result['data']


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
def fedclient(request):
    app = load('alltypes.parquet', FEDERATED='a_table')
    return TestClient(app)


@pytest.fixture(scope='module')
def aliasclient(request):
    columns = {'snakeId': 'snake_id', 'camelId': 'camelId'}
    app = load('alltypes.parquet', COLUMNS=json.dumps(columns))
    return TestClient(app)


@pytest.fixture(scope='module')
def executor():
    app = load('alltypes.parquet', FILTERS='{}')
    return TestClient(app).execute
