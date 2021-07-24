import os
import sys
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import strawberry
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
    os.environ['PARQUET_PATH'] = str(fixtures / path)
    os.environ.update(vars)
    sys.modules.pop('graphique.service', None)
    sys.modules.pop('graphique.settings', None)
    from graphique.service import app

    for var in vars:
        del os.environ[var]
    return app


@pytest.fixture(scope='module')
def table():
    return pq.read_table(fixtures / 'zipcodes.parquet')


@pytest.fixture(scope='module')
def client():
    app = load('zipcodes.parquet', INDEX='zipcode', FILTERS='[["zipcode", ">", 0]]')
    return TestClient(app)


@pytest.fixture(scope='module')
def dsclient(request):
    app = load('zipcodes.parquet', COLUMNS='zipcode,state,county', READ='0', INDEX='zipcode')
    return TestClient(app)


@pytest.fixture(scope='module')
def executor():
    app = load('alltypes.parquet', INDEX='snake_id,camelId', DICTIONARIES='string')
    schema = strawberry.Schema(query=type(app.root_value))

    def execute(query):
        result = schema.execute_sync(query, root_value=app.root_value)
        for error in result.errors or ():
            raise ValueError(error)
        return result.data

    return execute
