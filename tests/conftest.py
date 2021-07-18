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
    from graphique.service import root

    for var in vars:
        del os.environ[var]
    return root


@pytest.fixture(scope='module')
def table():
    return pq.read_table(fixtures / 'zipcodes.parquet')


@pytest.fixture(scope='module')
def client():
    load('zipcodes.parquet', INDEX='zipcode', FILTERS='[["zipcode", ">", 0]]')
    from graphique.service import app

    return TestClient(app)


@pytest.fixture(scope='module')
def dsclient():
    load('zipcodes.parquet', READ='0', COLUMNS='state,county')
    from graphique.service import app

    return TestClient(app)


@pytest.fixture(scope='module')
def executor():
    load('alltypes.parquet', INDEX='snake_id,camelId', DICTIONARIES='string')
    from graphique import service

    schema = strawberry.Schema(query=service.IndexedTable)

    def execute(query):
        result = schema.execute_sync(query, root_value=service.IndexedTable(service.table))
        for error in result.errors or ():
            raise ValueError(error)
        return result.data

    return execute
