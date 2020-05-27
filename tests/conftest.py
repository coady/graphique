import os
import sys
from pathlib import Path
import graphql
import pyarrow as pa
import strawberry
import pytest
from starlette import testclient

fixtures = Path(__file__).parent / 'fixtures'


def pytest_report_header(config):
    return f'pyarrow {pa.__version__}, strawberry {strawberry.__version__}'


class TestClient(testclient.TestClient):
    def execute(self, query, **variables):
        response = self.post('/graphql', json={'query': query, 'variables': variables})
        response.raise_for_status()
        result = response.json()
        for error in result.get('errors', []):
            raise ValueError(error)
        return result['data']


def load(path):
    os.environ['PARQUET_PATH'] = str(fixtures / path)
    sys.modules.pop('graphique.service', None)
    sys.modules.pop('graphique.settings', None)
    from graphique.service import table

    return table


@pytest.fixture(scope='module')
def table():
    return load('zipcodes.parquet')


@pytest.fixture(scope='module')
def execute():
    table = load('alltypes.parquet')
    from graphique.service import Columns

    schema = strawberry.Schema(query=Columns)

    def execute(query):
        result = graphql.graphql_sync(schema, query, root_value=Columns(table))
        for error in result.errors or ():
            raise ValueError(error)
        return result.data

    return execute


@pytest.fixture(scope='module')
def client(table):
    from graphique.service import app

    return TestClient(app)


@pytest.fixture(scope='module')
def schema():
    return open(fixtures / 'schema.graphql').read()
