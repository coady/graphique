import os
from pathlib import Path
import numpy as np
import pyarrow as pa
import strawberry
import pytest
from starlette import testclient

fixtures = Path(__file__).parent / 'fixtures'


def pytest_report_header(config):
    return f'pyarrow {pa.__version__}, strawberry {strawberry.__version__}, numpy {np.__version__}'


class TestClient(testclient.TestClient):
    def execute(self, **kwargs):
        response = self.post('/graphql', json=kwargs)
        assert response.status_code == 200
        result = response.json()
        for error in result.get('errors', []):
            raise ValueError(error)
        return result['data']


@pytest.fixture(scope='module')
def client():
    os.environ['PARQUET_PATH'] = str(fixtures / 'zipcodes.parquet')
    from graphique.service import app

    return TestClient(app)
