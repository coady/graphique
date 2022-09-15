from pathlib import Path
import pyarrow.dataset as ds
from graphique import GraphQL

fixtures = Path(__file__).parent / 'fixtures'
roots = {
    'zipcodes': ds.dataset(fixtures / 'zipcodes.parquet'),
    'zip_db': ds.dataset(fixtures / 'zip_db.parquet'),
}
app = GraphQL(roots)
