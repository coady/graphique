from pathlib import Path
import pyarrow.dataset as ds
from graphique import GraphQL, core

fixtures = Path(__file__).parent / 'fixtures'
dataset = ds.dataset(fixtures / 'zipcodes.parquet')
roots = {
    'zipcodes': dataset.scanner(),
    'states': core.Table.sort(dataset.to_table(), 'state', 'county', indices='indices'),
    'zip_db': ds.dataset(fixtures / 'zip_db.parquet'),
}
app = GraphQL(roots, keys={'zipcodes': ['zipcode']})
