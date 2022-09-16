check:
	python3 -m pytest -s --cov

lint:
	python3 -m black --check .
	flake8 --exclude .venv --ignore E501
	mypy -p graphique

html: docs/schema.md
	PYTHONPATH=$(PWD) python3 -m mkdocs build

docs/schema.md: docs/schema.graphql
	./node_modules/.bin/graphql-markdown \
		--title "Example Schema" \
		--no-toc \
		--prologue "Generated from a test fixture of zipcodes." \
		$? > $@

docs/schema.graphql: graphique/*.py
	PARQUET_PATH=tests/fixtures/zipcodes.parquet strawberry export-schema graphique.service:app.schema > $@
