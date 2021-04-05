all:
	cythonize -aX linetrace=True graphique/*.pyx
	python3 setup.py build_ext -i --define CYTHON_TRACE_NOGIL

check: all
	python3 -m pytest --cov

lint:
	python3 setup.py check -ms
	black --check .
	flake8
	flake8 graphique/*.pyx --ignore E999,E211
	mypy -p graphique

html: all docs/schema.md
	python3 -m mkdocs build

docs/schema.md: docs/schema.graphql
	./node_modules/.bin/graphql-markdown \
		--title "Example Schema" \
		--prologue "Generated from a test fixture of zipcodes." \
		$? > $@

docs/schema.graphql: tests/fixtures/zipcodes.parquet
	python3 -m docs.schema $? > $@

dist:
	python3 setup.py sdist bdist_wheel
	docker run --rm -v $(PWD):/usr/src -w /usr/src quay.io/pypa/manylinux_2_24_x86_64 make cp37 cp38 cp39

cp37:
	/opt/python/$@-$@m/bin/pip install cython
	/opt/python/$@-$@m/bin/python setup.py build
	/opt/python/$@-$@m/bin/pip wheel . -w dist
	auditwheel repair dist/*$@m-linux_x86_64.whl

cp38 cp39:
	/opt/python/$@-$@/bin/pip install cython
	/opt/python/$@-$@/bin/python setup.py build
	/opt/python/$@-$@/bin/pip wheel . -w dist
	auditwheel repair dist/*$@-linux_x86_64.whl
