include common/Makefile
all: format test-full

grep-todo:
	@grep -n -e "TODO:" -e "FIXME:" -r --exclude="Makefile" --exclude-dir=".venv" . || true

format: format-black format-isort

format-black:
	@echo [black] && poetry run black . -v
	#  --exclude "pnq\/__template__\.py" ".venv"

format-isort:
	@echo [isort] && poetry run isort --profile black --filter-files .

test:
	@echo [pytest] && poetry run pytest -sv -m "not slow" -x # x -１つエラーが発生したら中断する

test-mypy:
	@poetry run mypy tests/mypy > tests/mypy/result.txt || echo ""

test-full: test-mypy
	@echo [pytest] && poetry run pytest . -sv

doc-build:
	@poetry run mkdocs build -v
	@cat docs/index.md > README.md
	@echo "" >> README.md
	@echo "" >> README.md
	@cat CHANGELOG.md >> README.md

doc-serve: doc-build
	@poetry run mkdocs serve -a localhost:8001


gen: generate test

generate: generate_query unasync format

generate_query:
	@echo [GENERATE_QUERY]
	@poetry run python3 pnq_template/generate.py -i pnq/__queries__.py -o pnq/queries.py

unasync:
	@echo [GENERATE_UNASYNC]
	@poetry run python3 pnq/_itertools/generate_unasync.py
