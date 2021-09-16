all: format test

format: format-black format-isort

format-black:
	@echo [black] && poetry run black . -v
	#  --exclude "pnq\/__template__\.py" ".venv"

format-isort:
	@echo [isort] && poetry run isort --profile black --filter-files .

test: generate
	@echo [pytest] && poetry run pytest .

doc-build:
	@poetry run mkdocs build
	@cat docs/index.md > README.md
	@echo "" >> README.md
	@echo "" >> README.md
	@cat CHANGELOG.md >> README.md

doc-serve: doc-build
	# @poetry run mkdocs build
	@poetry run mkdocs serve -a localhost:8001


gen: generate
	@poetry run pytest . -x # １つエラーが発生したら中断する

generate:
	@poetry run python3 pnq_template/generate.py -i pnq/__queries__.py -o pnq/queries.py
	@make format