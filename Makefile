# Local data warehouse – common commands.
# Usage: make <target>. Override paths via env (e.g. WAREHOUSE_PATH, DBT_PROJECT).
# See README and docs/INTEGRATIONS.md for full workflow.

WAREHOUSE_PATH ?= warehouse/local.duckdb
DBT_PROJECT ?= dbt/medallion
SCHEMA_YML ?= docs/warehouse_schema.yml

ERD_OUTPUT ?= docs/erd.mmd
.PHONY: help load-bronze dbt-build schema-export schema-validate docs-serve erd

help:
	@echo "Targets:"
	@echo "  load-bronze     Load bronze CSV/gzip from data/bronze into DuckDB"
	@echo "  dbt-build       Run dbt build in $(DBT_PROJECT)"
	@echo "  schema-export   Export warehouse schema to $(SCHEMA_YML)"
	@echo "  schema-validate Validate warehouse against $(SCHEMA_YML)"
	@echo "  docs-serve      Generate dbt docs and serve"
	@echo "  erd             Generate DB architecture (ERD) from dbt artifacts to $(ERD_OUTPUT); requires manifest in $(DBT_PROJECT)/target (run dbt compile or dbt build first)"

load-bronze:
	python -c "from ingest.load import load_data; load_data('data/bronze', schema='bronze', pattern='*.csv.gz')"

dbt-build:
	cd $(DBT_PROJECT) && dbt build --profiles-dir .

schema-export:
	python scripts/warehouse_schema.py export --db $(WAREHOUSE_PATH) --out $(SCHEMA_YML)

schema-validate:
	python scripts/warehouse_schema.py validate --db $(WAREHOUSE_PATH) --schema $(SCHEMA_YML)

docs-serve:
	cd $(DBT_PROJECT) && dbt docs generate --profiles-dir . && dbt docs serve --profiles-dir .

erd:
	@test -f $(DBT_PROJECT)/target/manifest.json || (echo "Run 'make dbt-build' or 'cd $(DBT_PROJECT) && dbt compile --profiles-dir .' first." && exit 1)
	dbterd run --artifacts-dir $(DBT_PROJECT)/target --target mermaid --output $(ERD_OUTPUT)
	@echo "ERD written to $(ERD_OUTPUT). View in VS Code (Mermaid), GitHub, or render with mermaid-cli."
