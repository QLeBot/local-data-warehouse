# Extensions checklist

Track optional improvements and integrations. Check off as you implement.

## Integrations

- [ ] **Snowflake MCP server** – Add Snowflake-managed or custom MCP; document in `docs/INTEGRATIONS.md` (tools, env, mapping to pull_dbt / extract).
- [ ] **Snowflake CLI wrapper** – Script (e.g. `scripts/snowflake_pull_dbt.sh` or Python) that runs `snow stage copy` with stage/path from env or config.
- [ ] **Config layer** – YAML/TOML (e.g. `config/warehouse.yml`) for warehouse path, dbt project name, stage names, so scripts and MCP don’t hardcode paths.

## Orchestration & pipeline

- [ ] **Full refresh target** – Makefile or script: pull dbt (CLI) → load bronze → `dbt build` → schema export (one command for “full local refresh”).
- [ ] **Orchestration** – Optional `scripts/run_pipeline.sh` or task runner (e.g. invoke) for multi-step runs.

## dbt & lineage

- [x] **dbt docs** – Generate and serve dbt docs for lineage; document in README (see “dbt docs (lineage)” section).
- [x] **Meta** – Use `meta` on models/sources for owner, maturity, PII flags; visible in dbt docs (see `docs/DBT_META_AND_EXPOSURES.md`).
- [x] **Exposures** – Define Power BI / Metabase (and other BI) as exposures for downstream lineage (see `docs/DBT_META_AND_EXPOSURES.md`).
- [x] **DB architecture from dbt** – Generate ERD from dbt artifacts with **dbterd** (`make erd` → `docs/erd.mmd`); alternatives in `docs/DB_ARCHITECTURE_FROM_DBT.md` (dbt-diagrams, erdgen, dbt-dbml-erd).

## Testing & quality

- [ ] **Tests** – Pytest for `ingest.load`, `ingest.snowflake_extract`, `scripts.warehouse_schema` (e.g. temp DuckDB for export/validate).
- [ ] **Data quality** – Lightweight checks after load or after dbt: row counts, key nulls, or optional Great Expectations/Pandera on bronze/silver.

## Export & BI

- [ ] **Gold export** – Script to export gold tables to Parquet/CSV (e.g. `data/export/gold/`) for archival or Power BI file-based refresh.

---

*Source: `docs/INTEGRATIONS.md` and conversation. Update this file when you add or complete items.*
