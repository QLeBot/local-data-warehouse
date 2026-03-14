# Integrations and extensions

This doc describes how to plug **Snowflake MCP**, **Snowflake CLI**, and **Cortex** into the local-data-warehouse workflow, and suggests a few more extensions.

## Snowflake MCP server

A Snowflake MCP (Model Context Protocol) server lets AI assistants (e.g. in Cursor or other MCP clients) call Snowflake without running code in your repo. Use it to:

- List stages, databases, tables
- Run queries (e.g. sample data, DDL)
- Invoke Cortex (Analyst, Code, etc.) or custom tools

**Options:**

1. **Snowflake-managed MCP** – Create an MCP server in Snowflake with `CREATE MCP SERVER` and expose tools such as `SYSTEM_EXECUTE_SQL`, `CORTEX_ANALYST_MESSAGE`, `CORTEX_AGENT_RUN`. Your IDE/MCP client connects via Snowflake’s OAuth. See [Snowflake MCP documentation](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-agents-mcp).
2. **Custom MCP server** – Run an MCP server (e.g. in Python) that uses `snowflake-connector-python` or `snow` CLI under the hood; configure it with the same env vars as this project (see `.env.example`).

**Tying to this repo:** Point the MCP (or your client config) at the same Snowflake account/database/schema you use for `snowflake_extract.py` and `snow stage copy`. Use MCP to inspect stages before pull, run ad‑hoc SQL in Snowflake, or drive Cortex; keep local DuckDB as the dev mirror via extract + dbt pull + load + `dbt build`.

---

## Snowflake CLI (`snow`)

You already use the CLI for:

- **dbt project pull:** `snow stage copy @DB.SCHEMA.STAGE dbt/my_project --recursive`
- **Data/files:** `snow stage list`, `snow stage copy`, `snow sql -q "SELECT ..."` for quick queries

**Ways to extend:**

- **Scripts that wrap `snow`** – e.g. `scripts/snowflake_pull_dbt.sh` or a small Python script that runs `snow stage copy` with paths from env or config, so the “pull dbt” step is one command.
- **Same env as this repo** – Use `.env` (from `.env.example`) so `snow` and your Python ingest scripts share `SNOWFLAKE_*`; the CLI can read env or a `~/.snowflake/config.toml` that you keep in sync.

---

## Snowflake Cortex

Cortex (e.g. **Cortex Analyst**, **Cortex Code**, **Cortex Search**) fits in two places:

1. **Inside Snowflake** – Use Cortex for natural-language → SQL, summarization, or code generation over your Snowflake tables. Your MCP server can expose Cortex tools so an AI in the IDE can “ask Snowflake” or “run Cortex Analyst” without leaving the editor.
2. **Workflow** – Treat Snowflake + Cortex as the production side (query, analyze, govern); use this repo for **local dev**: pull dbt from Snowflake, load bronze (gzip CSV), run `dbt build` in DuckDB, validate schema and BI (Power BI, Metabase). When ready, deploy dbt back to Snowflake (e.g. `snow dbt deploy`) and run Cortex against the same schema.

No code changes are required in this repo; just document which Snowflake DB/schema your Cortex tools use so it stays aligned with the dbt project you pull.

---

## Shared configuration

- **`.env.example`** – Copy to `.env` and fill in `SNOWFLAKE_*` and optional paths (warehouse, dbt project). Use this for Python ingest, and optionally for Snowflake CLI or MCP client config.
- **Single source of truth** – Keep Snowflake account/database/schema/role in one place so MCP, CLI, and `ingest/snowflake_*.py` stay consistent.

---

## Optional extensions (not implemented)

Ideas you can add later:

| Area | Idea |
|------|------|
| **Orchestration** | `Makefile` or `scripts/run_pipeline.sh`: pull dbt (CLI) → load bronze → `dbt build` → schema export. One command for “full local refresh”. |
| **dbt docs** | Implemented: `make docs-serve`; meta and exposures in the dbt project. See README and `docs/DBT_META_AND_EXPOSURES.md`. |
| **Tests** | Pytest for `ingest.load`, `ingest.snowflake_extract`, `scripts.warehouse_schema` (e.g. test export/validate with a temp DuckDB). |
| **Data quality** | Lightweight checks after load or after dbt: row counts, key nulls, or optional Great Expectations/Pandera on bronze/silver. |
| **Gold export** | Script to export gold tables to Parquet/CSV (e.g. `data/export/gold/`) for archival or for Power BI file-based refresh if you move off direct DuckDB. |
| **Config layer** | YAML/TOML (e.g. `config/warehouse.yml`) for warehouse path, dbt project name, stage names, so scripts and MCP don’t hardcode paths. |

If you add a Snowflake MCP server (managed or custom), consider adding a short “MCP setup” section here with the exact tools you expose and how they map to this workflow (e.g. “tool: pull_dbt → runs snow stage copy for project X”).
