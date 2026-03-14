## local-data-warehouse

Local-first **medallion architecture** (raw → bronze → silver → gold) using **DuckDB** as the warehouse and **dbt** for transformations. It also includes:

- **Ingestion utilities** to extract data from **SQL Server** (optionally anonymized) and load files/DataFrames into DuckDB
- **Synthetic data generators** to bootstrap demo datasets (e.g., sites with coordinates)
- A small **Streamlit dashboard** to visualize sites on a map (either generated live or loaded from DuckDB)

### What you get

- **Warehouse**: a single local database file at `warehouse/local.duckdb`
- **Schemas / layers** (DuckDB):
  - **raw**: landing zone (1:1 with sources/files)
  - **bronze**: lightly cleaned/standardized
  - **silver/gold**: curated models built by dbt (see `dbt/medallion/models/`)

### Repo structure

- `ingest/extract.py`: SQL Server extraction + anonymization helpers
- `ingest/load.py`: load files or DataFrames into DuckDB (`raw` / `bronze` schemas)
- `data/generator/`: synthetic data generators (e.g. `sites.py` exports parquet)
- `dbt/medallion/`: dbt project (profile already points to `warehouse/local.duckdb`)
- `scripts/warehouse_schema.py`: export or validate warehouse schema (database architecture model)
- `docs/warehouse_schema.yml`: committed schema snapshot (after `warehouse_schema export`) for validation
- `docs/INTEGRATIONS.md`: Snowflake MCP, CLI, Cortex integration notes and extension ideas
- `docs/EXTENSIONS_CHECKLIST.md`: Checklist of optional extensions (integrations, tests, gold export, etc.)
- `docs/DBT_META_AND_EXPOSURES.md`: How meta and exposures work and how they’re used for dbt docs lineage
- `bi/`: Power BI (.pbix) and Metabase – local BI testing against gold layer (see `bi/README.md`)
- `Makefile`: shortcuts for load-bronze, dbt-build, schema-export, schema-validate
- `.env.example`: template for Snowflake and optional paths (copy to `.env`)
- `dashboard/dashboard_sites.py`: Streamlit app to view sites on a map
- `warehouse/local.duckdb`: local DuckDB database file (created/used by the project)

### Prerequisites

- **Python** (recommended via conda; an environment file is provided: `environment.yml`)
- If extracting from SQL Server:
  - A reachable SQL Server instance
  - A working ODBC driver (the code defaults to **ODBC Driver 17 for SQL Server**)

### Setup

Option A — conda (recommended on Windows):

```bash
conda env create -f environment.yml
conda activate local_wh
```

Option B — pip:

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

### Quickstart (generate → load → transform)

1) Generate a demo `sites` dataset (writes a parquet file under `data/generated/`):

```bash
python data/generator/sites.py 1000
```

2) Load that parquet file into DuckDB `raw.sites` (replace the filename if different):

```bash
python -c "from ingest.load import load_data; load_data('data/generated/sites_1000.parquet', table_name='sites', schema='raw')"
```

3) Run dbt models (from the dbt project directory):

```bash
cd dbt/medallion
dbt run --profiles-dir .
dbt test --profiles-dir .
```

### Extracting from SQL Server (optional)

`ingest/extract.py` can extract tables/queries via SQLAlchemy and **anonymize common PII** using Faker.

Example (Python):

```python
from ingest.extract import extract_and_anonymize
from ingest.load import load_data

df = extract_and_anonymize(
    server="YOUR_SERVER",
    database="YOUR_DATABASE",
    table_name="Customers",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    anonymize=True,
)

load_data(df, table_name="customers", schema="raw")
```

### Extracting from Snowflake to Parquet (optional)

Use `ingest/snowflake_extract.py` to pull data from Snowflake and store it locally as Parquet files.
This is useful to work **offline** (DuckDB/dbt) while keeping the source-of-truth in Snowflake.

1) Set environment variables (PowerShell example):

```powershell
$env:SNOWFLAKE_ACCOUNT="YOUR_ACCOUNT"
$env:SNOWFLAKE_USER="YOUR_USER"
$env:SNOWFLAKE_PASSWORD="YOUR_PASSWORD"
$env:SNOWFLAKE_WAREHOUSE="YOUR_WAREHOUSE"
$env:SNOWFLAKE_DATABASE="YOUR_DATABASE"
$env:SNOWFLAKE_SCHEMA="YOUR_SCHEMA"
# optional
$env:SNOWFLAKE_ROLE="YOUR_ROLE"
```

2) Extract a table to Parquet:

```bash
python ingest/snowflake_extract.py table --table YOUR_DB.YOUR_SCHEMA.YOUR_TABLE --out data/extracted/snowflake/your_table.parquet
```

3) Load the Parquet into DuckDB (example: `raw.your_table`):

```bash
python -c "from ingest.load import load_data; load_data('data/extracted/snowflake/your_table.parquet', table_name='your_table', schema='raw')"
```

You can also extract an arbitrary query:

```bash
python ingest/snowflake_extract.py query --sql \"SELECT * FROM YOUR_DB.YOUR_SCHEMA.YOUR_TABLE LIMIT 1000\" --out data/extracted/snowflake/sample.parquet
```

### Local dbt dev with Snowflake project and bronze gzip CSV

A typical workflow is: pull the dbt project from Snowflake (e.g. via Snowflake CLI), load a **bronze-layer dataset** (e.g. gzip CSV files) into DuckDB, then run **dbt build** to build and test silver and gold models locally.

1. **Pull the dbt project** (Snowflake CLI preserves directory structure):

```bash
snow stage copy @YOUR_DB.YOUR_SCHEMA.DBT_STAGE dbt/my_snowflake_project --recursive
```

Use a separate directory (e.g. `dbt/my_snowflake_project`) so it sits side by side with `dbt/medallion`. Ensure the project’s `profiles.yml` in that directory has a target pointing at `warehouse/local.duckdb` (or your local DuckDB path) for local runs.

2. **Put your raw/bronze data** in a folder as gzip CSV files (e.g. `data/bronze/*.csv.gz`). Table names are taken from the file name (e.g. `customers.csv.gz` → table `customers`).

3. **Load the bronze dataset into DuckDB** (loader supports `.csv.gz`; DuckDB infers compression):

```bash
python -c "
from ingest.load import load_data
load_data('data/bronze', schema='bronze', pattern='*.csv.gz')
"
```

Or load a single file:

```bash
python -c "from ingest.load import load_data; load_data('data/bronze/customers.csv.gz', schema='bronze')"
```

4. **Run dbt build** (builds silver/gold and runs tests) from the Snowflake project directory:

```bash
cd dbt/my_snowflake_project
dbt build --profiles-dir .
```

Your dbt sources / refs should point at the same schema you loaded into (e.g. `bronze`) and the same table names as the CSV stems.

### Validating with a database architecture model

After building dbt models, you can capture the warehouse schema as a **YAML architecture model** and later validate that the live DB still matches it (e.g. in CI or before releases).

1. **Export** the current schema (run after `dbt build`):

```bash
python scripts/warehouse_schema.py export
```

This writes `docs/warehouse_schema.yml` (schemas, tables, columns). Commit this file to version the expected architecture.

2. **Validate** the warehouse against that file:

```bash
python scripts/warehouse_schema.py validate
```

If the DB schema has drifted (missing/extra tables or columns), the command exits with an error. Use `--db` and `--out` / `--schema` to point at another DuckDB file or YAML path.

### dbt docs (lineage, meta, exposures)

Generate and serve the **dbt docs** site for lineage and documentation:

```bash
make docs-serve
# or: cd dbt/medallion && dbt docs generate --profiles-dir . && dbt docs serve --profiles-dir .
```

Then open http://localhost:8080 (or the URL dbt prints). You get:

- **Lineage** – DAG of sources → models → exposures.
- **Meta** – Custom metadata on models (e.g. owner, maturity, PII) defined in `schema.yml` under `meta:`; visible in the docs Details panel.
- **Exposures** – Downstream consumers (Power BI, Metabase, etc.) defined in `dbt/medallion/models/exposures.yml` with `depends_on` refs to gold models; they appear as nodes at the end of the DAG so you can see impact (e.g. “this model feeds these dashboards”).

See **`docs/DBT_META_AND_EXPOSURES.md`** for how to use meta and exposures and how they’re integrated in this project.

### Power BI (.pbix) and Metabase for gold-layer testing

Gold models are intended for BI. You can test reports and dashboards locally:

- **Power BI Desktop** – Store `.pbix` files in `bi/powerbi/` and connect them to your local DuckDB using the [DuckDB Power Query connector](https://github.com/MotherDuck-Open-Source/duckdb-power-query-connector) (MotherDuck). Use token `localtoken` and Database Location = path to `warehouse/local.duckdb`. See `bi/README.md` for setup.

- **Metabase** – Run Metabase with the [DuckDB driver plugin](https://github.com/MotherDuck-Open-Source/metabase_duckdb_driver) and add your warehouse as a DuckDB database (JDBC URL pointing at `warehouse/local.duckdb`). An example Docker setup is in `bi/` (Dockerfile + docker-compose); see `bi/README.md`.

After each `dbt build`, refresh Power BI or Metabase to test on the latest silver/gold data.

---

### Pulling dbt project files from Snowflake (alternatives)

**Snowflake CLI (recommended when you need directory structure)**  
[snow stage copy](https://docs.snowflake.com/en/developer-guide/snowflake-cli/install-cli) preserves folder layout:

```bash
snow stage copy @YOUR_DB.YOUR_SCHEMA.DBT_STAGE dbt/my_snowflake_project --recursive
```

**Python script (flat download)**  
If you don’t need subfolders preserved, you can use the project’s script (same Snowflake env vars as the extract script):

```bash
python ingest/snowflake_dbt_pull.py @YOUR_DB.YOUR_SCHEMA.DBT_STAGE dbt/my_snowflake_project
```

**Git as source**  
If the project is deployed from a Git repo, clone or pull that repo into a folder under `dbt/` instead of pulling from a stage.

**Two dbt projects side by side**  
Run the project you want by `cd`-ing into its directory: `cd dbt/medallion` or `cd dbt/my_snowflake_project`, then `dbt run --profiles-dir .` or `dbt build --profiles-dir .`.

### Streamlit dashboard (sites map)

The dashboard can **generate sites** live or **load them from DuckDB** (e.g. `raw.sites`).

```bash
streamlit run dashboard/dashboard_sites.py
```

### Notes / tips

- **DuckDB path**: by default the loader uses `warehouse/local.duckdb`. The dbt profile is also configured to use that file (see `dbt/medallion/profiles.yml`).
- **Layering**: `ingest/load.py` currently targets `raw` and `bronze` schemas; dbt is intended to build the downstream `silver`/`gold` models.

### Integrations and extensions

- **Config:** Copy `.env.example` to `.env` and set `SNOWFLAKE_*` (and optional paths). Use the same env for Snowflake CLI or MCP so everything shares one config.
- **Runbook:** A `Makefile` provides shortcuts for `load-bronze`, `dbt-build`, `schema-export`, `schema-validate` (e.g. `make dbt-build`). Override with `DBT_PROJECT=dbt/my_snowflake_project` or `WAREHOUSE_PATH=...`.
- **Snowflake MCP, CLI, Cortex:** See **`docs/INTEGRATIONS.md`** for how to plug in a Snowflake MCP server, Snowflake CLI (`snow`), and Cortex (Analyst, Code) into this workflow, plus optional ideas (orchestration, tests, data quality, gold export).
