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

### Streamlit dashboard (sites map)

The dashboard can **generate sites** live or **load them from DuckDB** (e.g. `raw.sites`).

```bash
streamlit run dashboard/dashboard_sites.py
```

### Notes / tips

- **DuckDB path**: by default the loader uses `warehouse/local.duckdb`. The dbt profile is also configured to use that file (see `dbt/medallion/profiles.yml`).
- **Layering**: `ingest/load.py` currently targets `raw` and `bronze` schemas; dbt is intended to build the downstream `silver`/`gold` models.
