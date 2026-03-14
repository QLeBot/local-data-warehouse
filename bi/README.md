# BI tools – local testing with gold layer

Use **Power BI** (.pbix) and **Metabase** against your local DuckDB warehouse to test reports and dashboards on the same gold-layer data you build with dbt.

## Power BI Desktop (.pbix)

You can keep **local .pbix files** in this folder (or a subfolder like `powerbi/`) and connect them to the local DuckDB database so you can validate data and layout before publishing.

### Setup (one-time)

1. **DuckDB Power Query connector (MotherDuck)**  
   - [DuckDB Power Query Connector](https://github.com/MotherDuck-Open-Source/duckdb-power-query-connector/releases) – download the `.mez` file.  
   - [DuckDB ODBC driver (Windows)](https://github.com/duckdb/duckdb-odbc/releases) – install if required by the connector.

2. **Power BI Desktop**  
   - File → Options and settings → Options → **Security** → Data Extensions: enable **Allow any extensions to load without validation or warning**.  
   - Copy the `.mez` into `[Documents]\Power BI Desktop\Custom Connectors`.  
   - Restart Power BI Desktop.

### Connecting to local DuckDB

1. Get Data → More… → search **DuckDB** → choose the DuckDB connector.
2. **MotherDuck Token:** enter `localtoken` (for a local file, not MotherDuck cloud).
3. **Database Location:** full path to your warehouse file, e.g.  
   `C:\path\to\local-data-warehouse\warehouse\local.duckdb`  
   or `~/Code/local-data-warehouse/warehouse/local.duckdb` (Power BI may resolve `~`).
4. Connect and select the **gold** (and silver if needed) tables to load or use in DirectQuery.

Store your `.pbix` files in this repo (e.g. `bi/powerbi/`) so you can version them and run the same reports locally after each `dbt build`.

---

## Metabase

Run Metabase locally and add DuckDB as a database so you can build and test questions/dashboards on the same gold layer.

### Option A – Metabase Cloud / hosted

Metabase Cloud does not support custom drivers. Use Option B for DuckDB.

### Option B – Self-hosted Metabase with DuckDB plugin

Metabase does not ship a DuckDB driver; use the community **Metabase DuckDB driver** (e.g. [MotherDuck’s plugin](https://github.com/MotherDuck-Open-Source/metabase_duckdb_driver)).

1. **Run Metabase** (Docker example below) with the DuckDB driver in the plugins directory.
2. **Add database** in Metabase: Database type **DuckDB**, then set the **JDBC URL** to your local file, e.g.  
   `jdbc:duckdb:warehouse/local.duckdb`  
   or an absolute path. If Metabase runs in Docker, mount the repo so the path inside the container points at `warehouse/local.duckdb`.
3. Sync metadata and build questions/dashboards against **gold** (and silver) tables.

### Example: Metabase + DuckDB in Docker

This repo includes an example in `bi/`:

- **Dockerfile.metabase** – Metabase image with the DuckDB driver plugin.
- **docker-compose.yml** – Runs Metabase and mounts `warehouse/` so it can open `warehouse/local.duckdb`.

From the **repo root**:

```bash
docker compose -f bi/docker-compose.yml up --build
```

Open http://localhost:3000, complete Metabase setup, then add a database: type **DuckDB**, JDBC URL: `jdbc:duckdb:/warehouse/local.duckdb`. Sync and build questions/dashboards on your gold (and silver) tables.

---

## Workflow

1. Run **dbt build** to refresh silver/gold in `warehouse/local.duckdb`.
2. **Power BI:** Open your .pbix and refresh; or create new reports pointing at the same DuckDB path.
3. **Metabase:** Sync the DuckDB database and refresh questions/dashboards.
4. Optionally **export/validate** the warehouse schema (see main README) so the database architecture stays in sync with expectations.
