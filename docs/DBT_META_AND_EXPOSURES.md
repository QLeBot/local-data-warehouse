# dbt meta and exposures

This doc explains **meta** and **exposures** in dbt and how they are used in this project for lineage and documentation.

## Meta

**What it is:** `meta` is a key on dbt resources (models, sources, seeds, snapshots, exposures) that holds **arbitrary key-value metadata**. It does not affect how dbt compiles or runs; it is for documentation and tooling.

**Where it appears:** Meta is compiled into `manifest.json` and shown in the **dbt docs** site (in the “Details” panel for each resource). You can also consume it via the manifest in custom tools (e.g. ownership, PII flags, BI hints).

**Common uses:**

| Use case | Example `meta` |
|----------|-----------------|
| Ownership | `owner: "analytics-team"` or `owner: { name: "...", email: "..." }` (exposures use a dedicated `owner` property; meta can add more) |
| Maturity | `maturity: high \| medium \| low` (exposures have a dedicated `maturity` property; meta can be used on models) |
| PII / governance | `contains_pii: true`, `tier: "gold"` |
| BI / tool hints | `powerbi_dataset: "Sales"`, `metabase_dashboard_id: 42` |

**How to add it:** In any YAML file that defines the resource (e.g. `schema.yml` for models), add a `meta:` key. In dbt v1.10+, `meta` on some resources may live under `config:`; check the [dbt meta docs](https://docs.getdbt.com/reference/resource-configs/meta) for your version.

Example on a model:

```yaml
models:
  - name: customer_metrics
    description: "Gold layer customer metrics"
    meta:
      owner: "analytics"
      maturity: high
      contains_pii: false
    config:
      tags: [gold, marts]
```

In this project, the medallion dbt project uses `meta` in the gold (and optionally silver) schema files so that ownership and maturity show up in dbt docs.

---

## Exposures

**What they are:** **Exposures** represent **downstream consumers** of your dbt project: dashboards (Power BI, Metabase, Tableau), notebooks, applications, or ML models. They are defined in YAML and linked to models (or sources) via `depends_on`.

**Why use them:**

- **Lineage** – In dbt docs, exposures appear as nodes at the “end” of the DAG. You see which models feed which dashboard or app, and you can do impact analysis (e.g. “if I change this model, which exposures are affected?”).
- **Documentation** – You document where your gold (and silver) data is used: Power BI report X, Metabase dashboard Y.
- **Governance** – Exposures can have `owner`, `maturity`, and `meta`; some tools use this for SLAs or alerts.

**Required properties:**

- **name** – Unique identifier (letters, numbers, underscores).
- **type** – One of: `dashboard`, `notebook`, `analysis`, `ml`, `application`.
- **owner** – At least `name` or `email`.
- **depends_on** – List of upstream refs/sources, e.g. `ref('my_gold_model')`, `source('raw', 'customers')`.

**Optional:** `description`, `url`, `label` (human-friendly name), `maturity`, `meta`, `tags`.

**Where to define them:** In any `.yml` file under your project (often under `models/`), with a top-level `exposures:` key. In this repo they live in `dbt/medallion/models/exposures.yml`.

**Example:**

```yaml
exposures:
  - name: powerbi_sales_report
    label: "Power BI – Sales report"
    type: dashboard
    maturity: high
    url: "file:///path/to/report.pbix"
    description: "Local Power BI report for testing gold layer."
    owner:
      name: Analytics
      email: analytics@example.com
    depends_on:
      - ref('customer_metrics')
```

In this project, we define placeholder exposures for **Power BI** and **Metabase** so that once you add gold models and point BI at them, the lineage in dbt docs shows “this model feeds this exposure.” You update `depends_on` to match your real gold (or silver) models.

---

## How it’s integrated in this repo

1. **Meta** – Used in `dbt/medallion/models/gold/schema.yml` (and optionally silver) so that gold-layer models carry owner/maturity (or other keys) and appear in dbt docs.
2. **Exposures** – `dbt/medallion/models/exposures.yml` defines Power BI and Metabase as dashboard-type exposures. They reference a small placeholder gold model so the project compiles and docs generate; when you add real gold models, change `depends_on` to `ref('your_gold_model')` (and add more exposures if needed).
3. **dbt docs** – Run `dbt docs generate` and `dbt docs serve` from the dbt project directory (or use `make docs-serve` from the repo root with the right `DBT_PROJECT`). The docs site shows model/source lineage and the exposure nodes, plus any meta you defined.

For more detail, see [dbt – Exposures](https://docs.getdbt.com/docs/build/exposures) and [dbt – meta](https://docs.getdbt.com/reference/resource-configs/meta).
