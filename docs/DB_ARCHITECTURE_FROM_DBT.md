# Generating database architecture from dbt models

Open-source **local** tools that produce an ERD or schema diagram from your dbt project (manifest/catalog or YAML). All run on your machine; no SaaS required.

---

## Recommended: **dbterd** (automatic from artifacts)

**What it does:** Reads dbt **artifacts** (`manifest.json`, optionally `catalog.json`) and infers entities and relationships from `ref()` and `source()` usage. No extra YAML or `meta` needed.

**Output formats:** Mermaid, DBML, PlantUML, GraphViz, D2. You can save the output and view it in editors (Mermaid in VS Code / GitHub), render to SVG/PNG with [mermaid-cli](https://github.com/mermaid-js/mermaid-cli), or open DBML in [dbdiagram.io](https://dbdiagram.io).

**Install:**

```bash
pip install dbterd
```

**Usage:**

1. Generate dbt artifacts (run from dbt project dir):

   ```bash
   cd dbt/medallion
   dbt compile --profiles-dir .
   # or: dbt docs generate --profiles-dir .
   ```

2. Run dbterd (from repo root or dbt project):

   ```bash
   dbterd run --artifacts-dir dbt/medallion/target --target mermaid --output docs/erd.mmd
   # or: --target dbml --output docs/erd.dbml
   ```

**Options:** `--omit-columns` to drop columns from the diagram; see `dbterd run --help`. Artifacts dir is where `manifest.json` (and optionally `catalog.json`) live, usually `dbt/<project>/target`.

**In this repo:** Use `make erd` after a successful `dbt compile` or `dbt build` (see Makefile). Output goes to `docs/erd.mmd` (or `docs/erd.dbml` if you change the target).

---

## Alternative: **dbt-diagrams** (meta-driven, inside dbt docs)

**What it does:** Lets you define ERD **relationships in model `meta`** (target model, cardinality, label). Renders as **Mermaid** inside your dbt docs, or exports to **SVG**.

**Pros:** ERD lives in dbt docs; you control exactly which relations and labels appear.  
**Cons:** You must declare each relationship in YAML; not inferred from `ref()` alone.

**Install:**

```bash
pip install dbt-diagrams
# For SVG export: pip install "dbt-diagrams[svg]"
```

**Usage:**

- In model YAML, add under `config.meta` (or `meta`):

  ```yaml
  meta:
    erd:
      connections:
        - diagram: main_erd
          target: orders
          source_cardinality: one
          target_cardinality: one_or_more
          label: creates
  ```

- Generate/serve docs with the plugin (replaces standard dbt docs commands):

  ```bash
  cd dbt/medallion
  dbt-diagrams docs generate
  dbt-diagrams docs serve
  ```

- Or export ERDs to SVG:

  ```bash
  dbt-diagrams render-erds -dbt-target-dir target --format svg --output docs/erd
  ```

**Docs:** [dbt-diagrams on PyPI](https://pypi.org/project/dbt-diagrams/), [GitHub](https://github.com/DJLemkes/dbt-diagrams).

---

## Other options

| Tool | Input | Output | Notes |
|------|--------|--------|--------|
| **erdgen** | dbt YAML | DBML | Lightweight; [GitHub](https://github.com/neo-andrew-moss/erdgen). |
| **dbt-dbml-erd** | dbt project | ERD | Python, MIT; [GitHub](https://github.com/ScalefreeCOM/dbt-dbml-erd). |

---

## Summary

- For **zero-config, automatic** architecture from dbt: use **dbterd** on `target/` after `dbt compile` or `dbt docs generate`.
- For **ERD embedded in dbt docs** and full control over relationships: use **dbt-diagrams** and define connections in model `meta`.

Both are open-source and run locally. In this repo, the Makefile target `make erd` uses **dbterd** to write `docs/erd.mmd` (or DBML if configured).
