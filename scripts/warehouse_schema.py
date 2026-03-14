"""
Export or validate the warehouse (DuckDB) schema as a database architecture model.

Use after `dbt build` to capture the current schema (schemas, tables, columns) into
a YAML file. Validate later to ensure the live DB still matches that model.

Usage:
  python scripts/warehouse_schema.py export [--db PATH] [--out PATH]
  python scripts/warehouse_schema.py validate [--db PATH] [--schema PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Project root (parent of scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = PROJECT_ROOT / "warehouse" / "local.duckdb"
DEFAULT_OUT = PROJECT_ROOT / "docs" / "warehouse_schema.yml"


def _introspect(conn) -> dict:
    """Return schema structure: { schema_name: { table_name: [ (col_name, dtype), ... ] } }."""
    q = """
    SELECT table_schema, table_name, column_name, data_type
    FROM information_schema.columns
    WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
    ORDER BY table_schema, table_name, ordinal_position
    """
    rows = conn.execute(q).fetchall()
    out = {}
    for schema, table, col, dtype in rows:
        out.setdefault(schema, {}).setdefault(table, []).append((col, dtype))
    return out


def export_schema(db_path: Path, out_path: Path) -> None:
    """Export current warehouse schema to a YAML file (architecture model)."""
    try:
        import yaml
    except ImportError:
        print("PyYAML is required for export: pip install pyyaml", file=sys.stderr)
        sys.exit(1)

    loader = DuckDBLoader(db_path=str(db_path), project_root=str(PROJECT_ROOT))
    with loader:
        conn = loader.connect()
        raw = _introspect(conn)

    # Convert to a serializable structure (list of dicts for stable order)
    data = {"schemas": []}
    for schema_name in sorted(raw.keys()):
        tables = []
        for table_name in sorted(raw[schema_name].keys()):
            columns = [{"name": c[0], "type": c[1]} for c in raw[schema_name][table_name]]
            tables.append({"name": table_name, "columns": columns})
        data["schemas"].append({"name": schema_name, "tables": tables})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"Exported schema to {out_path}")


def load_expected(out_path: Path) -> dict:
    """Load expected schema from YAML into same shape as _introspect."""
    try:
        import yaml
    except ImportError:
        print("PyYAML is required for validate: pip install pyyaml", file=sys.stderr)
        sys.exit(1)
    with open(out_path) as f:
        data = yaml.safe_load(f)
    out = {}
    for s in data.get("schemas", []):
        schema_name = s["name"]
        for t in s.get("tables", []):
            table_name = t["name"]
            cols = [(c["name"], c["type"]) for c in t.get("columns", [])]
            out.setdefault(schema_name, {})[table_name] = cols
    return out


def validate_schema(db_path: Path, schema_path: Path) -> bool:
    """Compare live DB to the architecture YAML; return True if match."""
    loader = DuckDBLoader(db_path=str(db_path), project_root=str(PROJECT_ROOT))
    with loader:
        conn = loader.connect()
        current = _introspect(conn)

    if not schema_path.exists():
        print(f"Schema file not found: {schema_path}. Run 'export' first.", file=sys.stderr)
        return False

    expected = load_expected(schema_path)
    ok = True
    for schema_name in sorted(set(current) | set(expected)):
        cur_tables = current.get(schema_name, {})
        exp_tables = expected.get(schema_name, {})
        for table_name in sorted(set(cur_tables) | set(exp_tables)):
            cur_cols = cur_tables.get(table_name, [])
            exp_cols = exp_tables.get(table_name, [])
            if cur_cols != exp_cols:
                ok = False
                full = f"{schema_name}.{table_name}"
                if not exp_cols:
                    print(f"  Unexpected: {full} exists in DB but not in architecture")
                elif not cur_cols:
                    print(f"  Missing: {full} in architecture but not in DB")
                else:
                    print(f"  Diff: {full} (columns or types differ)")
    if ok:
        print("Schema matches architecture model.")
    else:
        print("Schema does not match architecture model.", file=sys.stderr)
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Export or validate warehouse schema (architecture model)")
    parser.add_argument("command", choices=["export", "validate"], help="export: write schema to YAML; validate: compare DB to YAML")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help=f"DuckDB file (default: {DEFAULT_DB})")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"Schema YAML path for export (default: {DEFAULT_OUT})")
    parser.add_argument("--schema", type=Path, default=DEFAULT_OUT, help="Schema YAML path for validate (default: same as --out)")
    args = parser.parse_args()

    if args.command == "export":
        export_schema(args.db, args.out)
        return 0
    if args.command == "validate":
        return 0 if validate_schema(args.db, args.schema) else 1
    return 1


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT))
    raise SystemExit(main())
