"""
Snowflake extraction module.

Reads data from Snowflake and stores it locally as Parquet files for offline use
(e.g., loading into DuckDB or transforming with dbt).

Authentication/config is typically provided via environment variables:

- SNOWFLAKE_ACCOUNT
- SNOWFLAKE_USER
- SNOWFLAKE_PASSWORD
- SNOWFLAKE_WAREHOUSE
- SNOWFLAKE_DATABASE
- SNOWFLAKE_SCHEMA
- SNOWFLAKE_ROLE (optional)

Example:
    python ingest/snowflake_extract.py table --table MY_TABLE --out data/extracted/snowflake/my_table.parquet
"""

from __future__ import annotations

import os
import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence

import pandas as pd

try:
    import snowflake.connector  # type: ignore
except Exception as e:  # pragma: no cover
    snowflake = None  # type: ignore
    _snowflake_import_error = e
else:
    _snowflake_import_error = None


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SnowflakeConfig:
    account: str
    user: str
    password: str
    warehouse: str
    database: str
    schema: str
    role: Optional[str] = None

    @staticmethod
    def from_env(prefix: str = "SNOWFLAKE_") -> "SnowflakeConfig":
        def get(name: str, required: bool = True) -> Optional[str]:
            val = os.getenv(f"{prefix}{name}")
            if required and (val is None or val.strip() == ""):
                raise ValueError(f"Missing required environment variable: {prefix}{name}")
            return val

        return SnowflakeConfig(
            account=get("ACCOUNT") or "",
            user=get("USER") or "",
            password=get("PASSWORD") or "",
            warehouse=get("WAREHOUSE") or "",
            database=get("DATABASE") or "",
            schema=get("SCHEMA") or "",
            role=get("ROLE", required=False),
        )


class SnowflakeExtractor:
    """Extracts data from Snowflake and writes it to local Parquet."""

    def __init__(self, config: SnowflakeConfig):
        if _snowflake_import_error is not None:  # pragma: no cover
            raise ImportError(
                "snowflake-connector-python is required for Snowflake extraction. "
                "Install it (pip/conda) and try again."
            ) from _snowflake_import_error
        self.config = config

    def _connect(self):
        kwargs = {
            "account": self.config.account,
            "user": self.config.user,
            "password": self.config.password,
            "warehouse": self.config.warehouse,
            "database": self.config.database,
            "schema": self.config.schema,
        }
        if self.config.role:
            kwargs["role"] = self.config.role
        return snowflake.connector.connect(**kwargs)  # type: ignore[name-defined]

    def fetch_pandas_batches(self, query: str) -> Iterator[pd.DataFrame]:
        """
        Stream query results as pandas DataFrames using Snowflake connector batches.

        Notes:
            Uses cursor.fetch_pandas_batches() which avoids loading the full result
            set into memory for large extracts.
        """
        logger.info("Connecting to Snowflake")
        with self._connect() as conn:
            with conn.cursor() as cur:
                logger.info("Executing query")
                cur.execute(query)
                for batch_df in cur.fetch_pandas_batches():
                    # Ensure pandas DataFrame
                    if batch_df is None or len(batch_df) == 0:
                        continue
                    yield batch_df

    def extract_query_to_parquet(
        self,
        query: str,
        out: Path,
        *,
        compression: str = "snappy",
    ) -> Path:
        """
        Extract a SQL query to a single Parquet file.

        For very large extracts you may prefer writing multiple files (dataset),
        but a single Parquet file is simplest to consume locally.
        """
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)

        total_rows = 0
        first = True
        parquet_writer = None

        try:
            import pyarrow as pa  # type: ignore
            import pyarrow.parquet as pq  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError("pyarrow is required to write Parquet files") from e

        for df in self.fetch_pandas_batches(query):
            total_rows += len(df)
            table = pa.Table.from_pandas(df, preserve_index=False)

            if first:
                parquet_writer = pq.ParquetWriter(out, table.schema, compression=compression)
                first = False

            assert parquet_writer is not None
            parquet_writer.write_table(table)

        if parquet_writer is not None:
            parquet_writer.close()
        else:
            # No rows: still create an empty parquet with no columns is awkward.
            # Instead, create an empty file to signal "ran but returned nothing".
            out.write_bytes(b"")
            logger.warning("Query returned 0 rows; wrote an empty file: %s", out)
            return out

        size_mb = out.stat().st_size / (1024 * 1024)
        logger.info("Wrote %s rows to %s (%.2f MB)", total_rows, out, size_mb)
        return out

    def extract_table_to_parquet(
        self,
        table: str,
        out: Path,
        *,
        columns: Optional[Sequence[str]] = None,
        where: Optional[str] = None,
        limit: Optional[int] = None,
        compression: str = "snappy",
    ) -> Path:
        cols = "*" if not columns else ", ".join(columns)
        query = f"SELECT {cols} FROM {table}"
        if where:
            query += f" WHERE {where}"
        if limit is not None:
            query += f" LIMIT {int(limit)}"
        return self.extract_query_to_parquet(query, out, compression=compression)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract data from Snowflake to local Parquet")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_table = sub.add_parser("table", help="Extract a table to Parquet")
    p_table.add_argument("--table", required=True, help="Fully qualified table name (e.g. MY_DB.MY_SCHEMA.MY_TABLE)")
    p_table.add_argument("--out", required=True, help="Output Parquet path")
    p_table.add_argument("--where", default=None, help="Optional WHERE clause (without the 'WHERE' keyword)")
    p_table.add_argument("--limit", type=int, default=None, help="Optional LIMIT")
    p_table.add_argument("--compression", default="snappy", help="Parquet compression (default: snappy)")

    p_query = sub.add_parser("query", help="Extract a SQL query to Parquet")
    p_query.add_argument("--sql", required=True, help="SQL query to execute (wrap in quotes)")
    p_query.add_argument("--out", required=True, help="Output Parquet path")
    p_query.add_argument("--compression", default="snappy", help="Parquet compression (default: snappy)")

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    cfg = SnowflakeConfig.from_env()
    extractor = SnowflakeExtractor(cfg)

    out = Path(args.out)

    if args.cmd == "table":
        extractor.extract_table_to_parquet(
            table=args.table,
            out=out,
            where=args.where,
            limit=args.limit,
            compression=args.compression,
        )
        return 0

    if args.cmd == "query":
        extractor.extract_query_to_parquet(
            query=args.sql,
            out=out,
            compression=args.compression,
        )
        return 0

    raise RuntimeError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())

