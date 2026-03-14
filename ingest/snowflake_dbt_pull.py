"""
Pull dbt project files from a Snowflake stage into a local directory.

Use this to sync the dbt project that lives in Snowflake (e.g. in an internal
stage or the stage used by "dbt on Snowflake") so you can run and test models
locally (e.g. with DuckDB via dbt-duckdb).

Note: Snowflake's GET command flattens subdirectories. If your stage has
models/silver/, models/gold/, etc., use Snowflake CLI to preserve structure:
  snow stage copy @MY_DB.MY_SCHEMA.DBT_STAGE dbt/medallion --recursive
(Requires Snowflake CLI and connection configured.)

Authentication uses the same environment variables as snowflake_extract.py:

- SNOWFLAKE_ACCOUNT
- SNOWFLAKE_USER
- SNOWFLAKE_PASSWORD
- SNOWFLAKE_WAREHOUSE
- SNOWFLAKE_DATABASE
- SNOWFLAKE_SCHEMA
- SNOWFLAKE_ROLE (optional)

Examples:

  # Pull entire stage contents into dbt/medallion
  python ingest/snowflake_dbt_pull.py @MY_DB.MY_SCHEMA.DBT_STAGE dbt/medallion

  # Pull only dbt-relevant files (SQL, YAML, YML)
  python ingest/snowflake_dbt_pull.py @MY_DB.MY_SCHEMA.DBT_STAGE dbt/medallion --pattern ".*\\.(sql|yml|yaml)$"
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence

try:
    import snowflake.connector  # type: ignore
except Exception as e:  # pragma: no cover
    snowflake = None  # type: ignore
    _snowflake_import_error = e
else:
    _snowflake_import_error = None

from ingest.snowflake_extract import SnowflakeConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _file_uri(path: Path) -> str:
    """Build a file:// URI for Snowflake GET (local path)."""
    resolved = path.resolve()
    posix = resolved.as_posix()
    if posix.startswith("/"):
        return "file://" + posix
    return "file:///" + posix


def pull_dbt_from_stage(
    config: SnowflakeConfig,
    stage: str,
    out_dir: Path,
    *,
    pattern: Optional[str] = None,
    force: bool = False,
) -> Path:
    """
    Download files from a Snowflake stage into out_dir using GET.

    stage: e.g. @MY_DB.MY_SCHEMA.MY_STAGE or @MY_STAGE/path/to/project
    out_dir: local directory (created if missing)
    pattern: optional regex for file names (e.g. ".*\\.(sql|yml|yaml)$")
    force: if True, overwrite existing files (Snowflake GET FORCE=TRUE)
    """
    if _snowflake_import_error is not None:  # pragma: no cover
        raise ImportError(
            "snowflake-connector-python is required. Install it and try again."
        ) from _snowflake_import_error

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    kwargs = {
        "account": config.account,
        "user": config.user,
        "password": config.password,
        "warehouse": config.warehouse,
        "database": config.database,
        "schema": config.schema,
    }
    if config.role:
        kwargs["role"] = config.role

    logger.info("Connecting to Snowflake")
    with snowflake.connector.connect(**kwargs) as conn:
        with conn.cursor() as cur:
            # LIST to verify stage and optionally log what we'll get
            list_sql = f"LIST {stage}"
            logger.info("Listing stage: %s", list_sql)
            cur.execute(list_sql)
            rows = cur.fetchall()
            if not rows:
                logger.warning("Stage is empty or path has no files: %s", stage)
            else:
                logger.info("Found %s item(s) on stage", len(rows))

            # GET downloads from stage to local path (client-side)
            file_uri = _file_uri(out_dir)
            get_sql = f"GET {stage} {file_uri}"
            if pattern:
                get_sql += f" PATTERN = '{pattern}'"
            if force:
                get_sql += " FORCE = TRUE"
            logger.info("Downloading to %s", out_dir)
            cur.execute(get_sql)
            # GET returns result rows (file names, size, status)
            for row in cur.fetchall():
                logger.debug("GET: %s", row)

    return out_dir


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pull dbt project files from a Snowflake stage to a local directory"
    )
    p.add_argument(
        "stage",
        help="Stage location (e.g. @MY_DB.MY_SCHEMA.DBT_STAGE or @MY_STAGE/project/)",
    )
    p.add_argument(
        "out_dir",
        type=Path,
        help="Local directory to download into (e.g. dbt/medallion)",
    )
    p.add_argument(
        "--pattern",
        default=None,
        help="Regex to filter files (e.g. '.*\\.(sql|yml|yaml)$'). Default: all files",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing local files",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    config = SnowflakeConfig.from_env()
    pull_dbt_from_stage(
        config,
        args.stage.strip(),
        args.out_dir,
        pattern=args.pattern,
        force=args.force,
    )
    logger.info("Done. Project files are in %s", args.out_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
