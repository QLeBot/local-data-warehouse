"""
Data loading module for DuckDB warehouse.
Supports loading data from various sources into the medallion architecture layers.
"""

import os
from pathlib import Path
from typing import Optional, Union, Literal
import logging

import duckdb
import pandas as pd
import polars as pl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DuckDBLoader:
    """Loader for ingesting data into DuckDB warehouse."""
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        project_root: Optional[str] = None
    ):
        """
        Initialize the DuckDB loader.
        
        Args:
            db_path: Path to DuckDB database file. If None, uses default from project structure.
            project_root: Root directory of the project. If None, infers from current file location.
        """
        if project_root is None:
            # Infer project root (assuming ingest/ is at project root level)
            project_root = Path(__file__).parent.parent
        
        if db_path is None:
            # Default to warehouse/local.duckdb in project root
            db_path = str(Path(project_root) / "warehouse" / "local.duckdb")
        
        self.db_path = db_path
        self.project_root = Path(project_root)
        self.conn: Optional[duckdb.DuckDBPyConnection] = None
        
        logger.info(f"Initialized DuckDB loader with database: {db_path}")
    
    def connect(self) -> duckdb.DuckDBPyConnection:
        """Establish connection to DuckDB database."""
        if self.conn is None:
            self.conn = duckdb.connect(self.db_path)
            logger.info("Connected to DuckDB database")
        return self.conn
    
    def close(self):
        """Close the database connection."""
        if self.conn is not None:
            self.conn.close()
            self.conn = None
            logger.info("Closed DuckDB connection")
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def create_schema(self, schema_name: str):
        """
        Create a schema if it doesn't exist.
        
        Args:
            schema_name: Name of the schema to create.
        """
        conn = self.connect()
        conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")
        logger.info(f"Ensured schema '{schema_name}' exists")
    
    def load_from_file(
        self,
        file_path: Union[str, Path],
        table_name: str,
        schema: Literal["raw", "bronze"] = "raw",
        if_exists: Literal["fail", "replace", "append"] = "replace",
        file_format: Optional[Literal["csv", "parquet", "json"]] = None
    ) -> None:
        """
        Load data from a file into DuckDB.
        
        Args:
            file_path: Path to the source file.
            table_name: Name of the target table.
            schema: Target schema (raw or bronze).
            if_exists: What to do if table exists ('fail', 'replace', 'append').
            file_format: File format ('csv', 'parquet', 'json'). Auto-detected if None.
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Auto-detect file format if not specified
        if file_format is None:
            suffix = file_path.suffix.lower()
            if suffix == '.csv':
                file_format = 'csv'
            elif suffix in ['.parquet', '.pqt']:
                file_format = 'parquet'
            elif suffix == '.json':
                file_format = 'json'
            else:
                raise ValueError(f"Unsupported file format: {suffix}. Specify file_format parameter.")
        
        conn = self.connect()
        self.create_schema(schema)
        
        full_table_name = f"{schema}.{table_name}"
        
        # Handle table existence
        if if_exists == "replace":
            conn.execute(f"DROP TABLE IF EXISTS {full_table_name}")
        elif if_exists == "fail":
            result = conn.execute(
                f"SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = '{schema}' AND table_name = '{table_name}'"
            ).fetchone()
            if result[0] > 0:
                raise ValueError(f"Table {full_table_name} already exists and if_exists='fail'")
        
        # Load data based on file format
        if file_format == 'csv':
            conn.execute(
                f"""
                CREATE TABLE {full_table_name} AS
                SELECT * FROM read_csv_auto('{file_path}', header=true)
                """
            )
        elif file_format == 'parquet':
            conn.execute(
                f"""
                CREATE TABLE {full_table_name} AS
                SELECT * FROM read_parquet('{file_path}')
                """
            )
        elif file_format == 'json':
            conn.execute(
                f"""
                CREATE TABLE {full_table_name} AS
                SELECT * FROM read_json_auto('{file_path}')
                """
            )
        
        # Get row count
        row_count = conn.execute(f"SELECT COUNT(*) FROM {full_table_name}").fetchone()[0]
        logger.info(f"Loaded {row_count} rows into {full_table_name} from {file_path}")
    
    def load_from_dataframe(
        self,
        df: Union[pd.DataFrame, pl.DataFrame],
        table_name: str,
        schema: Literal["raw", "bronze"] = "raw",
        if_exists: Literal["fail", "replace", "append"] = "replace"
    ) -> None:
        """
        Load data from a pandas or polars DataFrame into DuckDB.
        
        Args:
            df: pandas or polars DataFrame to load.
            table_name: Name of the target table.
            schema: Target schema (raw or bronze).
            if_exists: What to do if table exists ('fail', 'replace', 'append').
        """
        conn = self.connect()
        self.create_schema(schema)
        
        full_table_name = f"{schema}.{table_name}"
        
        # Handle table existence
        if if_exists == "replace":
            conn.execute(f"DROP TABLE IF EXISTS {full_table_name}")
        elif if_exists == "fail":
            result = conn.execute(
                f"SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = '{schema}' AND table_name = '{table_name}'"
            ).fetchone()
            if result[0] > 0:
                raise ValueError(f"Table {full_table_name} already exists and if_exists='fail'")
        
        # Convert polars to pandas if needed (DuckDB works better with pandas)
        if isinstance(df, pl.DataFrame):
            df = df.to_pandas()
        
        # Register DataFrame and create table
        conn.register('temp_df', df)
        conn.execute(f"CREATE TABLE {full_table_name} AS SELECT * FROM temp_df")
        conn.unregister('temp_df')
        
        row_count = len(df)
        logger.info(f"Loaded {row_count} rows into {full_table_name} from DataFrame")
    
    def load_from_directory(
        self,
        directory: Union[str, Path],
        schema: Literal["raw", "bronze"] = "raw",
        pattern: str = "*.csv",
        if_exists: Literal["fail", "replace", "append"] = "replace"
    ) -> None:
        """
        Load all matching files from a directory into DuckDB.
        Table names are derived from file names (without extension).
        
        Args:
            directory: Directory containing files to load.
            schema: Target schema (raw or bronze).
            pattern: Glob pattern to match files (e.g., '*.csv', '*.parquet').
            if_exists: What to do if table exists ('fail', 'replace', 'append').
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        files = list(directory.glob(pattern))
        
        if not files:
            logger.warning(f"No files found matching pattern '{pattern}' in {directory}")
            return
        
        logger.info(f"Found {len(files)} files to load from {directory}")
        
        for file_path in files:
            # Derive table name from file name (without extension)
            table_name = file_path.stem
            
            try:
                self.load_from_file(
                    file_path=file_path,
                    table_name=table_name,
                    schema=schema,
                    if_exists=if_exists
                )
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
                raise


# Convenience function for quick loading
def load_data(
    source: Union[str, Path, pd.DataFrame, pl.DataFrame],
    table_name: Optional[str] = None,
    schema: Literal["raw", "bronze"] = "raw",
    **kwargs
) -> None:
    """
    Convenience function to load data into DuckDB.
    
    Args:
        source: File path, directory path, or DataFrame to load.
        table_name: Table name (required for DataFrames, optional for files).
        schema: Target schema (raw or bronze).
        **kwargs: Additional arguments passed to loader methods.
    """
    with DuckDBLoader() as loader:
        if isinstance(source, (str, Path)):
            source_path = Path(source)
            
            if source_path.is_dir():
                loader.load_from_directory(source_path, schema=schema, **kwargs)
            else:
                if table_name is None:
                    table_name = source_path.stem
                loader.load_from_file(source_path, table_name, schema=schema, **kwargs)
        elif isinstance(source, (pd.DataFrame, pl.DataFrame)):
            if table_name is None:
                raise ValueError("table_name is required when loading from DataFrame")
            loader.load_from_dataframe(source, table_name, schema=schema, **kwargs)
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")


if __name__ == "__main__":
    # Example usage
    loader = DuckDBLoader()
    
    # Example: Load a CSV file
    # loader.load_from_file(
    #     file_path="data/raw/example.csv",
    #     table_name="example",
    #     schema="raw"
    # )
    
    # Example: Load all CSV files from a directory
    # loader.load_from_directory(
    #     directory="data/raw",
    #     schema="raw",
    #     pattern="*.csv"
    # )
    
    # Example: Load from DataFrame
    # df = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
    # loader.load_from_dataframe(df, table_name="test", schema="raw")
    
    logger.info("DuckDB loader module ready. Import and use DuckDBLoader class or load_data() function.")
