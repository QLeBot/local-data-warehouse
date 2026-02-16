"""
Data extraction module for SQL Server.
Supports extracting data from SQL Server using SQLAlchemy and anonymizing it.
"""

import os
from pathlib import Path
from typing import Optional, Union, Dict, List, Any, Callable
import logging
from datetime import datetime, timedelta
import random
import string

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine
from faker import Faker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Faker for generating realistic random data
fake = Faker()


class SQLServerExtractor:
    """Extractor for SQL Server databases with anonymization capabilities."""
    
    def __init__(
        self,
        server: Optional[str] = None,
        database: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        driver: str = "ODBC Driver 17 for SQL Server",
        connection_string: Optional[str] = None,
        use_windows_auth: bool = False
    ):
        """
        Initialize the SQL Server extractor.
        
        Args:
            server: SQL Server hostname or IP address.
            database: Database name.
            username: Username for authentication.
            password: Password for authentication.
            driver: ODBC driver name (default: "ODBC Driver 17 for SQL Server").
            connection_string: Full connection string (overrides other parameters if provided).
            use_windows_auth: Use Windows Authentication (ignores username/password).
        """
        if connection_string:
            self.connection_string = connection_string
        else:
            if not server or not database:
                raise ValueError("server and database are required if connection_string is not provided")
            
            if use_windows_auth:
                # Windows Authentication
                self.connection_string = (
                    f"mssql+pyodbc://{server}/{database}?"
                    f"driver={driver}&"
                    f"trusted_connection=yes"
                )
            else:
                if not username or not password:
                    raise ValueError("username and password are required when not using Windows Authentication")
                
                # SQL Server Authentication
                self.connection_string = (
                    f"mssql+pyodbc://{username}:{password}@{server}/{database}?"
                    f"driver={driver}"
                )
        
        self.engine: Optional[Engine] = None
        logger.info(f"Initialized SQL Server extractor for database: {database or 'from connection string'}")
    
    def connect(self) -> Engine:
        """Establish connection to SQL Server database."""
        if self.engine is None:
            try:
                self.engine = create_engine(
                    self.connection_string,
                    pool_pre_ping=True,  # Verify connections before using
                    echo=False  # Set to True for SQL query logging
                )
                # Test connection
                with self.engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                logger.info("Connected to SQL Server database")
            except Exception as e:
                logger.error(f"Failed to connect to SQL Server: {e}")
                raise
        return self.engine
    
    def close(self):
        """Close the database connection."""
        if self.engine is not None:
            self.engine.dispose()
            self.engine = None
            logger.info("Closed SQL Server connection")
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def extract_table(
        self,
        table_name: str,
        schema: Optional[str] = None,
        query: Optional[str] = None,
        chunksize: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Extract data from a SQL Server table or query.
        
        Args:
            table_name: Name of the table to extract (ignored if query is provided).
            schema: Schema name (default: 'dbo').
            query: Custom SQL query to execute (overrides table_name).
            chunksize: Number of rows to read at a time (for large tables).
        
        Returns:
            pandas DataFrame with extracted data.
        """
        engine = self.connect()
        
        if query:
            sql_query = query
            logger.info(f"Executing custom query")
        else:
            if schema:
                full_table_name = f"{schema}.{table_name}"
            else:
                full_table_name = table_name
            
            sql_query = f"SELECT * FROM {full_table_name}"
            logger.info(f"Extracting data from table: {full_table_name}")
        
        try:
            if chunksize:
                # Read in chunks for large tables
                chunks = []
                for chunk in pd.read_sql(sql_query, engine, chunksize=chunksize):
                    chunks.append(chunk)
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.read_sql(sql_query, engine)
            
            logger.info(f"Extracted {len(df)} rows, {len(df.columns)} columns")
            return df
        except Exception as e:
            logger.error(f"Failed to extract data: {e}")
            raise
    
    def list_tables(self, schema: Optional[str] = None) -> List[str]:
        """
        List all tables in the database.
        
        Args:
            schema: Schema name to filter tables (None for all schemas).
        
        Returns:
            List of table names.
        """
        engine = self.connect()
        inspector = inspect(engine)
        
        if schema:
            tables = inspector.get_table_names(schema=schema)
        else:
            tables = inspector.get_table_names()
        
        logger.info(f"Found {len(tables)} tables")
        return tables


class DataAnonymizer:
    """Anonymizer for dataframes with various randomization strategies."""
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the data anonymizer.
        
        Args:
            seed: Random seed for reproducibility (None for random).
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            Faker.seed(seed)
        self.seed = seed
    
    def randomize_string(
        self,
        value: Any,
        length: Optional[int] = None,
        prefix: str = "",
        suffix: str = ""
    ) -> str:
        """
        Randomize a string value.
        
        Args:
            value: Original string value.
            length: Target length (uses original length if None).
            prefix: Prefix to add to randomized string.
            suffix: Suffix to add to randomized string.
        
        Returns:
            Randomized string.
        """
        if pd.isna(value):
            return value
        
        original = str(value)
        target_length = length if length is not None else len(original)
        
        # Generate random string
        random_chars = ''.join(random.choices(string.ascii_letters + string.digits, k=target_length))
        return f"{prefix}{random_chars}{suffix}"
    
    def randomize_email(self, value: Any) -> str:
        """Randomize an email address."""
        if pd.isna(value):
            return value
        return fake.email()
    
    def randomize_name(self, value: Any) -> str:
        """Randomize a name."""
        if pd.isna(value):
            return value
        return fake.name()
    
    def randomize_phone(self, value: Any) -> str:
        """Randomize a phone number."""
        if pd.isna(value):
            return value
        return fake.phone_number()
    
    def randomize_address(self, value: Any) -> str:
        """Randomize an address."""
        if pd.isna(value):
            return value
        return fake.address()
    
    def randomize_date(
        self,
        value: Any,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Any:
        """
        Randomize a date/datetime value.
        
        Args:
            value: Original date value.
            start_date: Minimum date for randomization (default: 10 years ago).
            end_date: Maximum date for randomization (default: today).
        
        Returns:
            Randomized date.
        """
        if pd.isna(value):
            return value
        
        if start_date is None:
            start_date = datetime.now() - timedelta(days=3650)  # 10 years ago
        if end_date is None:
            end_date = datetime.now()
        
        random_date = fake.date_between(start_date=start_date, end_date=end_date)
        
        # Preserve time component if original value has it
        if isinstance(value, pd.Timestamp) and value.hour != 0 or value.minute != 0 or value.second != 0:
            random_time = fake.time()
            random_date = datetime.combine(random_date.date(), datetime.strptime(random_time, "%H:%M:%S").time())
        
        return random_date
    
    def randomize_numeric(
        self,
        value: Any,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        distribution: str = "uniform"
    ) -> float:
        """
        Randomize a numeric value.
        
        Args:
            value: Original numeric value.
            min_value: Minimum value (uses 0 if None).
            max_value: Maximum value (uses 1000 if None).
            distribution: Distribution type ('uniform', 'normal').
        
        Returns:
            Randomized number.
        """
        if pd.isna(value):
            return value
        
        if min_value is None:
            min_value = 0.0
        if max_value is None:
            max_value = 1000.0
        
        if distribution == "uniform":
            return random.uniform(min_value, max_value)
        elif distribution == "normal":
            mean = (min_value + max_value) / 2
            std = (max_value - min_value) / 6
            return max(min_value, min(max_value, np.random.normal(mean, std)))
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
    
    def randomize_boolean(self, value: Any) -> bool:
        """Randomize a boolean value."""
        if pd.isna(value):
            return value
        return random.choice([True, False])
    
    def anonymize_column(
        self,
        df: pd.DataFrame,
        column: str,
        strategy: Union[str, Callable] = "randomize_string"
    ) -> pd.DataFrame:
        """
        Anonymize a specific column in a dataframe.
        
        Args:
            df: DataFrame to anonymize.
            column: Column name to anonymize.
            strategy: Anonymization strategy (string name or callable function).
        
        Returns:
            DataFrame with anonymized column.
        """
        if column not in df.columns:
            logger.warning(f"Column '{column}' not found in dataframe")
            return df
        
        df = df.copy()
        
        # Get strategy function
        if isinstance(strategy, str):
            strategy_map = {
                "randomize_string": self.randomize_string,
                "randomize_email": self.randomize_email,
                "randomize_name": self.randomize_name,
                "randomize_phone": self.randomize_phone,
                "randomize_address": self.randomize_address,
                "randomize_date": self.randomize_date,
                "randomize_numeric": self.randomize_numeric,
                "randomize_boolean": self.randomize_boolean,
            }
            
            if strategy not in strategy_map:
                raise ValueError(f"Unknown strategy: {strategy}. Available: {list(strategy_map.keys())}")
            
            strategy_func = strategy_map[strategy]
        else:
            strategy_func = strategy
        
        # Apply anonymization
        df[column] = df[column].apply(strategy_func)
        logger.info(f"Anonymized column '{column}' using strategy '{strategy}'")
        
        return df
    
    def anonymize_dataframe(
        self,
        df: pd.DataFrame,
        column_strategies: Optional[Dict[str, Union[str, Callable]]] = None,
        exclude_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Anonymize multiple columns in a dataframe.
        
        Args:
            df: DataFrame to anonymize.
            column_strategies: Dictionary mapping column names to strategies.
                              If None, auto-detects and anonymizes common PII columns.
            exclude_columns: List of columns to exclude from anonymization.
        
        Returns:
            Anonymized DataFrame.
        """
        df = df.copy()
        
        if exclude_columns is None:
            exclude_columns = []
        
        # Auto-detect PII columns if no strategies provided
        if column_strategies is None:
            column_strategies = {}
            
            # Common PII column patterns
            pii_patterns = {
                "email": ["email", "e_mail", "e-mail", "mail"],
                "name": ["name", "first_name", "last_name", "full_name", "customer_name", "user_name"],
                "phone": ["phone", "telephone", "mobile", "cell"],
                "address": ["address", "street", "city", "zip", "postal"],
                "ssn": ["ssn", "social_security"],
                "date": ["birth", "dob", "date_of_birth", "created_at", "updated_at"],
            }
            
            for col in df.columns:
                if col.lower() in exclude_columns:
                    continue
                
                col_lower = col.lower()
                
                # Check for email
                if any(pattern in col_lower for pattern in pii_patterns["email"]):
                    column_strategies[col] = "randomize_email"
                # Check for name
                elif any(pattern in col_lower for pattern in pii_patterns["name"]):
                    column_strategies[col] = "randomize_name"
                # Check for phone
                elif any(pattern in col_lower for pattern in pii_patterns["phone"]):
                    column_strategies[col] = "randomize_phone"
                # Check for address
                elif any(pattern in col_lower for pattern in pii_patterns["address"]):
                    column_strategies[col] = "randomize_address"
                # Check for dates
                elif any(pattern in col_lower for pattern in pii_patterns["date"]):
                    column_strategies[col] = "randomize_date"
                # Check for string columns that might contain PII
                elif df[col].dtype == "object" and col not in exclude_columns:
                    # Randomize long string columns (likely to contain PII)
                    sample_values = df[col].dropna().head(10)
                    if len(sample_values) > 0 and any(len(str(v)) > 20 for v in sample_values):
                        column_strategies[col] = "randomize_string"
        
        # Apply anonymization strategies
        for column, strategy in column_strategies.items():
            if column in df.columns and column not in exclude_columns:
                df = self.anonymize_column(df, column, strategy)
        
        logger.info(f"Anonymized {len(column_strategies)} columns")
        return df


def extract_and_anonymize(
    server: str,
    database: str,
    table_name: str,
    username: Optional[str] = None,
    password: Optional[str] = None,
    schema: Optional[str] = None,
    query: Optional[str] = None,
    anonymize: bool = True,
    column_strategies: Optional[Dict[str, Union[str, Callable]]] = None,
    exclude_columns: Optional[List[str]] = None,
    use_windows_auth: bool = False,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Convenience function to extract and anonymize data from SQL Server.
    
    Args:
        server: SQL Server hostname or IP address.
        database: Database name.
        table_name: Table name to extract.
        username: Username for authentication.
        password: Password for authentication.
        schema: Schema name.
        query: Custom SQL query (overrides table_name).
        anonymize: Whether to anonymize the data.
        column_strategies: Dictionary mapping column names to anonymization strategies.
        exclude_columns: List of columns to exclude from anonymization.
        use_windows_auth: Use Windows Authentication.
        seed: Random seed for reproducibility.
    
    Returns:
        Extracted and optionally anonymized DataFrame.
    """
    with SQLServerExtractor(
        server=server,
        database=database,
        username=username,
        password=password,
        use_windows_auth=use_windows_auth
    ) as extractor:
        df = extractor.extract_table(
            table_name=table_name,
            schema=schema,
            query=query
        )
    
    if anonymize:
        anonymizer = DataAnonymizer(seed=seed)
        df = anonymizer.anonymize_dataframe(
            df,
            column_strategies=column_strategies,
            exclude_columns=exclude_columns
        )
    
    return df


if __name__ == "__main__":
    # Example usage
    logger.info("SQL Server extractor module ready.")
    logger.info("Example usage:")
    logger.info("""
    # Extract and anonymize data
    from ingest.extract import extract_and_anonymize
    
    df = extract_and_anonymize(
        server="localhost",
        database="MyDatabase",
        table_name="Customers",
        username="user",
        password="password",
        anonymize=True
    )
    
    # Or use the classes directly
    from ingest.extract import SQLServerExtractor, DataAnonymizer
    
    with SQLServerExtractor(
        server="localhost",
        database="MyDatabase",
        username="user",
        password="password"
    ) as extractor:
        df = extractor.extract_table("Customers")
    
    anonymizer = DataAnonymizer(seed=42)
    df_anonymized = anonymizer.anonymize_dataframe(df)
    """)
