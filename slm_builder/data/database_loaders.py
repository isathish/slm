"""Database loaders for SQL and NoSQL databases."""

from typing import Any, Dict, List, Optional

from slm_builder.data.schemas import (
    create_classification_record,
    create_instruction_record,
    create_qa_record,
    record_to_dict,
)
from slm_builder.utils import get_logger

logger = get_logger(__name__)


class DatabaseLoader:
    """Base class for database loaders."""

    def __init__(self, task: str = "qa"):
        """Initialize loader.

        Args:
            task: Task type
        """
        self.task = task

    def load(self, query: str, connection_params: Dict[str, Any], **kwargs) -> List[Dict[str, Any]]:
        """Load data from database.

        Args:
            query: SQL query or collection/table name
            connection_params: Database connection parameters
            **kwargs: Additional arguments

        Returns:
            List of records
        """
        raise NotImplementedError


class SQLLoader(DatabaseLoader):
    """Load data from SQL databases (PostgreSQL, MySQL, SQLite, etc.)."""

    def load(
        self,
        query: str,
        connection_params: Dict[str, Any],
        column_mapping: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Load data from SQL database.

        Args:
            query: SQL query
            connection_params: Connection params (host, port, database, user, password, etc.)
            column_mapping: Mapping of columns to canonical format
            **kwargs: Additional arguments

        Returns:
            List of records

        Example:
            connection_params = {
                'host': 'localhost',
                'port': 5432,
                'database': 'mydb',
                'user': 'user',
                'password': 'pass'
            }
            column_mapping = {
                'text': 'question_text',
                'label.answer': 'answer_text'
            }
        """
        try:
            from sqlalchemy import create_engine, text
        except ImportError:
            raise ImportError(
                "SQLAlchemy not installed. Install with: pip install sqlalchemy psycopg2-binary"
            )

        logger.info("Loading from SQL database", query=query[:100])

        # Build connection string based on dialect
        dialect = connection_params.get("dialect", "postgresql")
        user = connection_params.get("user")
        password = connection_params.get("password")
        host = connection_params.get("host", "localhost")
        port = connection_params.get("port")
        database = connection_params.get("database")

        if dialect == "sqlite":
            connection_string = f"sqlite:///{database}"
        else:
            auth = f"{user}:{password}@" if user and password else ""
            port_str = f":{port}" if port else ""
            connection_string = f"{dialect}://{auth}{host}{port_str}/{database}"

        # Create engine and execute query
        engine = create_engine(connection_string)

        with engine.connect() as connection:
            result = connection.execute(text(query))
            rows = result.fetchall()
            columns = result.keys()

        logger.info("SQL query executed", rows=len(rows), columns=len(columns))

        # Convert to records
        records = []
        for row in rows:
            row_dict = dict(zip(columns, row))

            # Apply column mapping if provided
            if column_mapping:
                row_dict = self._apply_column_mapping(row_dict, column_mapping)

            # Convert to canonical format
            record = self._row_to_record(row_dict)
            if record:
                records.append(record_to_dict(record))

        logger.info("SQL load complete", records=len(records))
        return records

    def _apply_column_mapping(self, row: Dict[str, Any], mapping: Dict[str, str]) -> Dict[str, Any]:
        """Apply column mapping to row."""
        mapped = {}
        for target, source in mapping.items():
            if source in row:
                # Handle nested keys (e.g., 'label.answer')
                if "." in target:
                    keys = target.split(".")
                    current = mapped
                    for key in keys[:-1]:
                        if key not in current:
                            current[key] = {}
                        current = current[key]
                    current[keys[-1]] = row[source]
                else:
                    mapped[target] = row[source]
        return mapped

    def _row_to_record(self, row: Dict[str, Any]) -> Optional[Any]:
        """Convert row to canonical record."""
        text = row.get("text", "")
        if not text:
            return None

        record_id = row.get("id", "")
        metadata = row.get("metadata", {})

        if self.task == "qa":
            question = row.get("label", {}).get("question", "")
            answer = row.get("label", {}).get("answer", "")
            return create_qa_record(text, question, answer, record_id, metadata)
        elif self.task == "classification":
            label = row.get("label", {}).get("label", "")
            return create_classification_record(text, label, record_id, metadata)
        elif self.task == "instruction":
            instruction = row.get("label", {}).get("instruction", "")
            response = row.get("label", {}).get("response", "")
            return create_instruction_record(text, instruction, response, record_id, metadata)
        else:
            # Generation task
            return create_qa_record(text, "", "", record_id, metadata)


class MongoDBLoader(DatabaseLoader):
    """Load data from MongoDB."""

    def load(
        self,
        collection_name: str,
        connection_params: Dict[str, Any],
        query_filter: Optional[Dict[str, Any]] = None,
        projection: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Load data from MongoDB.

        Args:
            collection_name: Collection name
            connection_params: Connection params (host, port, database, etc.)
            query_filter: MongoDB query filter
            projection: Fields to include/exclude
            limit: Maximum documents to fetch
            **kwargs: Additional arguments

        Returns:
            List of records

        Example:
            connection_params = {
                'host': 'localhost',
                'port': 27017,
                'database': 'mydb',
                'username': 'user',
                'password': 'pass'
            }
            query_filter = {'status': 'active'}
            projection = {'text': 1, 'label': 1, '_id': 0}
        """
        try:
            from pymongo import MongoClient
        except ImportError:
            raise ImportError("pymongo not installed. Install with: pip install pymongo")

        logger.info("Loading from MongoDB", collection=collection_name)

        # Build connection string
        host = connection_params.get("host", "localhost")
        port = connection_params.get("port", 27017)
        database = connection_params.get("database")
        username = connection_params.get("username")
        password = connection_params.get("password")

        if username and password:
            connection_string = f"mongodb://{username}:{password}@{host}:{port}/{database}"
        else:
            connection_string = f"mongodb://{host}:{port}/{database}"

        # Connect and query
        client = MongoClient(connection_string)
        db = client[database]
        collection = db[collection_name]

        query_filter = query_filter or {}
        cursor = collection.find(query_filter, projection)

        if limit:
            cursor = cursor.limit(limit)

        documents = list(cursor)
        client.close()

        logger.info("MongoDB query executed", documents=len(documents))

        # Convert to records
        records = []
        for doc in documents:
            record = self._doc_to_record(doc)
            if record:
                records.append(record_to_dict(record))

        logger.info("MongoDB load complete", records=len(records))
        return records

    def _doc_to_record(self, doc: Dict[str, Any]) -> Optional[Any]:
        """Convert MongoDB document to canonical record."""
        text = doc.get("text", "")
        if not text:
            return None

        record_id = str(doc.get("_id", ""))
        metadata = doc.get("metadata", {})

        if self.task == "qa":
            label = doc.get("label", {})
            question = label.get("question", "")
            answer = label.get("answer", "")
            return create_qa_record(text, question, answer, record_id, metadata)
        elif self.task == "classification":
            label = doc.get("label", {})
            label_text = label.get("label", "")
            return create_classification_record(text, label_text, record_id, metadata)
        elif self.task == "instruction":
            label = doc.get("label", {})
            instruction = label.get("instruction", "")
            response = label.get("response", "")
            return create_instruction_record(text, instruction, response, record_id, metadata)
        else:
            return create_qa_record(text, "", "", record_id, metadata)


def load_from_sql(
    query: str,
    connection_params: Dict[str, Any],
    task: str = "qa",
    column_mapping: Optional[Dict[str, str]] = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    """Convenience function to load from SQL database.

    Args:
        query: SQL query
        connection_params: Database connection parameters
        task: Task type
        column_mapping: Column mapping
        **kwargs: Additional arguments

    Returns:
        List of records
    """
    loader = SQLLoader(task=task)
    return loader.load(query, connection_params, column_mapping, **kwargs)


def load_from_mongodb(
    collection_name: str,
    connection_params: Dict[str, Any],
    task: str = "qa",
    query_filter: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    """Convenience function to load from MongoDB.

    Args:
        collection_name: Collection name
        connection_params: Connection parameters
        task: Task type
        query_filter: Query filter
        **kwargs: Additional arguments

    Returns:
        List of records
    """
    loader = MongoDBLoader(task=task)
    return loader.load(collection_name, connection_params, query_filter, **kwargs)
