"""Dataset loaders for various source formats."""

import csv
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from tqdm import tqdm

from slm_builder.data.schemas import (
    DatasetRecord,
    create_classification_record,
    create_instruction_record,
    create_qa_record,
    record_to_dict,
)
from slm_builder.utils import get_logger, validate_column_mapping, validate_file_exists

logger = get_logger(__name__)


def generate_id(text: str, prefix: str = "") -> str:
    """Generate a unique ID for a record.

    Args:
        text: Text to hash
        prefix: Optional prefix for ID

    Returns:
        Unique ID string
    """
    hash_val = hashlib.md5(text.encode()).hexdigest()[:12]
    return f"{prefix}{hash_val}" if prefix else hash_val


class DataLoader:
    """Base class for data loaders."""

    def __init__(self, task: str = "qa"):
        """Initialize loader.

        Args:
            task: Task type (qa, classification, generation, instruction)
        """
        self.task = task

    def load(self, source: str, **kwargs) -> List[Dict[str, Any]]:
        """Load data from source.

        Args:
            source: Data source (path, URL, etc.)
            **kwargs: Loader-specific arguments

        Returns:
            List of records in canonical format
        """
        raise NotImplementedError


class CSVLoader(DataLoader):
    """Load data from CSV files."""

    def load(
        self,
        source: str,
        column_mapping: Optional[Dict[str, str]] = None,
        delimiter: str = ",",
        encoding: str = "utf-8",
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Load CSV file.

        Args:
            source: Path to CSV file
            column_mapping: Mapping of canonical fields to CSV columns
                For QA: {"question": "q_col", "answer": "a_col"}
                For classification: {"text": "text_col", "label": "label_col"}
            delimiter: CSV delimiter
            encoding: File encoding

        Returns:
            List of records
        """
        file_path = validate_file_exists(source)

        with open(file_path, "r", encoding=encoding) as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            columns = reader.fieldnames

            logger.info("Loading CSV", path=source, columns=list(columns))

            # Auto-detect column mapping if not provided
            if column_mapping is None:
                column_mapping = self._auto_detect_columns(columns)
            else:
                # Validate provided mapping
                if self.task == "qa":
                    required = ["question", "answer"]
                elif self.task == "classification":
                    required = ["text", "label"]
                elif self.task == "instruction":
                    required = ["instruction", "response"]
                else:
                    required = ["text"]

                validate_column_mapping(columns, column_mapping, required)

            records = []
            for i, row in enumerate(tqdm(reader, desc="Loading CSV")):
                try:
                    record = self._row_to_record(row, column_mapping, i)
                    records.append(record_to_dict(record))
                except Exception as e:
                    logger.warning("Failed to parse row", row=i, error=str(e))

            logger.info("Loaded CSV", count=len(records))
            return records

    def _auto_detect_columns(self, columns: List[str]) -> Dict[str, str]:
        """Auto-detect column mapping based on common patterns."""
        columns_lower = [c.lower() for c in columns]
        mapping = {}

        # Common patterns for different tasks
        if self.task == "qa":
            for q_pattern in ["question", "q", "query", "input"]:
                if q_pattern in columns_lower:
                    mapping["question"] = columns[columns_lower.index(q_pattern)]
                    break

            for a_pattern in ["answer", "a", "response", "output"]:
                if a_pattern in columns_lower:
                    mapping["answer"] = columns[columns_lower.index(a_pattern)]
                    break

            for c_pattern in ["context", "passage", "document"]:
                if c_pattern in columns_lower:
                    mapping["context"] = columns[columns_lower.index(c_pattern)]
                    break

        elif self.task == "classification":
            for t_pattern in ["text", "content", "input", "sentence"]:
                if t_pattern in columns_lower:
                    mapping["text"] = columns[columns_lower.index(t_pattern)]
                    break

            for l_pattern in ["label", "class", "category", "target"]:
                if l_pattern in columns_lower:
                    mapping["label"] = columns[columns_lower.index(l_pattern)]
                    break

        elif self.task == "instruction":
            for i_pattern in ["instruction", "prompt", "input"]:
                if i_pattern in columns_lower:
                    mapping["instruction"] = columns[columns_lower.index(i_pattern)]
                    break

            for r_pattern in ["response", "output", "answer", "completion"]:
                if r_pattern in columns_lower:
                    mapping["response"] = columns[columns_lower.index(r_pattern)]
                    break

        return mapping

    def _row_to_record(
        self, row: Dict[str, str], mapping: Dict[str, str], index: int
    ) -> DatasetRecord:
        """Convert CSV row to canonical record."""
        metadata = {"source": "csv", "row_index": index}
        record_id = generate_id(str(row), prefix="csv_")

        if self.task == "qa":
            question = row.get(mapping.get("question", ""), "")
            answer = row.get(mapping.get("answer", ""), "")
            context = row.get(mapping.get("context", ""), None)

            return create_qa_record(
                id=record_id, question=question, answer=answer, context=context, metadata=metadata
            )

        elif self.task == "classification":
            text = row.get(mapping.get("text", ""), "")
            label = row.get(mapping.get("label", ""), "")

            return create_classification_record(
                id=record_id, text=text, label=label, metadata=metadata
            )

        elif self.task == "instruction":
            instruction = row.get(mapping.get("instruction", ""), "")
            response = row.get(mapping.get("response", ""), "")
            input_text = row.get(mapping.get("input", ""), None)

            return create_instruction_record(
                id=record_id,
                instruction=instruction,
                response=response,
                input=input_text,
                metadata=metadata,
            )

        else:
            # Generic record
            text = " ".join([v for v in row.values() if v])
            return DatasetRecord(id=record_id, text=text, metadata=metadata, task=self.task)


class JSONLLoader(DataLoader):
    """Load data from JSONL files."""

    def load(
        self, source: str, max_records: Optional[int] = None, **kwargs
    ) -> List[Dict[str, Any]]:
        """Load JSONL file.

        Args:
            source: Path to JSONL file
            max_records: Optional limit on records to load

        Returns:
            List of records
        """
        import json

        file_path = validate_file_exists(source)
        records = []

        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(tqdm(f, desc="Loading JSONL")):
                if max_records and i >= max_records:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    record = self._dict_to_record(data, i)
                    records.append(record_to_dict(record))
                except Exception as e:
                    logger.warning("Failed to parse line", line=i, error=str(e))

        logger.info("Loaded JSONL", path=source, count=len(records))
        return records

    def _dict_to_record(self, data: Dict[str, Any], index: int) -> DatasetRecord:
        """Convert dict to canonical record."""
        # If already in canonical format, use it
        if "id" in data and "text" in data:
            if "task" not in data:
                data["task"] = self.task
            return DatasetRecord(**data)

        # Otherwise, convert based on task
        metadata = data.get("metadata", {"source": "jsonl", "index": index})
        record_id = data.get("id", generate_id(str(data), prefix="jsonl_"))

        if self.task == "qa" and "question" in data and "answer" in data:
            return create_qa_record(
                id=record_id,
                question=data["question"],
                answer=data["answer"],
                context=data.get("context"),
                metadata=metadata,
            )
        elif self.task == "classification" and "text" in data and "label" in data:
            return create_classification_record(
                id=record_id, text=data["text"], label=data["label"], metadata=metadata
            )
        elif self.task == "instruction" and "instruction" in data and "response" in data:
            return create_instruction_record(
                id=record_id,
                instruction=data["instruction"],
                response=data["response"],
                input=data.get("input"),
                metadata=metadata,
            )
        else:
            # Generic fallback
            text = data.get("text", str(data))
            return DatasetRecord(
                id=record_id, text=text, metadata=metadata, task=self.task, label=data.get("label")
            )


class TextDirLoader(DataLoader):
    """Load text files from a directory."""

    def load(
        self,
        source: str,
        pattern: str = "*.txt",
        recursive: bool = True,
        encoding: str = "utf-8",
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Load text files from directory.

        Args:
            source: Directory path
            pattern: File pattern to match
            recursive: Whether to search recursively
            encoding: File encoding

        Returns:
            List of records (one per file)
        """
        dir_path = Path(source)
        if not dir_path.is_dir():
            raise ValueError(f"Not a directory: {source}")

        # Find matching files
        if recursive:
            files = list(dir_path.rglob(pattern))
        else:
            files = list(dir_path.glob(pattern))

        logger.info("Found text files", count=len(files), pattern=pattern)

        records = []
        for file_path in tqdm(files, desc="Loading text files"):
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    text = f.read()

                record_id = generate_id(str(file_path), prefix="txt_")
                metadata = {
                    "source": "text_file",
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                }

                record = DatasetRecord(id=record_id, text=text, metadata=metadata, task=self.task)
                records.append(record_to_dict(record))
            except Exception as e:
                logger.warning("Failed to load file", path=str(file_path), error=str(e))

        logger.info("Loaded text files", count=len(records))
        return records


class URLLoader(DataLoader):
    """Load content from URLs."""

    def load(
        self, source: str, max_pages: int = 10, extract_text: bool = True, **kwargs
    ) -> List[Dict[str, Any]]:
        """Load content from URL(s).

        Args:
            source: URL or file with list of URLs
            max_pages: Maximum pages to scrape
            extract_text: Whether to extract text from HTML

        Returns:
            List of records
        """
        # Check if source is a URL or file
        if source.startswith(("http://", "https://")):
            urls = [source]
        else:
            # Read URLs from file
            with open(source, "r") as f:
                urls = [line.strip() for line in f if line.strip()]

        urls = urls[:max_pages]
        logger.info("Loading URLs", count=len(urls))

        records = []
        for url in tqdm(urls, desc="Fetching URLs"):
            try:
                record = self._fetch_url(url, extract_text)
                if record:
                    records.append(record_to_dict(record))
            except Exception as e:
                logger.warning("Failed to fetch URL", url=url, error=str(e))

        logger.info("Loaded URLs", count=len(records))
        return records

    def _fetch_url(self, url: str, extract_text: bool) -> Optional[DatasetRecord]:
        """Fetch content from a single URL."""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            if extract_text and "text/html" in response.headers.get("Content-Type", ""):
                # Basic HTML text extraction
                text = self._extract_html_text(response.text)
            else:
                text = response.text

            record_id = generate_id(url, prefix="url_")
            metadata = {
                "source": "url",
                "url": url,
                "content_type": response.headers.get("Content-Type"),
            }

            return DatasetRecord(id=record_id, text=text, metadata=metadata, task=self.task)
        except Exception as e:
            logger.error("URL fetch failed", url=url, error=str(e))
            return None

    def _extract_html_text(self, html: str) -> str:
        """Extract text from HTML (basic implementation)."""
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = " ".join(chunk for chunk in chunks if chunk)

            return text
        except ImportError:
            # BeautifulSoup not available, use regex fallback
            import re

            text = re.sub(r"<[^>]+>", "", html)
            text = re.sub(r"\s+", " ", text)
            return text.strip()


# Factory function
def get_loader(source: str, task: str = "qa", **kwargs) -> DataLoader:
    """Get appropriate loader for source.

    Args:
        source: Data source path or URL
        task: Task type
        **kwargs: Loader-specific arguments

    Returns:
        Appropriate DataLoader instance
    """
    source_lower = source.lower()

    if source_lower.endswith(".csv"):
        return CSVLoader(task=task)
    elif source_lower.endswith(".jsonl") or source_lower.endswith(".json"):
        return JSONLLoader(task=task)
    elif source_lower.startswith(("http://", "https://")):
        return URLLoader(task=task)
    elif Path(source).is_dir():
        return TextDirLoader(task=task)
    else:
        # Try to detect format
        path = Path(source)
        if path.is_file():
            # Check first line
            try:
                with open(path, "r") as f:
                    first_line = f.readline()
                    if first_line.strip().startswith("{"):
                        return JSONLLoader(task=task)
            except Exception:
                pass

        raise ValueError(f"Could not determine loader for source: {source}")


def load_dataset(source: str, task: str = "qa", **kwargs) -> List[Dict[str, Any]]:
    """Load dataset from any supported source.

    Args:
        source: Data source (file path, directory, URL)
        task: Task type
        **kwargs: Loader-specific arguments

    Returns:
        List of records in canonical format
    """
    loader = get_loader(source, task=task, **kwargs)
    return loader.load(source, **kwargs)
