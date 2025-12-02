"""Data preprocessing and transformation utilities."""

import re
import unicodedata
from typing import Any, Callable, Dict, List, Optional

from slm_builder.config import PreprocessConfig
from slm_builder.utils import get_logger

logger = get_logger(__name__)


class Transform:
    """Base class for data transformations."""

    def apply(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply transformation to records.

        Args:
            records: List of records to transform

        Returns:
            Transformed records
        """
        raise NotImplementedError

    def __call__(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return self.apply(records)


class NormalizeText(Transform):
    """Normalize text content."""

    def __init__(
        self,
        lowercase: bool = False,
        strip_urls: bool = True,
        normalize_unicode: bool = True,
        normalize_whitespace: bool = True,
    ):
        """Initialize normalizer.

        Args:
            lowercase: Convert to lowercase
            strip_urls: Remove URLs
            normalize_unicode: Normalize unicode characters
            normalize_whitespace: Normalize whitespace
        """
        self.lowercase = lowercase
        self.strip_urls = strip_urls
        self.normalize_unicode = normalize_unicode
        self.normalize_whitespace = normalize_whitespace

    def apply(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply normalization."""
        for record in records:
            text = record.get("text", "")
            text = self._normalize(text)
            record["text"] = text

        return records

    def _normalize(self, text: str) -> str:
        """Normalize a single text string."""
        if not text:
            return text

        # Unicode normalization
        if self.normalize_unicode:
            text = unicodedata.normalize("NFKC", text)

        # Remove URLs
        if self.strip_urls:
            text = re.sub(
                r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
                "",
                text,
            )

        # Normalize whitespace
        if self.normalize_whitespace:
            text = re.sub(r"\s+", " ", text)
            text = text.strip()

        # Lowercase
        if self.lowercase:
            text = text.lower()

        return text


class Deduplicate(Transform):
    """Remove duplicate records."""

    def __init__(self, key_field: str = "text"):
        """Initialize deduplicator.

        Args:
            key_field: Field to use for deduplication
        """
        self.key_field = key_field

    def apply(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicates."""
        seen = set()
        unique_records = []

        for record in records:
            key = record.get(self.key_field)
            if key and key not in seen:
                seen.add(key)
                unique_records.append(record)

        removed = len(records) - len(unique_records)
        if removed > 0:
            logger.info("Removed duplicates", count=removed)

        return unique_records


class ChunkLongTexts(Transform):
    """Chunk long texts into smaller pieces."""

    def __init__(
        self,
        max_tokens: int = 512,
        overlap: int = 64,
        tokenizer: Optional[Any] = None,
    ):
        """Initialize chunker.

        Args:
            max_tokens: Maximum tokens per chunk
            overlap: Overlap tokens between chunks
            tokenizer: Optional tokenizer (uses simple word split if None)
        """
        self.max_tokens = max_tokens
        self.overlap = overlap
        self.tokenizer = tokenizer

    def apply(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk records."""
        chunked_records = []

        for record in records:
            text = record.get("text", "")

            if self.tokenizer:
                chunks = self._chunk_with_tokenizer(text)
            else:
                chunks = self._chunk_simple(text)

            # Create a record for each chunk
            for i, chunk in enumerate(chunks):
                chunk_record = record.copy()
                chunk_record["text"] = chunk
                chunk_record["id"] = f"{record['id']}_chunk{i}"
                chunk_record["metadata"] = record.get("metadata", {}).copy()
                chunk_record["metadata"]["chunk_index"] = i
                chunk_record["metadata"]["total_chunks"] = len(chunks)
                chunk_record["metadata"]["original_id"] = record["id"]
                chunked_records.append(chunk_record)

        if len(chunked_records) > len(records):
            logger.info("Chunked texts", original=len(records), chunked=len(chunked_records))

        return chunked_records

    def _chunk_simple(self, text: str) -> List[str]:
        """Chunk text using simple word splitting."""
        words = text.split()

        if len(words) <= self.max_tokens:
            return [text]

        chunks = []
        i = 0
        while i < len(words):
            chunk_words = words[i : i + self.max_tokens]
            chunks.append(" ".join(chunk_words))
            i += self.max_tokens - self.overlap

        return chunks

    def _chunk_with_tokenizer(self, text: str) -> List[str]:
        """Chunk text using a tokenizer."""
        tokens = self.tokenizer.encode(text)

        if len(tokens) <= self.max_tokens:
            return [text]

        chunks = []
        i = 0
        while i < len(tokens):
            chunk_tokens = tokens[i : i + self.max_tokens]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)
            i += self.max_tokens - self.overlap

        return chunks


class FilterByLength(Transform):
    """Filter records by text length."""

    def __init__(self, min_length: int = 10, max_length: Optional[int] = None):
        """Initialize filter.

        Args:
            min_length: Minimum text length (characters)
            max_length: Maximum text length (characters), None for no limit
        """
        self.min_length = min_length
        self.max_length = max_length

    def apply(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter by length."""
        filtered = []

        for record in records:
            text = record.get("text", "")
            length = len(text)

            if length < self.min_length:
                continue
            if self.max_length and length > self.max_length:
                continue

            filtered.append(record)

        removed = len(records) - len(filtered)
        if removed > 0:
            logger.info("Filtered by length", removed=removed, kept=len(filtered))

        return filtered


class TokenizeRecords(Transform):
    """Tokenize text in records."""

    def __init__(self, tokenizer: Any, max_length: int = 512):
        """Initialize tokenizer transform.

        Args:
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

    def apply(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Tokenize records."""
        for record in records:
            text = record.get("text", "")

            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                return_attention_mask=False,
            )

            record["tokens"] = encoding["input_ids"]

        logger.info("Tokenized records", count=len(records))
        return records


class ConvertToInstructionFormat(Transform):
    """Convert QA pairs to instruction format."""

    def __init__(self, system_message: Optional[str] = None):
        """Initialize converter.

        Args:
            system_message: Optional system message to prepend
        """
        self.system_message = system_message

    def apply(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert to instruction format."""
        converted = []

        for record in records:
            if record.get("task") == "qa" and record.get("label"):
                label = record["label"]

                instruction = label.get("question", "")
                response = label.get("answer", "")
                input_context = label.get("context")

                # Build instruction text
                parts = []
                if self.system_message:
                    parts.append(f"System: {self.system_message}")
                parts.append(f"Instruction: {instruction}")
                if input_context:
                    parts.append(f"Input: {input_context}")
                parts.append(f"Response: {response}")

                record["text"] = "\n".join(parts)
                record["task"] = "instruction"
                record["label"] = {
                    "instruction": instruction,
                    "response": response,
                    "input": input_context,
                    "system": self.system_message,
                }

            converted.append(record)

        return converted


class Pipeline:
    """Pipeline of transformations."""

    def __init__(self, transforms: List[Transform]):
        """Initialize pipeline.

        Args:
            transforms: List of Transform objects
        """
        self.transforms = transforms

    def apply(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply all transformations in sequence."""
        for transform in self.transforms:
            records = transform.apply(records)

        return records

    def __call__(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return self.apply(records)


def create_default_pipeline(config: PreprocessConfig, tokenizer: Optional[Any] = None) -> Pipeline:
    """Create default preprocessing pipeline from config.

    Args:
        config: Preprocessing configuration
        tokenizer: Optional tokenizer

    Returns:
        Preprocessing Pipeline
    """
    transforms = []

    # Normalization
    transforms.append(
        NormalizeText(
            lowercase=config.lowercase,
            strip_urls=config.strip_urls,
            normalize_unicode=config.normalize_unicode,
            normalize_whitespace=True,
        )
    )

    # Filter short texts
    transforms.append(FilterByLength(min_length=10))

    # Deduplication
    if config.remove_duplicates:
        transforms.append(Deduplicate())

    # Chunking
    if tokenizer:
        transforms.append(
            ChunkLongTexts(
                max_tokens=config.max_tokens_per_chunk,
                overlap=config.chunk_overlap,
                tokenizer=tokenizer,
            )
        )

    # Tokenization (if tokenizer provided)
    if tokenizer:
        transforms.append(
            TokenizeRecords(
                tokenizer=tokenizer,
                max_length=config.max_tokens_per_chunk,
            )
        )

    return Pipeline(transforms)
