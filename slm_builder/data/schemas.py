"""Canonical dataset schemas for SLM Builder."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DatasetRecord(BaseModel):
    """Canonical dataset record format.

    This is the standard format used throughout the SLM Builder pipeline.
    All loaders must convert their data to this format.
    """

    id: str = Field(description="Unique record identifier")
    text: str = Field(description="Full text or prompt")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Source metadata (file, url, topic, etc.)"
    )
    task: str = Field(default="qa", description="Task type")
    label: Optional[Dict[str, Any]] = Field(
        default=None, description="Task-specific label/annotation"
    )
    tokens: Optional[List[int]] = Field(
        default=None, description="Tokenized representation (populated during preprocessing)"
    )

    class Config:
        # Allow extra fields for extensibility
        extra = "allow"


class QALabel(BaseModel):
    """Label schema for QA tasks."""

    answer: str = Field(description="Answer text")
    question: Optional[str] = Field(default=None, description="Question text")
    start: Optional[int] = Field(default=None, description="Answer start position")
    end: Optional[int] = Field(default=None, description="Answer end position")
    context: Optional[str] = Field(default=None, description="Context/passage")


class ClassificationLabel(BaseModel):
    """Label schema for classification tasks."""

    label: str = Field(description="Class label")
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    label_id: Optional[int] = Field(default=None)


class GenerationLabel(BaseModel):
    """Label schema for generation tasks."""

    target: str = Field(description="Target generation text")
    prefix: Optional[str] = Field(default=None, description="Generation prefix/prompt")


class InstructionLabel(BaseModel):
    """Label schema for instruction-tuning tasks."""

    instruction: str = Field(description="Instruction/prompt")
    response: str = Field(description="Expected response")
    input: Optional[str] = Field(default=None, description="Optional input context")
    system: Optional[str] = Field(default=None, description="System message")


# Task-specific label schemas
TASK_LABEL_SCHEMAS = {
    "qa": QALabel,
    "classification": ClassificationLabel,
    "generation": GenerationLabel,
    "instruction": InstructionLabel,
}


def validate_record(record: Dict[str, Any], task: str) -> DatasetRecord:
    """Validate a record against canonical schema.

    Args:
        record: Record dictionary
        task: Task type

    Returns:
        Validated DatasetRecord

    Raises:
        ValidationError: If record is invalid
    """
    # Ensure task is set
    if "task" not in record:
        record["task"] = task

    # Validate label if present
    if record.get("label") and task in TASK_LABEL_SCHEMAS:
        label_schema = TASK_LABEL_SCHEMAS[task]
        try:
            record["label"] = label_schema(**record["label"]).model_dump()
        except Exception:
            pass  # Allow flexible labels

    return DatasetRecord(**record)


def record_to_dict(record: DatasetRecord) -> Dict[str, Any]:
    """Convert DatasetRecord to dictionary.

    Args:
        record: DatasetRecord instance

    Returns:
        Dictionary representation
    """
    return record.model_dump(exclude_none=True)


def create_qa_record(
    id: str,
    question: str,
    answer: str,
    context: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> DatasetRecord:
    """Helper to create a QA record.

    Args:
        id: Record ID
        question: Question text
        answer: Answer text
        context: Optional context/passage
        metadata: Optional metadata

    Returns:
        DatasetRecord for QA task
    """
    text = f"Question: {question}\nAnswer: {answer}"
    if context:
        text = f"Context: {context}\n{text}"

    return DatasetRecord(
        id=id,
        text=text,
        metadata=metadata or {},
        task="qa",
        label={
            "question": question,
            "answer": answer,
            "context": context,
        },
    )


def create_instruction_record(
    id: str,
    instruction: str,
    response: str,
    input: Optional[str] = None,
    system: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> DatasetRecord:
    """Helper to create an instruction-tuning record.

    Args:
        id: Record ID
        instruction: Instruction/prompt
        response: Expected response
        input: Optional input context
        system: Optional system message
        metadata: Optional metadata

    Returns:
        DatasetRecord for instruction task
    """
    text_parts = []
    if system:
        text_parts.append(f"System: {system}")
    text_parts.append(f"Instruction: {instruction}")
    if input:
        text_parts.append(f"Input: {input}")
    text_parts.append(f"Response: {response}")

    text = "\n".join(text_parts)

    return DatasetRecord(
        id=id,
        text=text,
        metadata=metadata or {},
        task="instruction",
        label={
            "instruction": instruction,
            "response": response,
            "input": input,
            "system": system,
        },
    )


def create_classification_record(
    id: str,
    text: str,
    label: str,
    label_id: Optional[int] = None,
    confidence: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> DatasetRecord:
    """Helper to create a classification record.

    Args:
        id: Record ID
        text: Input text
        label: Class label
        label_id: Optional numeric label ID
        confidence: Optional confidence score
        metadata: Optional metadata

    Returns:
        DatasetRecord for classification task
    """
    return DatasetRecord(
        id=id,
        text=text,
        metadata=metadata or {},
        task="classification",
        label={
            "label": label,
            "label_id": label_id,
            "confidence": confidence,
        },
    )
