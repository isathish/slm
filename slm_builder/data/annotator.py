"""Annotation subsystem with Streamlit UI."""

import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from slm_builder.utils import get_logger, load_jsonl, save_jsonl

logger = get_logger(__name__)


class Annotator:
    """Annotation interface for labeling data."""

    def __init__(self, task: str = "qa"):
        """Initialize annotator.
        
        Args:
            task: Task type (qa, classification, etc.)
        """
        self.task = task

    def launch(
        self,
        records: List[Dict[str, Any]],
        output_path: Optional[str] = None,
        port: int = 8501,
        auto_launch: bool = True,
    ) -> str:
        """Launch Streamlit annotation UI.
        
        Args:
            records: Records to annotate
            output_path: Path to save annotated records
            port: Port for Streamlit server
            auto_launch: Whether to auto-launch browser
            
        Returns:
            Path to annotated records file
        """
        if output_path is None:
            output_path = "annotated_data.jsonl"

        # Create temporary app file
        app_code = self._generate_streamlit_app(records, output_path)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(app_code)
            app_file = f.name

        logger.info(
            "Starting annotation UI",
            port=port,
            output=output_path,
            records=len(records)
        )
        
        # Launch Streamlit
        try:
            subprocess.run([
                "streamlit", "run",
                app_file,
                "--server.port", str(port),
                "--server.headless", "false" if auto_launch else "true",
            ])
        except KeyboardInterrupt:
            logger.info("Annotation UI stopped")
        finally:
            # Cleanup
            Path(app_file).unlink(missing_ok=True)

        return output_path

    def _generate_streamlit_app(
        self, records: List[Dict[str, Any]], output_path: str
    ) -> str:
        """Generate Streamlit app code for annotation."""
        # Save records to temp file for the app to load
        temp_records_path = str(Path(output_path).parent / ".temp_records.jsonl")
        save_jsonl(records, temp_records_path)

        app_template = f'''
import json
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="SLM Builder Annotator", layout="wide")

# Load data
RECORDS_FILE = "{temp_records_path}"
OUTPUT_FILE = "{output_path}"
TASK = "{self.task}"

@st.cache_data
def load_records():
    records = []
    with open(RECORDS_FILE, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def save_annotations(records):
    with open(OUTPUT_FILE, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\\n")
    st.success(f"Saved {{len(records)}} annotations to {{OUTPUT_FILE}}")

# Initialize session state
if "records" not in st.session_state:
    st.session_state.records = load_records()
    st.session_state.current_idx = 0
    st.session_state.annotated = [False] * len(st.session_state.records)

records = st.session_state.records
current_idx = st.session_state.current_idx

# Header
st.title("🏷️ SLM Builder Annotator")
st.markdown(f"**Task:** {{TASK}} | **Progress:** {{sum(st.session_state.annotated)}}/{{len(records)}}")

# Navigation
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    if st.button("⬅️ Previous") and current_idx > 0:
        st.session_state.current_idx -= 1
        st.rerun()
with col2:
    st.progress(sum(st.session_state.annotated) / len(records))
with col3:
    if st.button("Next ➡️") and current_idx < len(records) - 1:
        st.session_state.current_idx += 1
        st.rerun()

st.divider()

# Current record
if records:
    record = records[current_idx]
    
    st.subheader(f"Record {{current_idx + 1}} / {{len(records)}}")
    st.markdown(f"**ID:** `{{record.get('id', 'N/A')}}`")
    
    # Display text
    st.text_area("Text", value=record.get("text", ""), height=150, disabled=True)
    
    # Annotation fields based on task
    if TASK == "qa":
        question = st.text_input("Question", value=record.get("label", {{}}).get("question", ""))
        answer = st.text_area("Answer", value=record.get("label", {{}}).get("answer", ""), height=100)
        
        if st.button("💾 Save Annotation"):
            record["label"] = {{"question": question, "answer": answer}}
            st.session_state.annotated[current_idx] = True
            st.success("Saved!")
    
    elif TASK == "classification":
        label = st.text_input("Label/Category", value=record.get("label", {{}}).get("label", ""))
        
        if st.button("💾 Save Annotation"):
            record["label"] = {{"label": label}}
            st.session_state.annotated[current_idx] = True
            st.success("Saved!")
    
    elif TASK == "instruction":
        instruction = st.text_area("Instruction", value=record.get("label", {{}}).get("instruction", ""), height=80)
        response = st.text_area("Response", value=record.get("label", {{}}).get("response", ""), height=100)
        
        if st.button("💾 Save Annotation"):
            record["label"] = {{"instruction": instruction, "response": response}}
            st.session_state.annotated[current_idx] = True
            st.success("Saved!")
    
    # Metadata
    with st.expander("Metadata"):
        st.json(record.get("metadata", {{}}))

# Sidebar
with st.sidebar:
    st.header("Actions")
    
    if st.button("💾 Export All Annotations"):
        save_annotations(records)
    
    st.markdown("---")
    st.header("Statistics")
    st.metric("Total Records", len(records))
    st.metric("Annotated", sum(st.session_state.annotated))
    st.metric("Remaining", len(records) - sum(st.session_state.annotated))
    
    st.markdown("---")
    st.markdown("**Jump to Record**")
    jump_to = st.number_input("Record #", min_value=1, max_value=len(records), value=current_idx + 1)
    if st.button("Go"):
        st.session_state.current_idx = jump_to - 1
        st.rerun()
'''
        return app_template


def annotate_dataset(
    records: List[Dict[str, Any]],
    task: str = "qa",
    output_path: Optional[str] = None,
    **kwargs
) -> str:
    """Convenience function to launch annotation UI.
    
    Args:
        records: Records to annotate
        task: Task type
        output_path: Output file path
        **kwargs: Additional arguments for Annotator.launch
        
    Returns:
        Path to annotated records
    """
    annotator = Annotator(task=task)
    return annotator.launch(records, output_path=output_path, **kwargs)
