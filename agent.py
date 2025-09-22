import os
import sys
import argparse
import logging
import subprocess
import time
from pathlib import Path
from typing import TypedDict, Optional

import pandas as pd
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
import google.generativeai as genai

# --- 1. Configuration and Setup ---

# Load environment variables from .env file
load_dotenv()

# Configure logging to show timestamps and log levels
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Configure Google Gemini API from environment variables
try:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")
    genai.configure(api_key=api_key)
    llm = genai.GenerativeModel("gemini-1.5-flash")
except Exception as e:
    logger.error(f"Failed to configure Gemini API: {e}")
    sys.exit(1)

# Project constants
ROOT_DIR = Path(__file__).parent.resolve()
DATA_DIR = ROOT_DIR / "data"
PARSERS_DIR = ROOT_DIR / "custom_parsers"
PARSERS_DIR.mkdir(exist_ok=True)
MAX_RETRIES = 3

# --- 2. Typed Agent State ---

class AgentState(TypedDict):
    """Defines the structure of our agent's state for type safety."""
    target_bank: str
    pdf_path: Path
    csv_path: Path
    attempt: int
    generated_code: str
    parser_path: Path
    error_feedback: Optional[str]

# --- 3. Tool Node: Code Execution and Validation ---

def execute_and_validate_node(state: AgentState) -> dict:
    """
    A robust tool that writes, runs the generated code in a separate process,
    and validates its output. This function does not use an LLM.
    """
    logger.info("Executing and validating the generated code...")
    parser_path = state["parser_path"]
    pdf_path = state["pdf_path"]
    csv_path = state["csv_path"]

    # Write the generated code to a file
    try:
        parser_path.write_text(state["generated_code"])
    except IOError as e:
        return {"error_feedback": f"Failed to write parser file: {e}"}

    # Execute the parser in a secure, isolated subprocess
    try:
        output_csv_path = str(csv_path) + ".output"
        subprocess.run(
            [sys.executable, str(parser_path), str(pdf_path), output_csv_path],
            check=True, capture_output=True, text=True, timeout=60
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        error_output = e.stderr if hasattr(e, 'stderr') else str(e)
        logger.error(f"Code execution failed:\n{error_output}")
        return {"error_feedback": f"Code execution failed:\n{error_output}"}

    # Validate the output by comparing the generated CSV with the expected one
    try:
        generated_df = pd.read_csv(output_csv_path)
        expected_df = pd.read_csv(csv_path)

        # Normalize DataFrames for a fair comparison (handle dtypes and NaNs)
        generated_df = generated_df.astype(str).fillna('')
        expected_df = expected_df.astype(str).fillna('')

        if generated_df.equals(expected_df):
            logger.info("✅ Validation successful: DataFrames match.")
            os.remove(output_csv_path) # Clean up temporary file
            return {"error_feedback": None}
        else:
            diff_report = f"""
Parsed DataFrame does not match the expected one.
- Parsed Shape: {generated_df.shape} | Expected Shape: {expected_df.shape}
- Parsed Columns: {generated_df.columns.tolist()}
- Expected Columns: {expected_df.columns.tolist()}

First 5 Parsed Rows:
{generated_df.head().to_string()}

First 5 Expected Rows:
{expected_df.head().to_string()}
"""
            logger.warning("Validation failed: DataFrames do not match.")
            os.remove(output_csv_path) # Clean up temporary file
            return {"error_feedback": diff_report}
    except FileNotFoundError:
        return {"error_feedback": "Validation error: The parser script did not create the output CSV file."}
    except Exception as e:
        return {"error_feedback": f"Validation failed with an unexpected error: {e}"}


# --- 4. LLM Node: Code Generation and Correction ---

PROMPT_TEMPLATE = """
You are an expert Python developer. Your task is to write a standalone Python script to parse a bank statement PDF.

**Primary Instructions:**
1.  You **MUST** use the `pdfplumber` library for PDF processing and `pandas` for data handling.
2.  The script must be a runnable file that accepts an input PDF path and an output CSV path as command-line arguments.
3.  **CRITICAL:** The PDF may have multiple pages. Your script **MUST** iterate through `pdf.pages`, extract tables from **EVERY PAGE**, and concatenate them into one final DataFrame.
4.  The final DataFrame **MUST** have these exact columns: {columns}
5.  After processing, save the DataFrame to the output CSV path without the index.

**Sample of the target CSV data:**
```csv
{csv_sample}
```

**Self-Correction Guidance:**
{feedback}

This is attempt {attempt}. Provide only the complete, runnable Python script, including the `if __name__ == "__main__":` block. Do not add any explanations or markdown.
"""

def generate_code_node(state: AgentState) -> dict:
    """Generates Python code based on the prompt, incorporating feedback from previous failures."""
    logger.info(f"--- Generating code for '{state['target_bank']}' (Attempt {state['attempt']}) ---")

    csv_df = pd.read_csv(state["csv_path"], nrows=5)
    csv_sample = csv_df.to_string(index=False)
    columns = csv_df.columns.tolist()

    if state["error_feedback"]:
        feedback = f"The previous attempt failed. Analyze this error report and generate a corrected script:\n**Error Report:**\n{state['error_feedback']}"
    else:
        feedback = "This is the first attempt. Please generate the initial version of the script."

    prompt = PROMPT_TEMPLATE.format(
        columns=columns, csv_sample=csv_sample, feedback=feedback, attempt=state['attempt']
    )

    response = llm.generate_content(prompt)
    generated_code = response.text.strip().replace("```python", "").replace("```", "")
    
    parser_path = PARSERS_DIR / f"{state['target_bank']}_parser.py"
    return {"generated_code": generated_code, "parser_path": parser_path}


# --- 5. Graph Control Flow ---

def initialize_attempt_counter(state: AgentState) -> dict:
    """Sets the initial attempt number."""
    return {"attempt": 1}

def decision_node(state: AgentState) -> str:
    """Decides whether to finish or retry based on the validation result."""
    if state["error_feedback"] is None:
        logger.info("✅ Agent finished successfully!")
        return END

    logger.error(f"VALIDATION FAILED (Attempt {state['attempt']}). REASON:\n{state['error_feedback']}")
    if state["attempt"] >= MAX_RETRIES:
        logger.error(f"❌ Agent failed after {MAX_RETRIES} attempts.")
        return END
    
    return "retry"

def retry_node(state: AgentState) -> dict:
    """Increments the attempt counter and waits before retrying to respect API rate limits."""
    logger.info("Waiting for 60 seconds before retrying...")
    time.sleep(60)
    return {"attempt": state["attempt"] + 1}

# --- 6. Graph Assembly ---

graph_builder = StateGraph(AgentState)

graph_builder.add_node("start", initialize_attempt_counter)
graph_builder.add_node("generate_code", generate_code_node)
graph_builder.add_node("execute_and_validate", execute_and_validate_node)
graph_builder.add_node("retry", retry_node)

graph_builder.set_entry_point("start")
graph_builder.add_edge("start", "generate_code")
graph_builder.add_edge("generate_code", "execute_and_validate")
graph_builder.add_conditional_edges(
    "execute_and_validate", decision_node, {END: END, "retry": "retry"}
)
graph_builder.add_edge("retry", "generate_code")

app = graph_builder.compile()

# --- 7. CLI Entrypoint ---

def main():
    parser = argparse.ArgumentParser(description="An AI agent that writes PDF parsers.")
    parser.add_argument(
        "--target", type=str, required=True,
        help="The target bank folder name inside the 'data' directory (e.g., 'icici')."
    )
    args = parser.parse_args()
    target_bank = args.target.lower()

    try:
        pdf_path = next((DATA_DIR / target_bank).glob("*.pdf"))
        csv_path = next((DATA_DIR / target_bank).glob("*.csv"))
    except StopIteration:
        logger.error(f"No PDF/CSV files found for target '{target_bank}' in {DATA_DIR / target_bank}")
        sys.exit(1)

    initial_state = AgentState(
        target_bank=target_bank, pdf_path=pdf_path, csv_path=csv_path,
        attempt=0, generated_code="", parser_path=Path(), error_feedback=None,
    )

    final_state = app.invoke(initial_state)

    if final_state.get("error_feedback"):
        logger.error("Agent failed to generate a valid parser. Please check the logs.")
        sys.exit(1)

if __name__ == "__main__":
    main()
