import os
import sys
import argparse
import logging
from pathlib import Path
import importlib.util
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

import json
from typing import Tuple, Optional

# --- Basic Setup ---

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("pdfminer").setLevel(logging.WARNING)
logging.getLogger("pdfplumber").setLevel(logging.WARNING)
logging.getLogger("camelot").setLevel(logging.WARNING)


logger = logging.getLogger("agent")

ROOT = Path(".").resolve()
DATA_DIR = ROOT / "data"
PARSERS_DIR = ROOT / "custom_parsers"
PARSERS_DIR.mkdir(exist_ok=True)

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
llm = genai.GenerativeModel("gemini-1.5-flash")

# --- Helper Functions ---

def find_sample_files(target: str):
    d = DATA_DIR / target
    pdfs = list(d.glob("*.pdf"))
    csvs = list(d.glob("*.csv"))
    if not pdfs or not csvs:
        raise FileNotFoundError(f"No PDF/CSV pair found in {d}")
    return pdfs[0], csvs[0]

def import_parser_module(file_path: Path):
    spec = importlib.util.spec_from_file_location("parser_module", str(file_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules["parser_module"] = module
    spec.loader.exec_module(module)
    return module

def validate_parser(parser_path: Path, pdf_path: Path, csv_path: Path) -> Tuple[bool, Optional[str]]:
    try:
        module = import_parser_module(parser_path)
        if not hasattr(module, "parse"):
            return False, "Validation error: Parser is missing the `parse` function."

        parsed_df = module.parse(str(pdf_path))
        expected_df = pd.read_csv(csv_path, dtype=str).fillna("")
        
        parsed_df = parsed_df.astype(str).fillna("")
        parsed_df = parsed_df.replace("<NA>", "").replace("nan", "")

        is_equal = parsed_df.reset_index(drop=True).equals(expected_df.reset_index(drop=True))
        if is_equal:
            return True, None
        else:
            error_report = f"""
Validation error: The parsed DataFrame does not match the expected CSV.

- Parsed Shape: {parsed_df.shape} vs Expected: {expected_df.shape}

First 5 rows of Parsed DataFrame:
{parsed_df.head().to_string()}

First 5 rows of Expected DataFrame:
{expected_df.head().to_string()}
"""
            return False, error_report
    except Exception as e:
        return False, f"Validation error: An exception occurred: {e}"

def generate_parser_code(target: str, pdf_path: Path, csv_path: Path, attempt: int, error_message: str | None = None) -> str:
    csv_df = pd.read_csv(csv_path, nrows=5)
    csv_sample = csv_df.to_csv(index=False)
    columns = list(csv_df.columns)

    feedback_prompt = ""
    if error_message:
        feedback_prompt = f"""
The previous attempt failed. Please analyze the following error and fix the code.

Error Details:
{error_message}

Generate the corrected Python code.
"""

    prompt = f"""
Generate only valid Python code (no explanations or markdown).

**Library Rules:**
- You MUST use `pdfplumber` for PDF parsing.
- You MUST use `pandas` for data manipulation.
- **DO NOT** use any other libraries like `camelot`. Using `camelot` will result in failure.

**Requirements:**
- Define a function: `def parse(pdf_path: str) -> pd.DataFrame:`
- The final DataFrame MUST have these exact columns: {columns}

**Logic Steps:**
1. Open the PDF with `pdfplumber`.
2. Loop through ALL pages to extract and combine tables.
3. After combining, filter out all duplicate header rows. A header row is one where the first cell is 'Date'.
4. Create the final pandas DataFrame.
5. Clean the data: ensure any missing values in 'Debit Amt' or 'Credit Amt' are empty strings `""`, not `0` or `NaN`.

- Here is a sample of the target CSV data to match:
{csv_sample}

This is attempt {attempt}.{feedback_prompt}
Return only Python code.
"""

    response = llm.generate_content(prompt)
    code = response.text

    # --- Sanitization Logic ---
    code = code.replace("`", "").replace("python", "").strip()
    code = code.replace(" ", "    ")

    try:
        compile(code, "<parser>", "exec")
    except SyntaxError as e:
        logger.warning(f"Gemini produced invalid Python: {e}")
        code = ""
    return code

def write_parser_file(target: str, code: str) -> Path:
    path = PARSERS_DIR / f"{target}_parser.py"
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    return path

# --- LangGraph Nodes ---

def plan_node(state: dict):
    new_state = dict(state)
    logger.info("Planning…")
    new_state["attempt"] = new_state.get("attempt", 0) + 1
    return new_state

def codegen_node(state: dict):
    new_state = dict(state)
    target = new_state["target"]
    pdf_path, csv_path = new_state["files"]
    attempt = new_state["attempt"]
    error_message = new_state.get("error_message")

    logger.info(f"Generating parser for {target}, attempt {attempt}")
    code = generate_parser_code(target, pdf_path, csv_path, attempt, error_message)
    parser_path = write_parser_file(target, code)
    new_state["parser_path"] = parser_path
    return new_state

def test_node(state: dict):
    new_state = dict(state)
    parser_path = new_state["parser_path"]
    pdf_path, csv_path = new_state["files"]

    new_state["error_message"] = None
    ok, error = validate_parser(parser_path, pdf_path, csv_path)

    new_state["success"] = ok
    if not ok:
        logger.warning(f"Validation failed for attempt {new_state['attempt']}.")
        new_state["error_message"] = error

    return new_state

def decide_node(state: dict):
    if state.get("success"):
        logger.info("Parser validated successfully ✅")
        return END
    elif state.get("attempt", 0) >= 3:
        logger.error("All attempts failed ❌")
        return END
    else:
        logger.info("Retrying with another attempt…")
        return "plan"

# --- Build and Compile Graph ---

graph = StateGraph(dict)
graph.add_node("plan", plan_node)
graph.add_node("codegen", codegen_node)
graph.add_node("test", test_node)
graph.set_entry_point("plan")

graph.add_edge("plan", "codegen")
graph.add_edge("codegen", "test")
graph.add_conditional_edges("test", decide_node)

app = graph.compile()

# --- CLI Entrypoint ---

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="Bank target folder under data/")
    args = ap.parse_args()

    target = args.target.lower()
    pdf, csv = find_sample_files(target)

    init_state = {
        "target": target,
        "files": (pdf, csv),
        "attempt": 0,
        "error_message": None
    }

    final_state = app.invoke(init_state)

    if not final_state.get("success"):
        sys.exit(1)

if __name__ == "__main__":
    main()