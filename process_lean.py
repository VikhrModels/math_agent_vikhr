import json
import re
import logging
from pathlib import Path

# Import configuration
from config import (
    LEAN_SOURCE_FILE,
    LEAN_OUTPUT_FILE,
    LOG_FORMAT,
    validate_config
)

# --- Logging Setup ---
# Configures basic logging to provide visibility into the script's operations.
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT
)
logger = logging.getLogger(__name__)

def read_lean_file(filepath: Path) -> str:
    """Reads the contents of a Lean file."""
    logger.info(f"Reading Lean file: {filepath}")
    try:
        with filepath.open('r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"Error: File not found at {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error reading file {filepath}: {e}")
        raise

def process_lean_theorems(lean_content: str) -> list[dict]:
    """
    Parses a string of Lean code to extract theorem and lemma statements.

    This function uses regular expressions to find all declarations (like theorem,
    lemma, def, etc.), extracts their names and full statements, and determines
    if they have a complete proof or are left with `sorry`.
    Solved proofs are replaced with `sorry` to create a uniform set of problems
    for the language model.

    Returns a list of dictionaries, where each dictionary represents a theorem.
    """
    logger.info("Processing theorems from Lean content.")
    theorems = []

    # This regular expression is designed to find and capture blocks of code
    # that define a theorem, lemma, definition, etc. It captures the full
    # statement and the name of the declaration.
    # It looks for a declaration keyword and captures until it finds the next
    # one or the end of the file.
    theorem_blocks_pattern = re.compile(
        r'(?m)^'  # Start of line (multiline mode)
        r'(?P<full_statement>'  # 'full_statement' group captures the entire declaration
        r'(?:theorem|lemma|def|example|instance|abbrev)\s+'  # Declaration keyword
        r'(?P<name>[a-zA-Z0-9_.]+)\s*'  # Declaration name (group 'name')
        r'.*?)'  # Non-greedy capture of any content until next declaration or end of file
        r'(?=\n*(?:theorem|lemma|def|example|instance|abbrev)|\Z)',  # Lookahead: until next declaration or end of file
        re.DOTALL
    )

    for match in theorem_blocks_pattern.finditer(lean_content):
        full_statement = match.group('full_statement').strip()
        name = match.group('name')

        # After finding a declaration, we check if it has a proof.
        # A proof is typically enclosed in a `begin...end` block.
        begin_end_pattern = re.compile(r'begin\s*(.*?)\s*end', re.DOTALL)
        begin_end_match = begin_end_pattern.search(full_statement)
        
        is_solved = False
        if begin_end_match:
            proof_content = begin_end_match.group(1).strip()
            # A proof is considered "solved" if the block is not empty
            # and does not contain `sorry`.
            if proof_content and 'sorry' not in proof_content:
                is_solved = True
                # To create a consistent problem format, we replace the
                # existing proof with a `sorry` block.
                full_statement = begin_end_pattern.sub('begin\n  sorry\nend', full_statement)

        theorems.append({
            'name': name,
            'statement': full_statement,
            'is_solved': is_solved
        })

    logger.info(f"Found {len(theorems)} theorems in total.")
    return theorems

def main():
    """
    Main function to execute the script.

    It defines the input Lean file and the output JSON file, then orchestrates
    the process of reading the Lean file, processing its content to extract
    theorems, and writing the results to the JSON file.
    """
    # Validate configuration
    validate_config()
    
    try:
        # Read file contents
        lean_content = read_lean_file(LEAN_SOURCE_FILE)
        
        # Process theorems
        theorems = process_lean_theorems(lean_content)
        
        # Save result to JSON
        with LEAN_OUTPUT_FILE.open('w', encoding='utf-8') as f:
            json.dump(theorems, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Successfully processed {len(theorems)} theorems and saved to {LEAN_OUTPUT_FILE}")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main() 